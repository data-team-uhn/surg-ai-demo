from warnings import warn
from collections import defaultdict
import torch as t
from torch import nn
import lightning as L
import torchmetrics as tm
import torchmetrics.classification as tmc
from .utils import cur_func_name
from .data import LAP_CHOLE_TASKS


class LapCholeMultiTaskModule(L.LightningModule):
    def __init__(
        self,
        model_class,
        model_args,
        model_kwargs,
        tasks,
        task_loss_weights,
        initial_lr,
        max_lr,
        min_lr,
        weight_decay,
    ):
        super().__init__()
        self.save_hyperparameters(logger=True)
        self.model_class = model_class
        self.model_args = model_args
        self.model_kwargs = model_kwargs
        self.model = model_class(*model_args, **model_kwargs)
        self.tasks = tasks
        for task, task_type in zip(self.tasks, self.model.task_types):
            if not task.type == task_type:
                raise ValueError(f"For {task=}, the model has invalid {task_type=}")
        self.task_nums = [LAP_CHOLE_TASKS.index(task) for task in self.tasks]
        if not isinstance(task_loss_weights, str):
            self.register_buffer("task_loss_weights", task_loss_weights)
            if len(self.tasks) != len(self.task_loss_weights):
                raise ValueError(
                    f"{len(self.tasks)=} != {len(self.task_loss_weights)=}"
                )
        # https://arxiv.org/abs/2111.10603
        elif task_loss_weights == "RLW":
            self.task_loss_weights = task_loss_weights
        else:
            raise NotImplementedError(
                f"Code for {task_loss_weights=} is not implemented"
            )

        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay

    def setup(self, stage):
        if stage == "fit":
            dm = self.trainer.datamodule
            if dm.drop_classification and any(
                task.type == "classification" for task in self.tasks
            ):
                raise ValueError(
                    f"{dm.drop_classification=}, but one of the {self.tasks=} is classification"
                )
            self.class_weightss = []
            for task_num, task, num_labels in zip(
                self.task_nums, self.tasks, self.model.num_labelss
            ):
                train_class_freqs = dm.train_dataset.class_frequencies[task_num]
                if len(train_class_freqs) != num_labels:
                    raise RuntimeError(
                        f"For {task=}, {len(train_class_freqs)=} in the dataset doesn't match {num_labels=} in the model"
                    )
                if not t.all(train_class_freqs):
                    raise RuntimeError(
                        f"For {task=}, {train_class_freqs=} has 0-valued elements"
                    )
                class_weights = 1 / train_class_freqs
                self.class_weightss.append(class_weights / class_weights.sum())
                self.register_buffer(
                    f"class_weights_{task.name}", self.class_weightss[-1]
                )
            print(f"{self.class_weightss=}")
            self.loss_funs = nn.ModuleList(
                [
                    t.nn.CrossEntropyLoss(weight=class_weights)
                    for class_weights in self.class_weightss
                ]
            )
            self.metric_ignore_index = -1
            self.val_metrics_funs = nn.ModuleList(
                [
                    tm.MetricCollection(
                        {
                            "cm_normalize_all": tmc.MulticlassConfusionMatrix(
                                num_labels,
                                ignore_index=self.metric_ignore_index,
                                normalize="all",
                            ),
                            # Average of recalls for every class yields balanced accuracy
                            "recall_average_macro": tmc.MulticlassRecall(
                                num_labels,
                                average="macro",
                                ignore_index=self.metric_ignore_index,
                            ),
                            "recall_average_none": tmc.MulticlassRecall(
                                num_labels,
                                average="none",
                                ignore_index=self.metric_ignore_index,
                            ),
                            "precision_average_macro": tmc.MulticlassPrecision(
                                num_labels,
                                average="macro",
                                ignore_index=self.metric_ignore_index,
                            ),
                            "precision_average_none": tmc.MulticlassPrecision(
                                num_labels,
                                average="none",
                                ignore_index=self.metric_ignore_index,
                            ),
                            "f1_average_macro": tmc.MulticlassF1Score(
                                num_labels,
                                average="macro",
                                ignore_index=self.metric_ignore_index,
                            ),
                            "f1_average_none": tmc.MulticlassF1Score(
                                num_labels,
                                average="none",
                                ignore_index=self.metric_ignore_index,
                            ),
                        },
                    )
                    for num_labels in self.model.num_labelss
                ]
            )
        elif stage == "predict":
            pass
        else:
            raise NotImplementedError(f"Code for {stage=} is not implemented")

    def training_step(self, batch, batch_idx):
        input, unpadded_region, *targets = batch
        outputs = self.model(input)
        task_losses = []
        for task_num, task, loss_fun, output in zip(
            self.task_nums, self.tasks, self.loss_funs, outputs
        ):
            target = targets[task_num]
            loss = loss_fun(output, target)
            task_losses.append(loss)
            self.log(
                f"train_{task.name}_loss",
                loss,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
        task_losses = t.stack(task_losses)
        if not isinstance(self.task_loss_weights, str):
            task_loss_weights = self.task_loss_weights
        elif self.task_loss_weights == "RLW":
            task_loss_weights = nn.functional.softmax(t.randn_like(task_losses), 0)
        else:
            raise NotImplementedError(
                f"Code for {self.task_loss_weights=} is not implemented"
            )
        total_loss = (task_losses * task_loss_weights).sum()
        self.log(
            "train_total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        return total_loss

    def validation_step(self, batch, batch_idx):
        sanity_check = self.trainer.state.stage == "sanity_check"
        input, unpadded_region, *targets = batch
        outputs = self.model(input)
        task_losses = []
        metric_task_means = defaultdict(list)
        for task_num, task, loss_fun, metrics_fun, output in zip(
            self.task_nums, self.tasks, self.loss_funs, self.val_metrics_funs, outputs
        ):
            target = targets[task_num]
            loss = loss_fun(output, target)
            task_losses.append(loss)
            if not sanity_check:
                self.log(
                    f"val_{task.name}_loss",
                    loss,
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
            if task.type == "segmentation":
                target = target.argmax(dim=1)
                target[~unpadded_region] = self.metric_ignore_index
            metrics = metrics_fun(output, target)
            if not sanity_check:
                for metric_name, metric in metrics.items():
                    metric_fun = metrics_fun[metric_name]
                    metric_name_ = metric_name.split("_")[0]
                    if isinstance(metric_fun, tmc.MulticlassConfusionMatrix):
                        for true_class_num in range(metric.shape[0]):
                            true_class_name = task.class_names[true_class_num]
                            for pred_class_num in range(metric.shape[1]):
                                pred_class_name = task.class_names[pred_class_num]
                                self.log(
                                    f"val_{task.name}_true_{true_class_name}_pred_{pred_class_name}_cm",
                                    metric[true_class_num, pred_class_num].item(),
                                    on_step=False,
                                    on_epoch=True,
                                    logger=True,
                                )
                    elif isinstance(
                        metric_fun,
                        (
                            tmc.MulticlassRecall,
                            tmc.MulticlassPrecision,
                            tmc.MulticlassF1Score,
                        ),
                    ):
                        if metric_fun.average == "macro":
                            metric_task_means[metric_name_].append(metric.item())
                            self.log(
                                f"val_{task.name}_mean_{metric_name_}",
                                metric.item(),
                                on_step=False,
                                on_epoch=True,
                                logger=True,
                            )
                        elif metric_fun.average == "none":
                            for class_num, metric_ in enumerate(metric):
                                class_name = task.class_names[class_num]
                                self.log(
                                    f"val_{task.name}_{class_name}_{metric_name_}",
                                    metric_.item(),
                                    on_step=False,
                                    on_epoch=True,
                                    logger=True,
                                )
                        else:
                            raise NotImplementedError(
                                f"Code for logging metric {metric_name} is not implemented"
                            )
                    else:
                        raise NotImplementedError(
                            f"Code for logging metric {metric_name} is not implemented"
                        )
        task_losses = t.stack(task_losses)
        if not isinstance(self.task_loss_weights, str):
            task_loss_weights = self.task_loss_weights
        elif self.task_loss_weights == "RLW":
            warn(
                f"In {cur_func_name()=}: {self.task_loss_weights=}, will use uniform weights for task losses"
            )
            task_loss_weights = t.ones_like(task_losses)
            task_loss_weights /= task_loss_weights.shape[0]
        else:
            raise NotImplementedError(
                f"Code for {self.task_loss_weights=} is not implemented"
            )
        total_loss = (task_losses * task_loss_weights).sum()
        if not sanity_check:
            self.log(
                "val_total_loss",
                total_loss,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            for metric_name, task_means in metric_task_means.items():
                self.log(
                    f"val_mean_mean_{metric_name}",
                    t.tensor(task_means).mean(),
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )

    def predict_step(self, batch, batch_idx):
        input, *_ = batch
        outputs = self.model(input)
        outputs = [nn.functional.softmax(output, 1) for output in outputs]
        return outputs

    def configure_optimizers(self):
        optimizer = t.optim.AdamW(
            self.model.parameters(), lr=self.initial_lr, weight_decay=self.weight_decay
        )
        lr_scheduler = t.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.max_lr,
            epochs=self.trainer.max_epochs,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            div_factor=self.max_lr / self.initial_lr,
            final_div_factor=self.initial_lr / self.min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }
