import warnings
import sys
import os
from os import path
from os.path import join
import argparse
import json
import pandas as pd
import cv2
import torch as t
import albumentations
import lightning as L
from transformers import (
    SegformerConfig,
)
from .utils import pd_merge_from
from .data import LapCholeDataModule, LAP_CHOLE_TASKS
from .models import (
    SegformerForMultiTaskSemanticSegmentationAndImageClassification,
    SEGFORMER_MODEL_VARIANT_CONFIG_OVERRIDES,
)
from .modules import LapCholeMultiTaskModule


def get_task_by_name(task_name):
    task_names = []
    for task in LAP_CHOLE_TASKS:
        if task.name == task_name:
            return task
        task_names.append(task.name)
    raise ValueError(
        f"Task with {task_name=} not found. Valid task names: {task_names}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("job_dir")
    parser.add_argument("data_dir")
    parser.add_argument("task_names", nargs="+")
    parser.add_argument("--task_loss_weights", nargs="+", default=["uniform"])
    parser.add_argument("--rescale_target_height", type=int, default=128)
    parser.add_argument("--pad_to_shape", type=int, nargs=2, default=(128, 288))
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_test_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--initial_lr", type=float, default=4 * 1e-4)
    parser.add_argument("--max_lr", type=float, default=1e-2)
    parser.add_argument("--min_lr", type=float, default=4 * 1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--segformer_model_variant", default="MiT-b0")
    parser.add_argument("--pretrained_checkpoint_path")
    parser.add_argument("--freeze_encoder", action="store_true")
    args = parser.parse_args()

    args_json = json.dumps(vars(args))
    print(f"{args_json=}")

    if not path.exists(args.job_dir):
        os.mkdir(args.job_dir)
    else:
        warnings.warn(f"{args.job_dir=} already exists and will be re-used")

    with open(
        path.join(args.job_dir, "cl_args.json"), "w", encoding="utf-8"
    ) as out_file:
        out_file.write(args_json + "\n")

    if len(set(args.task_names)) != len(args.task_names):
        raise ValueError(f"{args.task_names=} includes duplicates")

    if args.task_loss_weights[0] == "uniform":
        if len(args.task_loss_weights) > 1:
            raise ValueError(
                f"{args.task_loss_weights=}, given its 0-th value, is expected to have length of 1"
            )
        args.task_loss_weights = [1.0 for _ in args.task_names]
    # https://arxiv.org/abs/2111.10603
    elif args.task_loss_weights[0] == "RLW":
        if len(args.task_loss_weights) > 1:
            raise ValueError(
                f"{args.task_loss_weights=}, given its 0-th value, is expected to have length of 1"
            )
        args.task_loss_weights = args.task_loss_weights[0]
    else:
        if len(args.task_loss_weights) != len(args.task_names):
            raise ValueError(
                f"{len(args.task_loss_weights)=} != {len(args.task_names)=}"
            )
        args.task_loss_weights = [float(weight) for weight in args.task_loss_weights]
    print(f"{args.task_loss_weights=}")

    if isinstance(args.task_loss_weights, list):
        task_loss_weights = t.tensor(args.task_loss_weights)
        task_loss_weights = task_loss_weights / task_loss_weights.sum()
    else:
        task_loss_weights = args.task_loss_weights
    print(f"{task_loss_weights=}")

    tasks = [get_task_by_name(task_name) for task_name in args.task_names]
    print(f"{tasks=}")
    if not any(task.type == "classification" for task in tasks):
        # We don't have classification labels for all images, so dropping
        # classification labels increases the amount of images we can use
        print("None of the tasks is classification, will drop classification labels")
        only_segmentation = True
    else:
        # Geometric augmentations can change classification labels
        # unpredictably, so they should be disabled if we have
        # classification tasks
        print(
            "One of the tasks is classification, will disable geometric augmentations"
        )
        only_segmentation = False

    csv_names = [
        "images_labels.csv",
        "images_videos.csv",
        "videos_subsets.csv",
    ]
    if not only_segmentation:
        csv_names.append("retraction_exposure.csv")
    metadata_df = pd_merge_from(
        map(
            pd.read_csv,
            (join(args.data_dir, csv_name) for csv_name in csv_names),
        )
    )
    for col in metadata_df.columns:
        metadata_df[col] = metadata_df[col].apply(
            lambda x: (
                join(args.data_dir, x.removeprefix("data/"))
                if isinstance(x, str) and x.startswith("data/")
                else x
            )
        )
    for task in tasks:
        if task.name == "retraction":
            print('Retraction task included, dropping tiny "don\'t know" class')
            metadata_df = metadata_df[metadata_df["retraction"] != 0]
            metadata_df.loc[:, "retraction"] -= 1
            task.class_names.remove("dk")
            break
    print(f"{len(metadata_df)=}")

    train_pixel_augmentations = [
        albumentations.augmentations.transforms.ColorJitter(hue=0.05, p=0.5),
        albumentations.augmentations.transforms.RandomFog(0.1, 0.8, 0.0, p=0.5),
        albumentations.augmentations.transforms.GaussNoise((0.001, 0.01), p=0.5),
    ]
    train_pixel_transform = albumentations.core.composition.Sequential(
        train_pixel_augmentations, p=1
    )
    # mask_value should always be 0.
    # Real mask padding happens inside __getitem__ method of
    # LapCholeDataset class, see HACK description at the top of that class
    if only_segmentation:
        train_geometric_augmentations = [
            albumentations.augmentations.geometric.Rotate(
                (-20, 20),
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=0.5,
            ),
            albumentations.core.composition.OneOf(
                [
                    albumentations.augmentations.crops.transforms.RandomResizedCrop(
                        *args.pad_to_shape,
                        (0.8, 1.0),
                        (1.0, 1.0),
                        p=1,
                    ),
                    albumentations.augmentations.geometric.transforms.Affine(
                        scale=(0.8, 1.0),
                        keep_ratio=True,
                        cval=0,
                        cval_mask=0,
                        mode=cv2.BORDER_CONSTANT,
                        p=1,
                    ),
                ],
                p=0.5,
            ),
        ]
        train_geometric_augmentation = albumentations.core.composition.Sequential(
            train_geometric_augmentations, p=1
        )
    train_padding = albumentations.augmentations.geometric.PadIfNeeded(
        *args.pad_to_shape,
        position="random",
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        mask_value=0,
        always_apply=True,
    )
    train_geometric_transforms = [train_padding]
    if only_segmentation:
        train_geometric_transforms.append(train_geometric_augmentation)
    train_geometric_transform = albumentations.core.composition.Compose(
        train_geometric_transforms,
        additional_targets={"unpadded_region_mask": "mask"},
        p=1,
    )
    val_test_geometric_transform = albumentations.core.composition.Compose(
        [
            albumentations.augmentations.geometric.PadIfNeeded(
                *args.pad_to_shape,
                position="center",
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                always_apply=True,
            )
        ],
        additional_targets={"unpadded_region_mask": "mask"},
        p=1,
    )
    standardize = True

    dm = LapCholeDataModule(
        metadata_df,
        args.train_batch_size,
        args.val_test_batch_size,
        args.rescale_target_height,
        train_geometric_transform,
        train_pixel_transform,
        val_test_geometric_transform,
        None,
        standardize,
        drop_classification=only_segmentation,
    )
    num_labelss = [len(task.class_names) for task in tasks]
    print(f"{num_labelss=}")

    model_class = SegformerForMultiTaskSemanticSegmentationAndImageClassification
    model_variant_config_override = SEGFORMER_MODEL_VARIANT_CONFIG_OVERRIDES[
        args.segformer_model_variant
    ]
    model_config = SegformerConfig(**model_variant_config_override)
    task_config_overrides = []
    for num_labels in num_labelss:
        task_config_overrides.append({"num_labels": num_labels})
    task_types = [task.type for task in tasks]
    model_args = [model_config, task_types, task_config_overrides]

    module = LapCholeMultiTaskModule(
        model_class,
        model_args,
        {},
        tasks,
        task_loss_weights,
        args.initial_lr,
        args.max_lr,
        args.min_lr,
        args.weight_decay,
    )

    if args.pretrained_checkpoint_path is not None:
        print(f"Loading checkpoint from {args.pretrained_checkpoint_path=}")
        ckpt = t.load(args.pretrained_checkpoint_path, map_location="cpu")
        model_state_dict = {
            k: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")
        }
        module.model.load_state_dict(model_state_dict, strict=False)
        print("Successfully loaded model state dict from the checkpoint")

    if args.freeze_encoder:
        if args.pretrained_checkpoint_path is None:
            raise ValueError(
                f"{args.freeze_encoder=}, but {args.pretrained_checkpoint_path=}"
            )
        module.model.segformer.requires_grad_(False)

    checkpoint_callbacks = [
        L.pytorch.callbacks.ModelCheckpoint(
            dirpath=args.job_dir,
            filename="{epoch:03d}_{val_total_loss:.5f}",
            monitor="val_total_loss",
            mode="min",
        ),
    ]
    for metric_name in ("recall", "precision", "f1"):
        checkpoint_callbacks.append(
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=args.job_dir,
                filename=f"{{epoch:03d}}_{{val_mean_mean_{metric_name}:.5f}}",
                monitor=f"val_mean_mean_{metric_name}",
                mode="max",
            )
        )
        if len(args.task_names) > 1:
            for task_name in args.task_names:
                checkpoint_callbacks.append(
                    L.pytorch.callbacks.ModelCheckpoint(
                        dirpath=args.job_dir,
                        filename=f"{{epoch:03d}}_{{val_{task_name}_mean_{metric_name}:.5f}}",
                        monitor=f"val_{task_name}_mean_{metric_name}",
                        mode="max",
                    )
                )

    timer_callback = L.pytorch.callbacks.Timer()

    logger = L.pytorch.loggers.CSVLogger(args.job_dir, flush_logs_every_n_steps=1)

    trainer = L.Trainer(
        logger=logger,
        callbacks=[*checkpoint_callbacks, timer_callback],
        max_epochs=args.epochs,
        log_every_n_steps=1,
        enable_progress_bar=False,
        default_root_dir=args.job_dir,
    )
    trainer.fit(module, datamodule=dm)
    print(f"{timer_callback.state_dict()=}")
    for checkpoint_callback in checkpoint_callbacks:
        print(f"{checkpoint_callback.best_model_path=}")
    metrics_path = logger.experiment.metrics_file_path
    metrics_df = (
        pd.read_csv(metrics_path)
        .sort_values(by=["epoch", "step"])
        .drop(columns=["step"])
    )
    metrics_df = metrics_df[sorted(metrics_df.columns)]
    metrics_df_grouped = metrics_df.groupby("epoch")
    for epoch, epoch_df in metrics_df_grouped:
        if len(epoch_df) > 2:
            raise RuntimeError(
                f"In metrics DataFrame there are {len(epoch_df)=} > 2 rows for {epoch=}"
            )
    metrics_df = metrics_df_grouped.first().sort_index()
    metrics_df.to_csv(metrics_path.removesuffix(".csv") + "_processed.csv")
