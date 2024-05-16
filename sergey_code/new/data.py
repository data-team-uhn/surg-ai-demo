import warnings
from functools import cache, cached_property
from collections import namedtuple
import numpy as np
import skimage as si
import torch as t
from torch.utils.data import DataLoader
import lightning as L
from .utils import rescale_to_height
from .preprocess import process_mask_arr


Task = namedtuple("Task", ("name", "type", "class_names", "label_col_name"))

# task name, class_names, label col name
# task type inserted automatically after task name
LAP_CHOLE_SEG_TASKS = [
    ["dangerous_safe", ["dangerous", "safe", "bg"], "label_path"],
    ["anatomy", ["liver", "HT", "gallbladder", "bg"], "label_path_HT"],
    ["IEoG_LoS", ["IEoG", "LoS", "bg"], "label_path_IEoG_LoS"],
]
LAP_CHOLE_CLS_TASKS = [
    # TODO: check retraction class names
    ["retraction", ["dk", "bad", "good", "great"], "retraction"],
    ["IEoG_exposure", ["no", "yes"], "exposure_ieg"],
    ["LoS_exposure", ["no", "yes"], "exposure_los"],
]

for task_group, task_type in zip(
    (LAP_CHOLE_SEG_TASKS, LAP_CHOLE_CLS_TASKS), ("segmentation", "classification")
):
    for i, task in enumerate(task_group):
        task.insert(1, task_type)
        task_group[i] = Task(*task)

LAP_CHOLE_TASKS = LAP_CHOLE_SEG_TASKS + LAP_CHOLE_CLS_TASKS
if len(set(task.name for task in LAP_CHOLE_TASKS)) != len(LAP_CHOLE_TASKS):
    raise RuntimeError(f"For {LAP_CHOLE_TASKS=} task names are not unique")
LAP_CHOLE_MASK_PADDING_VALS = []
for task in LAP_CHOLE_SEG_TASKS:
    padding_val = [0 if class_name != "bg" else 1 for class_name in task.class_names]
    LAP_CHOLE_MASK_PADDING_VALS.extend(padding_val)


# HACK: none of the augmentation libraries seems to be able
# to deal with multiple masks for the same image in a manner
# suitable for us. For example, Torchvision transforms v2 and
# Albumentations can't have different padding values for different
# masks, Kornia can't process images and masks in the same
# manner simultaneously. Therefore, "true" padding will happen
# in __getitem__ method of this class, whereas external transforms
# will pad an array of ones for us to show which region corresponds
# to original input and where this input was padded. Having such a
# mask is useful for inference anyway.
class LapCholeDataset(t.utils.data.Dataset):
    # TODO: gamma correction?
    def __init__(
        self,
        metadata_df,
        rescale_target_height,
        geometric_transform,
        pixel_transform,
        standardize,
        # We don't have classification labels for all images for
        # which we have segmentation labels, so if we only do
        # segmentation and want maximal amount of data, this is
        # a legitimate reason to not include classification labels
        drop_classification=False,
    ):
        self.metadata_df = metadata_df
        self.rescale_target_height = rescale_target_height
        self.geometric_transform = geometric_transform
        self.pixel_transform = pixel_transform
        self.standardize = standardize
        self.drop_classification = drop_classification

    def __len__(self):
        return len(self.metadata_df)

    @cache
    def prepare_item(self, idx):
        metadata_row = self.metadata_df.iloc[idx]
        image_arr = si.io.imread(metadata_row["image_path"])[..., :3]
        image_arr = rescale_to_height(image_arr, self.rescale_target_height, 1)
        image_arr = si.util.img_as_float32(image_arr)
        unpadded_region_mask_arr = np.ones(image_arr.shape[:2], dtype=np.float32)
        mask_arrs = []
        for task in LAP_CHOLE_SEG_TASKS:
            mask_arr = si.io.imread(metadata_row[task.label_col_name])
            mask_arr = process_mask_arr(mask_arr, image_arr.shape[:2], task)
            if (mask_shape := mask_arr.shape[2]) != len(task.class_names):
                raise RuntimeError(
                    f"For {task=}, {mask_shape=} shows wrong amount of channels"
                )
            mask_arr = si.util.img_as_float32(mask_arr)
            mask_arrs.append(mask_arr)
        mask_arr = np.concatenate(mask_arrs, axis=-1)
        return_vals = [
            image_arr,
            unpadded_region_mask_arr,
            mask_arr,
        ]
        if not self.drop_classification:
            class_arr = (
                metadata_row[[task.label_col_name for task in LAP_CHOLE_CLS_TASKS]]
                .to_numpy()
                .astype(np.int64)
            )
            return_vals.append(class_arr)
        return return_vals

    def __getitem__(self, idx):
        (
            image_arr,
            unpadded_region_mask_arr,
            mask_arr,
            *class_arr,
        ) = self.prepare_item(idx)
        if class_arr:
            class_arr = class_arr[0]
        if self.geometric_transform is not None:
            transformed = self.geometric_transform(
                image=image_arr,
                unpadded_region_mask=unpadded_region_mask_arr,
                mask=mask_arr,
            )
            image_arr = transformed["image"]
            unpadded_region_mask_arr = transformed["unpadded_region_mask"]
            unpadded_region_mask_arr = unpadded_region_mask_arr > 0.5
            if transformed["mask"] is mask_arr:
                raise RuntimeError(
                    'transformed["mask"] is mask_arr, we can\'t safely proceed with padding it inplace!'
                )
            mask_arr = transformed["mask"]
            # See HACK description at the top of this class
            mask_arr[~unpadded_region_mask_arr] = LAP_CHOLE_MASK_PADDING_VALS
        if self.pixel_transform is not None:
            image_arr = self.pixel_transform(image=image_arr)["image"]
        if self.standardize:
            image_arr -= image_arr.mean(axis=(0, 1), keepdims=True)
            image_arr /= image_arr.std(axis=(0, 1), keepdims=True)
        image_ten = t.from_numpy(image_arr).moveaxis(-1, 0)
        unpadded_region_mask_ten = t.from_numpy(unpadded_region_mask_arr)
        mask_tens = []
        prev_end_idx = 0
        for task in LAP_CHOLE_SEG_TASKS:
            cur_end_idx = prev_end_idx + len(task.class_names)
            mask_tens.append(
                t.from_numpy(mask_arr[..., prev_end_idx:cur_end_idx]).moveaxis(-1, 0)
            )
            prev_end_idx = cur_end_idx
        return image_ten, unpadded_region_mask_ten, *mask_tens, *class_arr

    def check_cls_labels(self):
        if self.drop_classification:
            warnings.warn(
                f"{self.drop_classification=}, but check_cls_labels was called"
            )
        tasks_with_problems = []
        for task in LAP_CHOLE_CLS_TASKS:
            label_col = self.metadata_df[task.label_col_name]
            unique_labels = label_col.unique()
            if len(unique_labels) != len(task.class_names) or set(unique_labels) != set(
                range(max(unique_labels) + 1)
            ):
                tasks_with_problems.append(task)
                warnings.warn(
                    f"For {task=}, {unique_labels=} as found in metadata_df has wrong length, isn't 0-based or has holes"
                )
        if tasks_with_problems:
            raise RuntimeError(f"{tasks_with_problems=}")

    @cached_property
    def class_frequencies(self):
        seg_class_counts = [[] for _ in LAP_CHOLE_SEG_TASKS]
        if not self.drop_classification:
            self.check_cls_labels()
            cls_classes = [[] for _ in LAP_CHOLE_CLS_TASKS]
        # TODO: make it configurable
        for i in range(10):
            for item in self:
                seg_mask_tens = item[2 : 2 + len(LAP_CHOLE_SEG_TASKS)]
                for seg_class_counts_, seg_mask_ten in zip(
                    seg_class_counts, seg_mask_tens
                ):
                    seg_class_counts_.append(seg_mask_ten.sum(axis=(1, 2)))
                if not self.drop_classification:
                    cls_class_vals = item[2 + len(LAP_CHOLE_SEG_TASKS) :]
                    for cls_classes_, cls_class in zip(cls_classes, cls_class_vals):
                        cls_classes_.append(cls_class)
        seg_class_freqs = []
        for seg_class_counts_ in seg_class_counts:
            mean_class_counts = t.stack(seg_class_counts_).mean(axis=0)
            class_freqs = mean_class_counts / mean_class_counts.sum()
            seg_class_freqs.append(class_freqs)
        if not self.drop_classification:
            cls_class_freqs = []
            for cls_classes_, task in zip(cls_classes, LAP_CHOLE_CLS_TASKS):
                classes, counts = np.unique(cls_classes_, return_counts=True)
                cls_class_freqs.append(
                    t.from_numpy(counts / counts.sum()).to(t.float32)
                )
        else:
            cls_class_freqs = []
        class_freqs = seg_class_freqs + cls_class_freqs
        return class_freqs


class LapCholeDataModule(L.LightningDataModule):
    # TODO: if we switch to num_workers > 1 in the DataLoaders,
    # we should first populate the Dataset's cache once, so that
    # it is copied into every worker. setup() is probably a good
    # place to do that.
    def __init__(
        self,
        metadata_df,
        train_batch_size,
        val_test_batch_size,
        rescale_target_height,
        train_geometric_transform,
        train_pixel_transform,
        val_test_geometric_transform,
        val_test_pixel_transform,
        standardize,
        drop_classification=False,
    ):
        super().__init__()
        self.metadata_df = metadata_df
        self.train_batch_size = train_batch_size
        self.val_test_batch_size = val_test_batch_size
        self.rescale_target_height = rescale_target_height
        self.train_geometric_transform = train_geometric_transform
        self.train_pixel_transform = train_pixel_transform
        self.val_test_geometric_transform = val_test_geometric_transform
        self.val_test_pixel_transform = val_test_pixel_transform
        self.standardize = standardize
        self.drop_classification = drop_classification

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = LapCholeDataset(
                self.metadata_df[self.metadata_df["subset"] == "train"],
                self.rescale_target_height,
                self.train_geometric_transform,
                self.train_pixel_transform,
                self.standardize,
                self.drop_classification,
            )
            if not self.drop_classification:
                self.train_dataset.check_cls_labels()
            self.val_dataset = LapCholeDataset(
                self.metadata_df[self.metadata_df["subset"] == "valid"],
                self.rescale_target_height,
                self.val_test_geometric_transform,
                self.val_test_pixel_transform,
                self.standardize,
                self.drop_classification,
            )
            print(f"{len(self.train_dataset)=}")
            print(f"{len(self.val_dataset)=}")
            print(f"{len(self.train_dataloader())=}")
            print(f"{len(self.val_dataloader())=}")
        else:
            raise NotImplementedError(f"Code for {stage=} is not implemented")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_test_batch_size,
            shuffle=False,
            pin_memory=True,
        )
