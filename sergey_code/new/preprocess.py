import numpy as np
import skimage as si


def process_mask_arr(mask_arr, resize_target_shape, task):
    mask_arr = si.util.img_as_ubyte(mask_arr[..., :3])
    mask_arr = np.where(mask_arr > 100, 255, 0).astype(np.uint8)
    mask_arr = si.transform.resize(
        mask_arr,
        resize_target_shape,
        order=0,
        anti_aliasing=False,
    )
    mask_arr = si.util.img_as_ubyte(mask_arr)
    if task.name == "dangerous_safe":
        dangerous_mask = mask_arr[..., 0] == 255
        safe_mask = mask_arr[..., 1] == 255
        mask_arr[dangerous_mask & safe_mask, 1] = 0
        mask_arr[..., 2] = np.where(dangerous_mask | safe_mask, 0, 255)
    elif task.name == "anatomy":
        mask_arr_ = np.zeros_like(mask_arr, shape=(*mask_arr.shape[:2], 4))
        mask_arr_[..., :3] = mask_arr
        mask_arr_[..., 3] = np.where(np.all(mask_arr == 0, axis=2), 255, 0)
        mask_arr = mask_arr_
    elif task.name == "IEoG_LoS":
        pass
    else:
        raise ValueError(f"Unknown {task=}")
    if (mask_shape := mask_arr.shape[2]) != len(task.class_names):
        raise RuntimeError(f"For {task=}, {mask_shape=} shows wrong amount of channels")
    return mask_arr / mask_arr.sum(axis=2, keepdims=True)
