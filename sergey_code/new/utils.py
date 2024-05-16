import inspect
from functools import reduce
from collections.abc import Iterator
import pandas as pd
import skimage as si


def rescale_to_height(arr, target_height, resize_order, ensure_even_sizes=True):
    # ensure_even_sizes == True is consistent with ffmpeg
    if ensure_even_sizes and target_height % 2 != 0:
        raise ValueError(f"{ensure_even_sizes=}, but {target_height=} is not even")

    if arr.shape[0] == target_height and (
        not ensure_even_sizes or arr.shape[1] % 2 == 0
    ):
        return arr.copy()
    rescaling_factor = target_height / arr.shape[0]
    # Using int() rather than round() for consistency with ffmpeg
    target_width = int(arr.shape[1] * rescaling_factor)
    if ensure_even_sizes and target_width % 2 != 0:
        target_width += 1
    if resize_order == 0:
        anti_aliasing = False
    else:
        anti_aliasing = None
    arr = si.transform.resize(
        arr,
        (target_height, target_width),
        order=resize_order,
        anti_aliasing=anti_aliasing,
    )
    return arr


def pd_merge_from(dfs):
    if not isinstance(dfs, Iterator):
        dfs = iter(dfs)
    initial = next(dfs)
    return reduce(pd.merge, dfs, initial)


def cur_func_name():
    return inspect.stack()[1][3]
