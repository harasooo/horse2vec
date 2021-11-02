from collections import abc
from typing import List, Generator
import numpy as np


def flatten(nested_list: List[List[int]]) -> Generator:
    for el in nested_list:
        if isinstance(el, abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def rank_order(image):
    flat_image = image.ravel()
    sort_order = flat_image.argsort()
    flat_image = flat_image[sort_order]
    sort_rank = np.zeros_like(sort_order)
    is_different = flat_image[:-1] != flat_image[1:]
    np.cumsum(is_different, out=sort_rank[1:])
    original_values = np.zeros((sort_rank[-1] + 1,), image.dtype)
    original_values[0] = flat_image[0]
    original_values[1:] = flat_image[1:][is_different]
    int_image = np.zeros_like(sort_order)
    int_image[sort_order] = sort_rank
    return int_image.reshape(image.shape)
