from collections import abc
from typing import List, Generator


def flatten(nested_list: List[List[int]]) -> Generator:
    for el in nested_list:
        if isinstance(el, abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el
