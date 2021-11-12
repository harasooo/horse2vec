from argparse import Namespace
from typing import (
    Any,
    Callable,
    Dict,
    MutableMapping,
    Union,
)


def convert_params(params: Union[Dict[str, Any], Namespace]) -> Dict[str, Any]:
    if isinstance(params, Namespace):
        params = vars(params)

    if params is None:
        params = {}

    return params


def flatten_dict(params: Dict[Any, Any], delimiter: str = "/") -> Dict[str, Any]:
    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


def sanitize_callable_params(params: Dict[str, Any]) -> Dict[str, Any]:
    def _sanitize_callable(val):
        if isinstance(val, Callable):
            try:
                _val = val()
                if isinstance(_val, Callable):
                    return val.__name__
                return _val
            except Exception:
                return getattr(val, "__name__", None)
        return val

    return {key: _sanitize_callable(val) for key, val in params.items()}


def transform_log_hyperparams(
    params: Union[Dict[str, Any], Namespace]
) -> Dict[str, Any]:
    params = convert_params(params)
    params = flatten_dict(params)
    params = sanitize_callable_params(params)
    return params
