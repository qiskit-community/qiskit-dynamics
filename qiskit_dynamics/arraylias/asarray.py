import numpy as np


def register_to_asarray(alias):
    @alias.register_function(lib="iterable", path="asarray")
    def _(arr):
        if isinstance(arr[0], (list, tuple)):
            return np.asarray(arr)
        return alias(like=arr[0]).asarray([alias().asarray(sub_arr) for sub_arr in arr])

    @alias.register_fallback(path="asarray")
    def _(arr):
        return np.asarray(arr)
