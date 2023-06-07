import numpy as np
from scipy.sparse import csr_matrix, issparse


def register_to_asarray(alias):
    @alias.register_function(lib="scipy_sparse", path="asarray")
    def _(arr):
        if issparse(arr):
            return arr
        return csr_matrix(arr)

    @alias.register_function(lib="list", path="asarray")
    def _(arr):
        if len(arr) == 0 or isinstance(arr[0], (list, tuple)):
            return np.asarray(arr)
        return alias(like=arr[0]).asarray([alias().asarray(sub_arr) for sub_arr in arr])

    @alias.register_fallback(path="asarray")
    def _(arr):
        return np.asarray(arr)

    try:
        from jax.experimental.sparse import BCOO

        @alias.register_function(lib="jax_sparse", path="asarray")
        def _(arr):
            if type(arr).__name__ == "BCOO":
                return arr
            return BCOO.fromdense(arr)

    except ImportError:
        pass
