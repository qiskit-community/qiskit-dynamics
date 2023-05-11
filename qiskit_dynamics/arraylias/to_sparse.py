import numpy as np
from scipy.sparse import csr_matrix
from .utils import isinstance_qutip_qobj


def register_to_sparse(alias):
    @alias.register_default(path="to_sparse")
    def _(op):
        if op is None:
            return None
        if isinstance_qutip_qobj(op):
            return op.data
        return op

    @alias.register_function(lib="numpy", path="to_sparse")
    def _(op):
        if op.ndim < 3:
            return csr_matrix(op)
        return np.array([csr_matrix(sub_op) for sub_op in op])

    try:
        from jax.experimental.sparse import BCOO

        @alias.register_function(lib="jax", path="to_sparse")
        def _(op):
            return BCOO.fromdense(op)

        @alias.register_function(lib="jax_sparse", path="to_sparse")
        def _(op):
            return op

    except ImportError:
        pass

    @alias.register_function(lib="scipy_sparse", path="to_sparse")
    def _(op):
        return op

    @alias.register_fallback(path="to_sparse")
    def _(op):
        return csr_matrix(op)

    @alias.register_function(lib="iterable", path="to_sparse")
    def _(op):
        try:
            import jax.numpy as jnp

            if isinstance(op[0], jnp.ndarray):
                return BCOO.fromdense(op)
        except ImportError:
            return np.array([csr_matrix(sub_op) for sub_op in op])
        return np.array([csr_matrix(sub_op) for sub_op in op])
