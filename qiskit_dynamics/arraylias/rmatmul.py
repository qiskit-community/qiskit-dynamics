import numpy as np


def register_rmatmul(alias):
    @alias.register_function(lib="numpy", path="rmatmul")
    def _(x, y):
        return np.matmul(y, x)

    @alias.register_function(lib="scipy_sparse", path="rmatmul")
    def _(x, y):
        return y * x

    try:
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        jsparse_matmul = jsparse.sparsify(jnp.matmul)

        @alias.register_function(lib="jax", path="rmatmul")
        def _(x, y):
            return jnp.matmul(y, x)

        @alias.register_function(lib="jax_sparse", path="rmatmul")
        def _(x, y):
            return jsparse_matmul(y, x)

    except:
        pass
