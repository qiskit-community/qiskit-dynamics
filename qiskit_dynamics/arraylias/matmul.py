def register_matmul(alias):
    @alias.register_function(lib="scipy_sparse", path="matmul")
    def _(x, y):
        return x * y

    try:
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        jsparse_matmul = jsparse.sparsify(jnp.matmul)

        @alias.register_function(lib="jax_sparse", path="matmul")
        def _(x, y):
            return jsparse_matmul(x, y)

    except:
        pass
