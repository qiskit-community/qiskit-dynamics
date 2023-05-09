def register_multiply(alias):
    @alias.register_function(lib="scipy_sparse", path="multiply")
    def _(x, y):
        return x.multiply(y)

    try:
        from jax.experimental import sparse as jsparse
        import jax.numpy as jnp

        jsparse_multiply = jsparse.sparsify(jnp.multiply)

        @alias.register_function(lib="jax_sparse", path="multiply")
        def _(x, y):
            return jsparse_multiply(x, y)

    except:
        pass
