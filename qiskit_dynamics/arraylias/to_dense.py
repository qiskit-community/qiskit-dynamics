import numpy as np


def register_to_dense(alias):
    @alias.register_default(path="to_dense")
    def _(op):
        if op is None:
            return None
        return op

    @alias.register_function(lib="numpy", path="to_dense")
    def _(op):
        return op

    try:

        @alias.register_function(lib="jax", path="to_dense")
        def _(op):
            return op

        @alias.register_function(lib="jax_sparse", path="to_dense")
        def _(op):
            return op.todense()

    except:
        pass

    @alias.register_function(lib="scipy_sparse", path="to_dense")
    def _(op):
        return op.toarray()

    @alias.register_fallback(path="to_dense")
    def _(op):
        return np.asarray(op)

    @alias.register_function(lib="iterable", path="to_dense")
    def _(op):
        return alias().asarray([alias().to_dense(sub_op) for sub_op in op])
