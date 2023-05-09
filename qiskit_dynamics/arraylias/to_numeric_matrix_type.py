from scipy.sparse import spmatrix


def register_to_numeric_matrix_type(alias):
    @alias.register_default(path="to_numeric_matrix_type")
    def _(op):
        return None

    @alias.register_function(lib="numpy", path="to_numeric_matrix_type")
    def _(op):
        return op

    try:

        @alias.register_function(lib="jax", path="to_numeric_matrix_type")
        def _(op):
            return op

        @alias.register_function(lib="jax_sparse", path="to_numeric_matrix_type")
        def _(op):
            return op

    except:
        pass

    @alias.register_function(lib="scipy_sparse", path="to_numeric_matrix_type")
    def _(op):
        return op

    @alias.register_function(lib="iterable", path="to_numeric_matrix_type")
    def _(op):
        if isinstance(op[0], spmatrix):
            return [alias().to_sparse(sub_op) for sub_op in op]
        return alias().asarray([alias().to_dense(sub_op) for sub_op in op])

    @alias.register_fallback(path="to_numeric_matrix_type")
    def _(op):
        return op
