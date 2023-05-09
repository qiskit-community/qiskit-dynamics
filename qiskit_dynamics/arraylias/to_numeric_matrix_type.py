from scipy.sparse import spmatrix
from qiskit_dynamics.type_utils import isinstance_qutip_qobj


def register_to_numeric_matrix_type(alias):
    @alias.register_default(path="to_numeric_matrix_type")
    def _(op):
        if op is None:
            return None
        if isinstance_qutip_qobj(op):
            return alias().to_sparse(op.data)
        return op

    @alias.register_function(lib="numpy", path="to_numeric_matrix_type")
    def _(op):
        return op

    @alias.register_function(lib="operator", path="to_numeric_matrix_type")
    def _(op):
        return alias().to_dense(op)

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
        if isinstance(op[0], spmatrix) or isinstance_qutip_qobj(op[0]):
            return [alias().to_sparse(sub_op) for sub_op in op]
        return alias().asarray([alias().to_dense(sub_op) for sub_op in op])

    @alias.register_fallback(path="to_numeric_matrix_type")
    def _(op):
        return op
