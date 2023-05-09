from typing import Union
from collections.abc import Iterable
from arraylias import numpy_alias
from scipy.sparse import spmatrix
from qiskit.quantum_info.operators import Operator
from .asarray import register_to_asarray
from .to_dense import register_to_dense
from .to_numeric_matrix_type import register_to_numeric_matrix_type
from .to_sparse import register_to_sparse
from .matmul import register_matmul
from .rmatmul import register_rmatmul
from .multiply import register_multiply

from ..array import Array


DYNAMICS_ALIAS = numpy_alias()

# Set qiskit_dynamics.array.Array to be dispatched to numpy
DYNAMICS_ALIAS.register_type(Array, "numpy")

# register required custom versions of functions for sparse type here
DYNAMICS_ALIAS.register_type(spmatrix, lib="scipy_sparse")

try:
    from jax.experimental.sparse import BCOO

    # register required custom versions of functions for BCOO type here
    DYNAMICS_ALIAS.register_type(BCOO, lib="jax_sparse")
except ImportError:
    pass

# register required custom versions of functions for Operator type here
DYNAMICS_ALIAS.register_type(Operator, lib="operator")

# register required custom versions of functions for Iterable type here
# need to discuss registering Iterable type because the coverage of Iterable is too broad.
DYNAMICS_ALIAS.register_type(Iterable, lib="iterable")


register_to_asarray(alias=DYNAMICS_ALIAS)
register_to_dense(alias=DYNAMICS_ALIAS)
register_to_numeric_matrix_type(alias=DYNAMICS_ALIAS)
register_to_sparse(alias=DYNAMICS_ALIAS)
register_matmul(alias=DYNAMICS_ALIAS)
register_multiply(alias=DYNAMICS_ALIAS)
register_rmatmul(alias=DYNAMICS_ALIAS)

DYNAMICS_NUMPY = DYNAMICS_ALIAS()


ArrayLike = Union[DYNAMICS_ALIAS.registered_types()]
