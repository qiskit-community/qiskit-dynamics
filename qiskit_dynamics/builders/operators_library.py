from .dynamical_operators import DynamicalOperator, Id, Zero
from typing import Any
import numpy as np


"""
These are the predefined operators available for creating dynamical system simulations.
Currently implemented are spin operators and truncated harmonic oscillator operators,
in addition to some more general operators, all realizing numpy matrices by default.
The classes ``Id`` and ``Zero`` are implemented separately, in the file dynamical_operators.py
together with the base class ``DynamicalOperator``.
"""

# TODO: Replace qubit creation with Qiskit Operator.from_label().
# TODO: Support the following operators at arbitrary dimensions:
# 		i x y z sp sm n p q a a_ n n^2 0 1 and initial states as in Qiskit


class Sx(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli x matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'x')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'x':
			return np.asarray([[0, 1], [1, 0]], complex)
		super().get_operator_matrix(dim)


class Sy(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli y matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'y')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'y':
			return np.asarray([[0, -1j], [1j, 0]], complex)
		super().get_operator_matrix(dim)


class Sz(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli z matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'z')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'z':
			return np.asarray([[1, 0], [0, -1]], complex)
		super().get_operator_matrix(dim)


class Sp(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli ladder |1><0| matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'sp')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'sp':
			return np.asarray([[0, 1], [0, 0]], complex)
		super().get_operator_matrix(dim)


class Sm(DynamicalOperator):
	"""A dynamical operator that builds a numpy Pauli ladder |0><1| matrix."""
	def __init__(self, system_id = ''):
		super().__init__(system_id, 'sm')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'sm':
			return np.asarray([[0, 0], [1, 0]], complex)
		super().get_operator_matrix(dim)


# class SmSp_SpSm(DynamicalOperator):
# 	def __init__(self, system_id = ''):
# 		super().__init__(system_id, 'smspspsm')
#
# 	def get_operator_matrix(self, dim: int) -> Any:
# 		"""Returns a matrix describing a realization of the operator specified in the parameters.
#
# 		Args:
# 			dim: The physical dimension of the matrix to generate.
# 		"""
# 		if dim == 2 and self.s_type == 'smspspsm':
# 			smsp = np.kron(np.asarray([[0, 0], [1, 0]], complex), np.asarray([[0, 1], [0, 0]], complex))
# 			return smsp + np.transpose(smsp)
# 		raise Exception(
# 			f"Operator type {self.s_type} unknown or unsupported for matrix generation with dimension {dim}.")
