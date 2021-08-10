# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,no-member,attribute-defined-outside-init
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

# TODO: Support the following operators at arbitrary dimensions:
# 		i zero, projectors?
# 		x y z sp sm 0 1 + - r l, for spins
# 		n p q a a_ n n^2, for oscillators and


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


class PlusZ(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for Up (|0><0|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '0')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '0':
			return np.asarray([[1, 0], [0, 0]], complex)
		super().get_operator_matrix(dim)


class MinusZ(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for Down (|1><1|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '1')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '1':
			return np.asarray([[0, 0], [0, 1]], complex)
		super().get_operator_matrix(dim)


class PlusX(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for plus state (|+><+|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '+')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '+':
			return np.asarray([[0.5, 0.5], [0.5, 0.5]], complex)
		super().get_operator_matrix(dim)


class MinusX(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for minus state (|-><-|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, '-')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == '-':
			return np.asarray([[0.5, -0.5], [-0.5, 0.5]], complex)
		super().get_operator_matrix(dim)


class PlusY(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for right y (|i><i|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'r')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'r':
			return np.asarray([[0.5, -0.5j], [0.5j, 0.5]], complex)
		super().get_operator_matrix(dim)


class MinusY(DynamicalOperator):
	"""A dynamical operator that builds a numpy density matrix for left y (|-i><-i|)."""

	def __init__(self, system_id = ''):
		super().__init__(system_id, 'l')

	def get_operator_matrix(self, dim: int) -> Any:
		"""Returns a matrix describing a realization of the operator specified in the parameters.

		Args:
			dim: The physical dimension of the matrix to generate.
		"""
		if dim == 2 and self.s_type == 'l':
			return np.asarray([[0.5, 0.5j], [-0.5j, 0.5]], complex)
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
