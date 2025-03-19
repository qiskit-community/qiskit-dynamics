# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Base classes and abstract subsystem operators.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List
from copy import copy

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import ArrayLike

from .subsystem import Subsystem


class AbstractSubsystemOperator(ABC):
    """Abstract base class for operators acting on Subsystems."""

    def __init__(self, subsystems: Union[Subsystem, List[Subsystem]]):
        """Initialize with the list of subsystems the operator acts on."""
        if isinstance(subsystems, Subsystem):
            subsystems = [subsystems]
        self._subsystems = subsystems

    @abstractmethod
    def base_matrix(self):
        """Return the matrix defined on the internal subsystems."""

    @abstractmethod
    def __str__(self):
        """String representation."""

    def __repr__(self):
        return str(self)

    @property
    def subsystems(self) -> List[Subsystem]:
        """Get the subsystems the operator acts on."""
        return self._subsystems

    def matrix(self, ordered_subsystems: Optional[List[Subsystem]] = None):
        """Build the matrix for the operator relative to the ordered subsystems."""

        if ordered_subsystems is None:
            ordered_subsystems = self.subsystems

        if any(subsystem not in ordered_subsystems for subsystem in self.subsystems):
            raise QiskitError("Attempted to build matrix but missing subsystem.")

        return _matrix_implicit_identity(self.base_matrix(), self.subsystems, ordered_subsystems)

    def restrict_subsystems(self, subsystems: List[Subsystem]) -> "AbstractSubsystemOperator":
        """Reduce the operator to the list of subsystems.

        Components of operators with support on removed subsystems will be set to 0.
        """
        if any(subsystem not in subsystems for subsystem in self.subsystems):
            new_subsystems = []
            for subsystem in subsystems:
                if subsystem in self.subsystems:
                    new_subsystems.append(subsystem)

            return ZeroOperator(new_subsystems)

        return self

    def remove_subsystems(self, subsystems: List[Subsystem]) -> "AbstractSubsystemOperator":
        """Return operator with subsystems removed.

        Components of operators with support on removed subsystems will be set to 0.
        """
        if any(x in subsystems for x in self.subsystems):
            new_subsystems = []
            for subsystem in self.subsystems:
                if subsystem not in subsystems:
                    new_subsystems.append(subsystem)

            return ZeroOperator(new_subsystems)

        return self

    def __add__(self, other):
        """Add other to self."""

        if _isscalar(other):
            other = ScalarOperator(other, subsystems=self.subsystems)

        if isinstance(other, ZeroOperator):
            return self

        if isinstance(self, ZeroOperator):
            return other

        return OperatorSum(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __matmul__(self, other):
        """Multiply other with self."""

        if _isscalar(other):
            raise QiskitError("matmul of a scalar with a AbstractSubsystemOperator is not defined.")

        if isinstance(other, ZeroOperator):
            return other

        if isinstance(self, ZeroOperator):
            return self

        return OperatorMatmul(self, other)

    def __rmatmul__(self, other):
        """Could add more validation here."""
        if _isscalar(other):
            raise QiskitError("matmul of a scalar with a AbstractSubsystemOperator is not defined.")

    def __mul__(self, other):
        if isinstance(other, ZeroOperator):
            return other

        if isinstance(self, ZeroOperator):
            return self

        if _isscalar(other):
            return ScalarOperatorProduct(other, self)

        return OperatorMul(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)


class ScalarOperator(AbstractSubsystemOperator):
    """A scalar operator."""

    def __init__(self, value, subsystems):
        """Initialize.

        Args:
            value: The value of the scalar.
            subsystems: List of subsystems.
        """
        self._value = value
        super().__init__(subsystems)

    @property
    def value(self):
        """The scalar value."""
        return self._value

    def base_matrix(self):
        total_dim = int(np.prod([x.dim for x in self.subsystems]))
        return self.value * np.eye(total_dim, dtype=complex)

    def __str__(self):
        return str(self.value)


class ZeroOperator(ScalarOperator):
    """The zero operator."""

    def __init__(self, subsystems):
        """Initialize.

        Args:
            subsystems: List of subsystems.
        """
        super().__init__(value=0.0, subsystems=subsystems)

    def base_matrix(self):
        if len(self.subsystems) == 0:
            raise QiskitError("Attempting to create matrix of zero operator with no subsystems.")
        total_dim = int(np.prod([x.dim for x in self.subsystems]))
        return np.zeros((total_dim, total_dim), dtype=complex)


class CompositeOperator(AbstractSubsystemOperator):
    """An operator built from multiple sub operators."""

    def __init__(self, operators):
        """Initialize.

        Args:
            operators: A list of operators.
        """
        subsystems = copy(operators[0].subsystems)
        for op in operators[1:]:
            for x in op.subsystems:
                if x not in subsystems:
                    subsystems.append(x)

        self._operators = operators
        super().__init__(subsystems)

    @property
    def operators(self):
        """The operators making up the composite operator."""
        return self._operators


class OperatorSum(CompositeOperator):
    """Sum of operators A + B."""

    def __init__(self, a, b):
        """Initialize.

        Args:
            a: An operator.
            b: An operator.
        """
        super().__init__(operators=[a, b])

    def base_matrix(self):
        return self.operators[0].matrix(self.subsystems) + self.operators[1].matrix(self.subsystems)

    def restrict_subsystems(self, subsystems: List[Subsystem]) -> "SubsystemOperator":
        """Reduce the operator to the list of subsystems.

        Components of operators with support on removed subsystems will be set to 0.
        """
        return self.operators[0].restrict_subsystems(subsystems) + self.operators[
            1
        ].restrict_subsystems(subsystems)

    def remove_subsystems(self, subsystems: List[Subsystem]) -> "SubsystemOperator":
        """Return operator with subsystems removed.

        Components of operators with support on removed subsystems will be set to 0.
        """
        return self.operators[0].remove_subsystems(subsystems) + self.operators[
            1
        ].remove_subsystems(subsystems)

    def __str__(self):
        if isinstance(self.operators[0], CompositeOperator):
            op1_string = f"({self.operators[0]})"
        else:
            op1_string = str(self.operators[0])

        if isinstance(self.operators[1], CompositeOperator):
            op2_string = f"({self.operators[1]})"
        else:
            op2_string = str(self.operators[1])
        return f"{op1_string} + {op2_string}"


class OperatorMatmul(CompositeOperator):
    """Matmul of operators A @ B."""

    def __init__(self, a, b):
        super().__init__(operators=[a, b])

    def base_matrix(self):
        return self.operators[0].matrix(self.subsystems) @ self.operators[1].matrix(self.subsystems)

    def __str__(self):
        if isinstance(self.operators[0], CompositeOperator):
            op1_string = f"({self.operators[0]})"
        else:
            op1_string = str(self.operators[0])

        if isinstance(self.operators[1], CompositeOperator):
            op2_string = f"({self.operators[1]})"
        else:
            op2_string = str(self.operators[1])
        return f"{op1_string} @ {op2_string}"


class OperatorMul(CompositeOperator):
    """Mul of operators A * B."""

    def __init__(self, a, b):
        super().__init__(operators=[a, b])

    def base_matrix(self):
        return self.operators[0].matrix(self.subsystems) * self.operators[1].matrix(self.subsystems)

    def __str__(self):
        if isinstance(self.operators[0], CompositeOperator):
            op1_string = f"({self.operators[0]})"
        else:
            op1_string = str(self.operators[0])

        if isinstance(self.operators[1], CompositeOperator):
            op2_string = f"({self.operators[1]})"
        else:
            op2_string = str(self.operators[1])
        return f"{op1_string} * {op2_string}"


class ScalarOperatorProduct(AbstractSubsystemOperator):
    """Product of a scalar and an operator."""

    def __init__(self, scalar, operator):
        """Initialize.

        Args:
            scalar: The scalar.
            operator: The operator.
        """
        self._scalar = scalar
        self._operator = operator
        super().__init__(operator.subsystems)

    @property
    def scalar(self):
        """The scalar."""
        return self._scalar

    def base_matrix(self):
        return self._scalar * self._operator.matrix()

    def __str__(self):
        return f"{self.scalar} * {self._operator}"


class FunctionOperator(AbstractSubsystemOperator):
    """A function applied on an operator. This assumes the output is the same shape/dimension as the
    input.
    """

    def __init__(self, func, operator, func_name=None):
        """Initialize.

        Args:
            func: The function.
            operator: The operator.
            func_name: The name of the function.
        """
        self._func = func
        self._operator = operator
        self._func_name = func_name
        super().__init__(operator.subsystems)

    @property
    def func(self):
        """The function applied to the operator."""
        return self._func

    def base_matrix(self):
        return self.func(self._operator.matrix())

    def __str__(self):
        func_name = self._func_name or "f"
        return f"{func_name}({self._operator})"


def _matrix_implicit_identity(matrix, matrix_subsystems, target_subsystems):
    """Given a matrix defined on a tensor product system described by matrix_subsystems, return the
    definition of the matrix as it acts on target_subsystems in the given order, with implicit
    identity on the subsystems in the complement of matrix_subsystems.
    """
    if any(s not in target_subsystems for s in matrix_subsystems):
        raise QiskitError("matrix_subsystems not all contained in target_subsystems.")

    subsystems_complement = []
    for s in target_subsystems:
        if s not in matrix_subsystems:
            subsystems_complement.append(s)

    if len(subsystems_complement) > 0:
        total_dim = np.prod([s.dim for s in subsystems_complement])
        matrix = unp.kron(matrix, np.eye(total_dim, dtype=matrix.dtype))
        matrix_subsystems = subsystems_complement + matrix_subsystems

    # from now on assume matrix_subsystems has the same elements as target_subsystems
    subsystem_dims = [s.dim for s in matrix_subsystems]
    subsystem_dims.reverse()

    matrix = matrix.reshape(subsystem_dims + subsystem_dims)

    num_subsystems = len(matrix_subsystems)
    matrix_subsystems = list(reversed(matrix_subsystems))
    target_subsystems = list(reversed(target_subsystems))

    reorder = [matrix_subsystems.index(x) for x in target_subsystems]
    reorder = reorder + [x + num_subsystems for x in reorder]

    matrix = matrix.transpose(reorder)
    full_dim = np.prod(subsystem_dims)
    return matrix.reshape((full_dim, full_dim))


def _isscalar(x):
    """Check if x is a scalar."""
    if isinstance(x, ArrayLike):
        try:
            x = unp.asarray(x)
            if x.ndim == 0:
                return True
        # pylint: disable=broad-exception-caught
        except Exception:
            pass
    return False
