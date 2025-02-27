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
Concrete subsystem operators.
"""

from .abstract_subsystem_operators import AbstractSubsystemOperator

import numpy as np


class SubsystemOperator(AbstractSubsystemOperator):

    def __init__(self, matrix, subsystems, str_label=None):
        if matrix.shape[0] != np.prod([s.dim for s in subsystems]):
            raise Exception("Subsystem dimensions don't match matrix shape.")

        self._matrix = matrix
        self._str_label = str_label
        super().__init__(subsystems)
    
    def base_matrix(self):
        return self._matrix
    
    def __str__(self):
        str_label = self._str_label or "SubsystemOperator"
        
        subsystem_str = str(self.subsystems[0])
        for s in self.subsystems[1:]:
            subsystem_str += f", {s}"
        return f"{str_label}({subsystem_str})"


class A(AbstractSubsystemOperator):
    """Annihilation operator."""

    def base_matrix(self):
        return np.diag(np.sqrt(np.arange(1, self.subsystems[0].dim, dtype=complex)), 1)

    def __str__(self):
        return f"a({self.subsystems[0]})"


class Adag(AbstractSubsystemOperator):
    """Creation operator."""

    def base_matrix(self):
        return np.diag(np.sqrt(np.arange(1, self.subsystems[0].dim, dtype=complex)), -1)

    def __str__(self):
        return f"adag({self.subsystems[0]})"


class N(AbstractSubsystemOperator):
    """Number operator."""

    def base_matrix(self):
        return np.diag(np.arange(self.subsystems[0].dim, dtype=complex))

    def __str__(self):
        return f"N({self.subsystems[0]})"


class I(AbstractSubsystemOperator):  # noqa: E742
    """identity."""

    def base_matrix(self):
        return np.eye(self.subsystems[0].dim, dtype=complex)

    def __str__(self):
        return f"I({self.subsystems[0]})"


class X(AbstractSubsystemOperator):
    """X operator. For higher dimensions, generalizes to A + Adag."""

    def base_matrix(self):
        return A(self.subsystems).base_matrix() + Adag(self.subsystems).base_matrix()

    def __str__(self):
        return f"X({self.subsystems[0]})"


class Y(AbstractSubsystemOperator):
    """Y operator.For higher dimensions, generalizes to -1j * (A - Adag)."""

    def base_matrix(self):
        return -1j * (A(self.subsystems).base_matrix() + (-1 * Adag(self.subsystems).base_matrix()))

    def __str__(self):
        return f"Y({self.subsystems[0]})"


class Z(AbstractSubsystemOperator):
    """Z operator.

    To do:
        - Implement higher dimensional versions in terms of a/adagger.
    """

    def base_matrix(self):
        return I(self.subsystems).base_matrix() - 2 * N(self.subsystems).base_matrix()

    def __str__(self):
        return f"Z({self.subsystems[0]})"
