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
Mappings on subsystems.
"""

from typing import Optional, Union, List
from qiskit import QiskitError

import numpy as np

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics.systems import Subsystem, quantum_system_model
from .abstract_subsystem_operators import AbstractSubsystemOperator


class SubsystemMapping:
    r"""A linear mapping from a list of subsystems representing a tensor product space to another.

    This class represents a linear map :math:`A : V_1 \otimes \dots \otimes V_n \rightarrow W_1
    \otimes \dots \otimes W_m`, where :math:`A` is specified as a matrix, and the tensor factors of
    both the input and output spaces are given as lists of :class:`Subsystem` instances. The main
    usage is for mapping abstract operators or :class:`QuantumSystemModel` instances: the
    :meth:`.conjugate` method, or simply treating the mapping as ``Callable``, conjugates an
    operator or all operators within the :class:`QuantumSystemModel` by :math:`A`. As usual, for any
    subsystems in the ``in_subsystems`` of the mapping that the operator are not explicitly defined
    on, the operator is assumed to act as the identity.

    See the :ref:`How-to use advanced system modelling functionality <systems modelling userguide>`
    userguide entry for example usage of this class.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        in_subsystems: Union[Subsystem, List[Subsystem]],
        out_subsystems: Optional[Union[Subsystem, List[Subsystem]]] = None,
    ):
        """Initialize.

        Args:
            matrix: The matrix form of the linear map.
            in_subsystems: The list of input subsystems.
            out_subsystems: The list of output subsystems.
        """
        if in_subsystems is None or in_subsystems == []:
            raise QiskitError("in_subsystems cannot be None or [] for SystemMapping.")

        if isinstance(in_subsystems, Subsystem):
            in_subsystems = [in_subsystems]

        if isinstance(out_subsystems, Subsystem):
            out_subsystems = [out_subsystems]
        elif out_subsystems is None:
            out_subsystems = in_subsystems

        self._in_subsystems = in_subsystems
        self._out_subsystems = out_subsystems

        in_dim = np.prod([x.dim for x in in_subsystems])
        out_dim = np.prod([x.dim for x in out_subsystems])

        if matrix.shape != (out_dim, in_dim):
            raise QiskitError("matrix.shape does not match input and output dimensions.")

        self._matrix = matrix
        self._matrix_adj = matrix.conj().transpose()

    @property
    def in_subsystems(self):
        """Subsystems for input to the mapping."""
        return self._in_subsystems

    @property
    def out_subsystems(self):
        """Subsystems for mapping output."""
        return self._out_subsystems

    @property
    def matrix(self):
        """Concrete matrix encoding the action of the mapping."""
        return self._matrix

    def conjugate(
        self, operator: Union[AbstractSubsystemOperator, "quantum_system_model.QuantumSystemModel"]
    ):
        r"""Conjugate a subsystem operator or model.

        Returns a subsystem operator representing :math:`A O A^\dagger`, where :math:`A` is the
        mapping matrix, and :math:`O` is the input operator. If applied to a
        :class:`QuantumSystemModel`, the mapping is applied to all operators in the model.

        Args:
            operator: The operator to be conjugated.
        Returns:
            Union[MappedOperator, QuantumSystemModel]: The conjugated operator or model.
        """

        if isinstance(operator, AbstractSubsystemOperator):
            return MappedOperator(operator, self)
        elif type(operator).__name__ == "QuantumSystemModel":
            return operator._map_model(lambda x: MappedOperator(x, self))

        raise QiskitError(
            f"Input of type {type(operator)} not recognized by SubsystemMapping.conjugate."
        )

    def __call__(
        self, operator: Union[AbstractSubsystemOperator, "quantum_system_model.QuantumSystemModel"]
    ):
        """Apply the conjugation function."""
        return self.conjugate(operator)


class MappedOperator(AbstractSubsystemOperator):
    """An operator mapped by a SystemMapping."""

    def __init__(self, operator, system_mapping):
        """Initialize."""

        self._operator = operator
        self._system_mapping = system_mapping

        # validate that none of the out_subsystems of system_mapping are in the operator definition
        if any(s in system_mapping.out_subsystems for s in operator.subsystems):
            raise QiskitError("Output subsystem found in input operator subsystems.")

        unmapped_subsystems = []
        for s in operator.subsystems:
            if s not in system_mapping.in_subsystems:
                unmapped_subsystems.append(s)

        self._unmapped_subsystems = unmapped_subsystems
        super().__init__(system_mapping.out_subsystems + unmapped_subsystems)

    @property
    def operator(self):
        """The operator being mapped."""
        return self._operator

    @property
    def system_mapping(self):
        """The system mapping being applied to the operator."""
        return self._system_mapping

    @property
    def unmapped_subsystems(self):
        """Subsystems of underlying operator unaffected by mapping."""
        return self._unmapped_subsystems

    def base_matrix(self):
        mat = self._operator.matrix(self.system_mapping.in_subsystems + self.unmapped_subsystems)
        A = self.system_mapping.matrix
        if len(self.unmapped_subsystems) > 0:
            unmapped_dim = np.prod([x.dim for x in self.unmapped_subsystems])
            A = unp.kron(np.eye(unmapped_dim, dtype=complex), A)
        return A @ mat @ A.conj().transpose()

    def __str__(self):
        return f"""MappedOperator({self._operator}, {self.system_mapping.in_subsystems} ->
                {self.system_mapping.out_subsystems})"""
