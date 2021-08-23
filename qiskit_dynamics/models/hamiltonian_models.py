# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
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
Hamiltonian models module.
"""

from typing import Union, List, Optional
import numpy as np

from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.type_utils import to_array
from .generator_models import GeneratorModel
from .rotating_frame import RotatingFrame


class HamiltonianModel(GeneratorModel):
    r"""A model of a Hamiltonian.

    This class represents a Hamiltonian as a time-dependent decomposition the form:

    .. math::
        H(t) = \sum_{i=0}^{k-1} s_i(t) H_i,

    where :math:`H_i` are Hermitian operators, and the :math:`s_i(t)` are
    time-dependent functions represented by :class:`Signal` objects.

    This class inherits

    Currently the functionality of this class is as a subclass of
    :class:`GeneratorModel`, with the following modifications:

    - The operators in the linear decomposition are verified to be
        Hermitian.

    - Frames are dealt with assuming the structure of the Schrodinger
        equation. I.e. Evaluating the Hamiltonian :math:`H(t)` in a
        frame :math:`F = -iH`, evaluates the expression
        :math:`e^{-tF}H(t)e^{tF} - H`. This is in contrast to
        the base class :class:`OperatorModel`, which would ordinarily
        evaluate :math:`e^{-tF}H(t)e^{tF} - F`."""

    def __init__(
        self,
        operators: List[Operator],
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        drift: Optional[Array] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        validate: bool = True,
        evaluation_mode: str = "dense",
    ):
        """Initialize, ensuring that the operators are Hermitian.

        Args:
            operators: list of Operator objects.
            drift: optional, time-independent term in the Hamiltonian.
            Note: If both frame and drift are provided, assumed that
            drift term includes frame contribution. If
            frame but not drift given, a frame drift will be constructed.
            signals: Specifiable as either a SignalList, a list of
            Signal objects, or as the inputs to signal_mapping.
            OperatorModel can be instantiated without specifying
            signals, but it can not perform any actions without them.
            rotating_frame: Rotating frame operator / rotating frame object.
            If specified with a 1d array, it is interpreted as the
            diagonal of a diagonal matrix. Assumed to store
            the antihermitian matrix F = -iH.
            validate: If True check input operators are Hermitian.
            evaluation_mode: Flag for what type of evaluation should
            be used. Currently supported options are
            dense (DenseOperatorCollection)
            sparse (SparseOperatorCollection)

        Raises:
            Exception: if operators are not Hermitian
        """

        # verify operators are Hermitian, and if so instantiate
        operators = to_array(operators)

        if validate:
            adj = np.transpose(np.conjugate(operators), (0, 2, 1))
            if np.linalg.norm(adj - operators) > 1e-10 or (
                drift is not None
                and np.linalg.norm(drift - np.conjugate(np.transpose(drift))) > 1e-10
            ):
                raise Exception("""HamiltonianModel only accepts Hermitian operators.""")

        super().__init__(
            operators=operators,
            signals=signals,
            rotating_frame=rotating_frame,
            drift=drift,
            evaluation_mode=evaluation_mode,
        )

    @property
    def rotating_frame(self) -> RotatingFrame:
        return super().rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]) -> Array:
        """Sets frame. RotatingFrame objects will always store antihermitian F = -iH.
        The drift needs to be adjusted by -H in the new frame."""
        if self._rotating_frame is not None and self._rotating_frame.frame_diag is not None:
            self._drift = self._drift + Array(np.diag(1j * self._rotating_frame.frame_diag))
            self._operators = self.rotating_frame.operator_out_of_frame_basis(self._operators)
            self._drift = self.rotating_frame.operator_out_of_frame_basis(self._drift)

        self._rotating_frame = RotatingFrame(rotating_frame)

        if self._rotating_frame.frame_diag is not None:
            self._operators = self.rotating_frame.operator_into_frame_basis(self._operators)
            self._drift = self.rotating_frame.operator_into_frame_basis(self._drift)
            self._drift = self._drift - Array(np.diag(1j * self.rotating_frame.frame_diag))

        self.set_evaluation_mode(self.evaluation_mode)

    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        return -1j * super().evaluate(time, in_frame_basis=in_frame_basis)

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        return -1j * super().evaluate_rhs(time, y, in_frame_basis=in_frame_basis)
