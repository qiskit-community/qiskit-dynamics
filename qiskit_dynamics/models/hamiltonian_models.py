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
        H(t) = H_d + \sum_{i=0}^{k-1} s_i(t) H_i,

    where :math:`H_i` are Hermitian operators, :math:`H_d` is the drift component,
    and the :math:`s_i(t)` are time-dependent functions represented by :class:`Signal` objects.

    This class inherits most functionality from :class:`GeneratorModel`,
    with the following modifications:
        - The operators :math:`H_d` and :math:`H_i` are assumed and verified to be Hermitian.
        - Rotating frames are dealt with assuming the structure of the Schrodinger
          equation. I.e. Evaluating the Hamiltonian :math:`H(t)` in a
          frame :math:`F = -iH`, evaluates the expression
          :math:`e^{-tF}H(t)e^{tF} - H`. This is in contrast to
          the base class :class:`OperatorModel`, which would ordinarily
          evaluate :math:`e^{-tF}H(t)e^{tF} - F`.
    """

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
            drift: Optional, time-independent term in the Hamiltonian.
            signals: List of coefficients :math:`s_i(t)`. Not required at instantiation, but
                     necessary for evaluation of the model.
            rotating_frame: Rotating frame operator.
                            If specified with a 1d array, it is interpreted as the
                            diagonal of a diagonal matrix. Assumed to store
                            the antihermitian matrix F = -iH.
            validate: If True check input operators are Hermitian.
            evaluation_mode: Evaluation mode to use. Supported options are:
                                - 'dense' (DenseOperatorCollection)
                                - 'sparse' (SparseOperatorCollection)
                             See :method:`GeneratorModel.set_evaluation_mode` for more details.

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

        # Reset internal operator collection
        if self.evaluation_mode is not None:
            self.set_evaluation_mode(self.evaluation_mode)

    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        return -1j * super().evaluate(time, in_frame_basis=in_frame_basis)

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        return -1j * super().evaluate_rhs(time, y, in_frame_basis=in_frame_basis)
