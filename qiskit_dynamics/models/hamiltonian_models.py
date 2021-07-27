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
          evaluate :math:`e^{-tF}H(t)e^{tF} - F`.
    """

    def __init__(
        self,
        operators: List[Operator],
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        frame: Optional[Union[Operator, Array]] = None,
        cutoff_freq: Optional[float] = None,
        validate: bool = True,
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
            frame: Rotating frame operator. If specified with a 1d array, it
                    is interpreted as the diagonal of a diagonal matrix. If
                    provided as part of the constructor, it is assumed that
                    all operators are in the frame basis. Assumed to store
                    the antihermitian matrix F = -iH.
            validate: If True check input operators are Hermitian.

        Raises:
            Exception: if operators are not Hermitian
        """
        # verify operators are Hermitian, and if so instantiate
        operators = to_array(operators)

        if validate:
            adj = np.transpose(np.conjugate(operators), (0, 2, 1))
            if np.linalg.norm(adj - operators) > 1e-10:
                raise Exception("""HamiltonianModel only accepts Hermitian operators.""")

        super().__init__(operators=operators, signals=signals, frame=frame, cutoff_freq=cutoff_freq)

    def evaluate(self, time: float, in_frame_basis: bool = False) -> Array:
        """Evaluate the Hamiltonian at a given time.

        Note: This function from :class:`OperatorModel` needs to be overridden,
        due to frames for Hamiltonians being relative to the Schrodinger
        equation, rather than the Hamiltonian itself.
        See the class doc string for details.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            Array: the evaluated model

        Raises:
            Exception: if signals are not present
        """

        if self.signals is None:
            raise Exception(
                """OperatorModel cannot be evaluated without
                               signals."""
            )

        sig_vals = self.signals.complex_value(time)

        op_combo = self._evaluate_in_frame_basis_with_cutoffs(sig_vals)

        op_to_add_in_fb = None
        if self.frame.frame_operator is not None:
            op_to_add_in_fb = -1j * np.diag(self.frame.frame_diag)

        return self.frame._conjugate_and_add(
            time,
            op_combo,
            op_to_add_in_fb=op_to_add_in_fb,
            operator_in_frame_basis=True,
            return_in_frame_basis=in_frame_basis,
        )

    def __call__(self, t: float, y: Optional[Array] = None, in_frame: Optional[bool] = False):
        """Evaluate generator RHS functions. Needs to be overriden from base class
        to include :math:`-i`. I.e. if ``y is None``, returns :math:`-iH(t)`,
        and otherwise returns :math:`-iH(t)y`.

        Args:
            t: Time.
            y: Optional state.
            in_frame_basis: Whether or not to evaluate in the frame basis.

        Returns:
            Array: Either the evaluated model or the RHS for the given y
        """

        if y is None:
            return -1j * self.evaluate_without_state(t, in_frame=in_frame)

        return -1j * self.evaluate_with_state(t, y, in_frame=in_frame)
