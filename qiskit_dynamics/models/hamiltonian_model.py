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
from scipy.sparse import issparse
from scipy.sparse.csr import csr_matrix
from scipy.sparse.linalg import norm as spnorm

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.array import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.type_utils import to_numeric_matrix_type, to_array
from .generator_model import (
    GeneratorModel,
    transfer_static_operator_between_frames,
    transfer_operators_between_frames,
    construct_operator_collection,
)
from .rotating_frame import RotatingFrame


class HamiltonianModel(GeneratorModel):
    r"""A model of a Hamiltonian for the Schrodinger equation.

    This class represents a Hamiltonian as a time-dependent decomposition the form:

    .. math::
        H(t) = H_d + \sum_j s_j(t) H_j,

    where :math:`H_j` are Hermitian operators, :math:`H_d` is the static component,
    and the :math:`s_j(t)` are either :class:`~qiskit_dynamics.signals.Signal` objects or
    are numerical constants. Constructing a :class:`~qiskit_dynamics.models.HamiltonianModel`
    requires specifying the above decomposition, e.g.:

    .. code-block:: python

        hamiltonian = HamiltonianModel(static_operator=static_operator,
                                       operators=operators,
                                       signals=signals)

    This class inherits most functionality from :class:`GeneratorModel`,
    with the following modifications:

        * The operators :math:`H_d` and :math:`H_j` are assumed and verified to be Hermitian.
        * Rotating frames are dealt with assuming the structure of the Schrodinger
          equation. I.e. Evaluating the Hamiltonian :math:`H(t)` in a
          frame :math:`F = -iH_0`, evaluates the expression
          :math:`e^{-tF}H(t)e^{tF} - H_0`.
    """

    def __init__(
        self,
        static_operator: Optional[Array] = None,
        operators: Optional[List[Operator]] = None,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        evaluation_mode: str = "dense",
        validate: bool = True,
    ):
        """Initialize, ensuring that the operators are Hermitian.

        Args:
            static_operator: Time-independent term in the Hamiltonian.
            operators: List of Operator objects.
            signals: List of coefficients :math:`s_i(t)`. Not required at instantiation, but
                     necessary for evaluation of the model.
            rotating_frame: Rotating frame operator.
                            If specified with a 1d array, it is interpreted as the
                            diagonal of a diagonal matrix. Assumed to store
                            the antihermitian matrix F = -iH.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                            frame operator is diagonalized.
            evaluation_mode: Evaluation mode to use. Supported options are ``'dense'`` and
                             ``'sparse'``. Call ``help(HamiltonianModel.evaluation_mode)`` for more
                             details.
            validate: If True check input operators are Hermitian.

        Raises:
            QiskitError: if operators are not Hermitian
        """

        # verify operators are Hermitian, and if so instantiate
        operators = to_numeric_matrix_type(operators)
        static_operator = to_numeric_matrix_type(static_operator)

        if validate:
            if (operators is not None) and (not is_hermitian(operators)):
                raise QiskitError("""HamiltonianModel operators must be Hermitian.""")
            if (static_operator is not None) and (not is_hermitian(static_operator)):
                raise QiskitError("""HamiltonianModel static_operator must be Hermitian.""")

        super().__init__(
            static_operator=static_operator,
            operators=operators,
            signals=signals,
            rotating_frame=rotating_frame,
            in_frame_basis=in_frame_basis,
            evaluation_mode=evaluation_mode,
        )

    @property
    def rotating_frame(self) -> RotatingFrame:
        return super().rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]) -> Array:
        """Sets frame. RotatingFrame objects will always store antihermitian F = -iH.
        The static_operator needs to be adjusted by -H in the new frame."""
        new_frame = RotatingFrame(rotating_frame)

        # convert static operator to new frame setup
        static_op = self._get_static_operator(in_frame_basis=True)
        if static_op is not None:
            static_op = -1j * static_op

        new_static_operator = transfer_static_operator_between_frames(
            static_op,
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        if new_static_operator is not None:
            new_static_operator = 1j * new_static_operator

        # convert operators to new frame set up
        new_operators = transfer_operators_between_frames(
            self._get_operators(in_frame_basis=True),
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        self._rotating_frame = new_frame

        self._operator_collection = construct_operator_collection(
            self.evaluation_mode, new_static_operator, new_operators
        )

    def evaluate(self, time: float) -> Array:
        return -1j * super().evaluate(time)

    def evaluate_rhs(self, time: float, y: Array) -> Array:
        return -1j * super().evaluate_rhs(time, y)


def is_hermitian(
    operators: Union[Array, csr_matrix, List[csr_matrix], "BCOO"], tol: Optional[float] = 1e-10
) -> bool:
    """Validate that operators are Hermitian.

    Args:
        operators: Either a 2d array representing a single operator, a 3d array
                   representing a list of operators, a csr_matrix, or a list
                   of csr_matrix.
        tol: Tolerance for checking zeros.

    Returns:
        bool: Whether or not the operators are Hermitian to within tolerance.

    Raises:
        QiskitError: If an unexpeted type.
    """
    if isinstance(operators, (np.ndarray, Array)):
        adj = None
        if operators.ndim == 2:
            adj = np.transpose(np.conjugate(operators))
        elif operators.ndim == 3:
            adj = np.transpose(np.conjugate(operators), (0, 2, 1))
        return np.linalg.norm(adj - operators) < tol
    elif issparse(operators):
        return spnorm(operators - operators.conj().transpose()) < tol
    elif isinstance(operators, list) and issparse(operators[0]):
        return all(spnorm(op - op.conj().transpose()) < tol for op in operators)
    elif type(operators).__name__ == "BCOO":
        # fall back on array case for BCOO
        return is_hermitian(to_array(operators))

    raise QiskitError("is_hermitian got an unexpected type.")
