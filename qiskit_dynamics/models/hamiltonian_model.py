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
from scipy.sparse.linalg import norm as spnorm

from qiskit import QiskitError

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS as numpy_alias
from qiskit_dynamics.arraylias.alias import ArrayLike
from qiskit_dynamics.signals import Signal, SignalList
from .generator_model import GeneratorModel
from .rotating_frame import RotatingFrame


class HamiltonianModel(GeneratorModel):
    r"""A model of a Hamiltonian for the Schrodinger equation.

    This class represents a Hamiltonian as a time-dependent decomposition the form:

    .. math::
        H(t) = H_d + \sum_j s_j(t) H_j,

    where :math:`H_j` are Hermitian operators, :math:`H_d` is the static component, and the
    :math:`s_j(t)` are either :class:`~qiskit_dynamics.signals.Signal` objects or numerical
    constants. Constructing a :class:`~qiskit_dynamics.models.HamiltonianModel` requires specifying
    the above decomposition, e.g.:

    .. code-block:: python

        hamiltonian = HamiltonianModel(
            static_operator=static_operator,
            operators=operators,
            signals=signals
        )

    This class inherits most functionality from :class:`GeneratorModel`, with the following
    modifications:

        * The operators :math:`H_d` and :math:`H_j` are assumed and verified to be Hermitian.
        * Rotating frames are dealt with assuming the structure of the Schrodinger equation. I.e.
          Evaluating the Hamiltonian :math:`H(t)` in a frame :math:`F = -iH_0`, evaluates the
          expression :math:`e^{-tF}H(t)e^{tF} - H_0`.
    """

    def __init__(
        self,
        static_operator: Optional[ArrayLike] = None,
        operators: Optional[ArrayLike] = None,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        rotating_frame: Optional[Union[ArrayLike, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        array_library: Optional[str] = None,
        validate: bool = True,
    ):
        """Initialize, ensuring that the operators are Hermitian.

        Args:
            static_operator: Time-independent term in the Hamiltonian.
            operators: List of Operator objects.
            signals: List of coefficients :math:`s_i(t)`. Not required at instantiation, but
                necessary for evaluation of the model.
            rotating_frame: Rotating frame operator. If specified with a 1d array, it is interpreted
                as the diagonal of a diagonal matrix. Assumed to store the antihermitian matrix
                F = -iH.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                frame operator is diagonalized.
            array_library: Array library for storing the operators in the model. Supported options
                are ``'numpy'``, ``'jax'``, ``'jax_sparse'``, and ``'scipy_sparse'``. If ``None``,
                the arrays will be handled by general dispatching rules. Call
                ``help(GeneratorModel.array_library)`` for more details.
            validate: If ``True`` check input operators are Hermitian. Note that this is
                incompatible with JAX transformations.

        Raises:
            QiskitError: if operators are not Hermitian
        """

        # prepare and validate operators
        if static_operator is not None:
            if validate and not is_hermitian(static_operator):
                raise QiskitError("""HamiltonianModel static_operator must be Hermitian.""")
            static_operator = -1j * numpy_alias(like=array_library).asarray(static_operator)

        if operators is not None:
            if validate and any(not is_hermitian(op) for op in operators):
                raise QiskitError("""HamiltonianModel operators must be Hermitian.""")

            if array_library == "scipy_sparse" or (
                array_library is None and "scipy_sparse" in numpy_alias.infer_libs(operators)
            ):
                operators = [-1j * numpy_alias(like=array_library).asarray(op) for op in operators]
            else:
                operators = -1j * numpy_alias(like=array_library).asarray(operators)

        super().__init__(
            static_operator=static_operator,
            operators=operators,
            signals=signals,
            rotating_frame=rotating_frame,
            in_frame_basis=in_frame_basis,
            array_library=array_library,
        )

    @property
    def static_operator(self) -> Union[ArrayLike, None]:
        """The static operator."""
        if self._operator_collection.static_operator is None:
            return None

        if self.in_frame_basis:
            return self._operator_collection.static_operator
        return 1j * self.rotating_frame.operator_out_of_frame_basis(
            self._operator_collection.static_operator
        )

    @property
    def operators(self) -> Union[ArrayLike, None]:
        """The operators in the model."""
        if self._operator_collection.operators is None:
            return None

        if self.in_frame_basis:
            ops = self._operator_collection.operators
        else:
            ops = self.rotating_frame.operator_out_of_frame_basis(
                self._operator_collection.operators
            )

        if isinstance(ops, list):
            return [1j * op for op in ops]

        return 1j * ops


def is_hermitian(operator: ArrayLike, tol: Optional[float] = 1e-10) -> bool:
    """Validate that an operator is Hermitian.

    Args:
        operator: A 2d array representing a single operator.
        tol: Tolerance for checking zeros.

    Returns:
        bool: Whether or not the operator is Hermitian to within tolerance.

    Raises:
        QiskitError: If an unexpeted type is received.
    """
    operator = unp.asarray(operator)

    if issparse(operator):
        return spnorm(operator - operator.conj().transpose()) < tol
    elif type(operator).__name__ == "BCOO":
        # fall back on array case for BCOO
        return is_hermitian(operator.todense())
    elif isinstance(operator, ArrayLike):
        adj = None
        adj = np.transpose(np.conjugate(operator))
        return np.linalg.norm(adj - operator) < tol

    raise QiskitError("is_hermitian got an unexpected type.")
