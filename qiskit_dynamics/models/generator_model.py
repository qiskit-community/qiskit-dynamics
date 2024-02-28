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
Generator models module.
"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from warnings import warn
from scipy.sparse import diags

from qiskit import QiskitError

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS as numpy_alias
from qiskit_dynamics.arraylias.alias import ArrayLike
from qiskit_dynamics.models.operator_collections import (
    OperatorCollection,
    ScipySparseOperatorCollection,
)
from qiskit_dynamics.signals import Signal, SignalList
from .rotating_frame import RotatingFrame

try:
    import jax
except ImportError:
    pass


class BaseGeneratorModel(ABC):
    r"""Defines an interface for a time-dependent linear differential equation of the form
    :math:`\dot{y}(t) = \Lambda(t, y)`, where :math:`\Lambda` is linear in :math:`y`. The core
    functionality is evaluation of :math:`\Lambda(t, y)`, as well as, if possible, evaluation of the
    map :math:`\Lambda(t, \cdot)`.
    """

    def __init__(self, array_library: Optional[str] = None):
        """Set up general information used by all subclasses."""
        self._array_library = array_library

    @property
    @abstractmethod
    def dim(self) -> int:
        """The matrix dimension."""

    @property
    @abstractmethod
    def rotating_frame(self) -> RotatingFrame:
        """The rotating frame."""

    @property
    @abstractmethod
    def in_frame_basis(self) -> bool:
        """Whether or not the model is evaluated in the basis in which the frame is diagonalized."""

    @property
    def array_library(self) -> Union[None, str]:
        """Array library with which to represent the operators in the model, and to evaluate the
        model.

        See the list of supported array libraries in the :mod:`.arraylias` submodule API
        documentation.
        """
        return self._array_library

    @abstractmethod
    def evaluate(self, time: float) -> ArrayLike:
        r"""If possible, evaluate the map :math:`\Lambda(t, \cdot)`.

        Args:
            time: The time to evaluate the model at.
        """

    @abstractmethod
    def evaluate_rhs(self, time: float, y: ArrayLike) -> ArrayLike:
        r"""Evaluate the right hand side :math:`\dot{y}(t) = \Lambda(t, y)`.

        Args:
            time: The time to evaluate the model at.
            y: State of the differential equation.
        """

    def __call__(self, time: float, y: Optional[ArrayLike] = None) -> ArrayLike:
        r"""Evaluate generator RHS functions. If ``y is None``, attemps to evaluate
        :math:`\Lambda(t, \cdot)`, otherwise, calculates :math:`\Lambda(t, y)`.

        Args:
            time: The time to evaluate at.
            y: Optional state.

        Returns:
            ArrayLike: Either the evaluated model, or the RHS for the given y.
        """
        return self.evaluate(time) if y is None else self.evaluate_rhs(time, y)


class GeneratorModel(BaseGeneratorModel):
    r"""A model for a a linear matrix differential equation in standard form.

    :class:`GeneratorModel` is a concrete instance of :class:`BaseGeneratorModel`, where the map
    :math:`\Lambda(t, y)` is explicitly constructed as:

    .. math::

        \Lambda(t, y) = G(t)y,

        G(t) = \sum_i s_i(t) G_i + G_d

    where the :math:`G_i` are matrices (represented by :class:`Operator` or :class:`Array` objects),
    the :math:`s_i(t)` are signals represented by a list of :class:`Signal` objects, and :math:`G_d`
    is the constant-in-time static term of the generator.
    """

    def __init__(
        self,
        static_operator: Optional[ArrayLike] = None,
        operators: Optional[ArrayLike] = None,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        rotating_frame: Optional[Union[ArrayLike, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        array_library: Optional[str] = None,
    ):
        """Initialize.

        Args:
            static_operator: Constant part of the generator.
            operators: A list of operators :math:`G_i`.
            signals: Stores the terms :math:`s_i(t)`. While required for evaluation,
                :class:`GeneratorModel` signals are not required at instantiation.
            rotating_frame: Rotating frame operator.
            in_frame_basis: Whether to represent the model in the basis in which the rotating frame
                operator is diagonalized.
            array_library: Array library with which to represent the operators in the model, and to
                evaluate the model. See the list of supported array libraries in the
                :mod:`.arraylias` submodule API documentation. If ``None``, the arrays will be
                handled by general dispatching rules.
        Raises:
            QiskitError: If model not sufficiently specified.
        """
        if static_operator is None and operators is None:
            raise QiskitError(
                f"{type(self).__name__} requires at least one of static_operator or operators to "
                "be specified at construction."
            )

        self._rotating_frame = RotatingFrame(rotating_frame)
        self._in_frame_basis = in_frame_basis

        # set up internal operators
        static_operator = _static_operator_into_frame_basis(
            static_operator=static_operator,
            rotating_frame=self._rotating_frame,
            array_library=array_library,
        )

        operators = _operators_into_frame_basis(
            operators=operators, rotating_frame=self._rotating_frame, array_library=array_library
        )

        self._operator_collection = _get_operator_collection(
            static_operator=static_operator, operators=operators, array_library=array_library
        )

        self._signals = None
        self.signals = signals

        super().__init__(array_library=array_library)

    @property
    def dim(self) -> int:
        """The matrix dimension."""
        return self._operator_collection.dim

    @property
    def rotating_frame(self) -> RotatingFrame:
        """The rotating frame."""
        return self._rotating_frame

    @property
    def in_frame_basis(self) -> bool:
        """Whether or not the model is evaluated in the basis in which the frame is diagonalized."""
        return self._in_frame_basis

    @in_frame_basis.setter
    def in_frame_basis(self, in_frame_basis: bool):
        self._in_frame_basis = in_frame_basis

    @property
    def static_operator(self) -> Union[ArrayLike, None]:
        """The static operator."""
        if self._operator_collection.static_operator is None:
            return None

        if self.in_frame_basis:
            return self._operator_collection.static_operator
        return self.rotating_frame.operator_out_of_frame_basis(
            self._operator_collection.static_operator
        )

    @property
    def operators(self) -> Union[ArrayLike, None]:
        """The operators in the model."""
        if self._operator_collection.operators is None:
            return None

        if self.in_frame_basis:
            return self._operator_collection.operators
        return self.rotating_frame.operator_out_of_frame_basis(self._operator_collection.operators)

    @property
    def signals(self) -> SignalList:
        """The signals in the model.

        Raises:
            QiskitError: If set to ``None`` when operators exist, or when set to a number of signals
                different then the number of operators.
        """
        return self._signals

    @signals.setter
    def signals(self, signals: Union[SignalList, List[Signal]]):
        if signals is None:
            self._signals = None
        elif signals is not None and self.operators is None:
            raise QiskitError("Signals must be None if operators is None.")
        else:
            # if signals is a list, instantiate a SignalList
            if isinstance(signals, list):
                signals = SignalList(signals)

            # if it isn't a SignalList by now, raise an error
            if not isinstance(signals, SignalList):
                raise QiskitError("Signals specified in unaccepted format.")

            # verify signal length is same as operators
            if isinstance(self.operators, list):
                len_operators = len(self.operators)
            else:
                len_operators = self.operators.shape[0]
            if len(signals) != len_operators:
                raise QiskitError("Signals needs to have the same length as operators.")

            self._signals = signals

    def evaluate(self, time: float) -> ArrayLike:
        """Evaluate the model in array format as a matrix, independent of state.

        Args:
            time: The time to evaluate the model at.

        Returns:
            ArrayLike: The evaluated model as a matrix.

        Raises:
            QiskitError: If model cannot be evaluated.
        """
        if self._signals is None and self._operator_collection.operators is not None:
            raise QiskitError(
                f"{type(self).__name__} with non-empty operators must be evaluated signals."
            )

        # Evaluated in frame basis, but without rotations
        op_combo = self._operator_collection(self._signals(time) if self._signals else None)

        # Apply rotations e^{-Ft}Ae^{Ft} in frame basis where F = D
        return self.rotating_frame.operator_into_frame(
            time, op_combo, operator_in_frame_basis=True, return_in_frame_basis=self._in_frame_basis
        )

    def evaluate_rhs(self, time: float, y: ArrayLike) -> ArrayLike:
        r"""Evaluate ``G(t) @ y``.

        Args:
            time: The time to evaluate the model at .
            y: Array specifying system state.

        Returns:
            Array defined by :math:`G(t) \times y`.

        Raises:
            QiskitError: If model cannot be evaluated.
        """
        if self._signals is None:
            if self._operator_collection.operators is not None:
                raise QiskitError(
                    f"{type(self).__name__} with non-empty operators must be evaluated signals."
                )
            sig_vals = None
        else:
            sig_vals = self._signals(time)

        if self.rotating_frame is not None:
            # First, compute e^{tF}y as a pre-rotation in the frame basis
            out = self.rotating_frame.state_out_of_frame(
                time, y, y_in_frame_basis=self._in_frame_basis, return_in_frame_basis=True
            )
            # Then, compute the product Ae^{tF}y
            out = self._operator_collection(sig_vals, out)
            # Finally, we have the full operator e^{-tF}Ae^{tF}y
            out = self.rotating_frame.state_into_frame(
                time, out, y_in_frame_basis=True, return_in_frame_basis=self._in_frame_basis
            )
            return out

        return self._operator_collection(sig_vals, y)


def _static_operator_into_frame_basis(
    static_operator: Union[None, ArrayLike],
    rotating_frame: RotatingFrame,
    array_library: Optional[ArrayLike] = None,
) -> Union[None, ArrayLike]:
    """Converts the static_operator into the frame basis, including a subtraction of the frame
    operator. This function also enforces typing via array_library.
    """

    # handle static_operator is None case
    if static_operator is None:
        if rotating_frame.frame_operator is None:
            return None
        if array_library == "scipy_sparse":
            return -diags(rotating_frame.frame_diag, format="csr")
        return unp.diag(-rotating_frame.frame_diag)

    static_operator = numpy_alias(like=array_library).asarray(static_operator)

    return rotating_frame.generator_into_frame(
        t=0.0, operator=static_operator, return_in_frame_basis=True
    )


def _operators_into_frame_basis(
    operators: Union[None, list, ArrayLike],
    rotating_frame: RotatingFrame,
    array_library: Optional[str] = None,
) -> Union[None, ArrayLike]:
    """Converts operators into the frame basis. This function also enforces typing via
    array_library.
    """
    if operators is None:
        return None

    if array_library == "scipy_sparse" or (
        array_library is None and "scipy_sparse" in numpy_alias.infer_libs(operators)
    ):
        ops = []
        for op in operators:
            op = numpy_alias(like="scipy_sparse").asarray(op)
            ops.append(rotating_frame.operator_into_frame_basis(op))
        return ops

    return rotating_frame.operator_into_frame_basis(
        numpy_alias(like=array_library).asarray(operators)
    )


def _get_operator_collection(
    static_operator: Union[None, ArrayLike],
    operators: Union[None, ArrayLike],
    array_library: Optional[str] = None,
) -> Union[OperatorCollection, ScipySparseOperatorCollection]:
    """Construct an operator collection for :class:`GeneratorModel`.

    Args:
        static_operator: Static operator of the model.
        operators: Operators for the model.
        array_library: Array library to use.

    Returns:
        Union[OperatorCollection, ScipySparseOperatorCollection]: The relevant operator collection.
    """

    if array_library == "scipy_sparse":
        return ScipySparseOperatorCollection(static_operator=static_operator, operators=operators)

    if array_library == "jax_sparse":
        # warn that sparse mode when using JAX is primarily recommended for use on CPU
        if jax.default_backend() != "cpu":
            warn(
                """Using sparse mode with JAX is primarily recommended for use on CPU.""",
                stacklevel=2,
            )

    return OperatorCollection(
        static_operator=static_operator, operators=operators, array_library=array_library
    )
