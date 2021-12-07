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
from typing import Tuple, Union, List, Optional
from warnings import warn
from copy import copy
import numpy as np
from scipy.sparse.csr import csr_matrix
from scipy.sparse import issparse, diags

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.operator_collections import (
    BaseOperatorCollection,
    DenseOperatorCollection,
    SparseOperatorCollection,
    JAXSparseOperatorCollection,
)
from qiskit_dynamics.array import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.type_utils import to_numeric_matrix_type
from .rotating_frame import RotatingFrame

try:
    import jax
except ImportError:
    pass


class BaseGeneratorModel(ABC):
    r"""Defines an interface for a time-dependent
    linear differential equation of the form :math:`\dot{y}(t) = \Lambda(t, y)`,
    where :math:`\Lambda` is linear in :math:`y`. The core functionality is
    evaluation of :math:`\Lambda(t, y)`, as well as,
    if possible, evaluation of the map :math:`\Lambda(t, \cdot)`.

    Additionally, the class defines interfaces for:

     * Setting a "rotating frame", specified either directly as a :class:`RotatingFrame`
        instance, or an operator from which a :class:`RotatingFrame` instance can be constructed.
        The exact meaning of this transformation is determined by the structure of
        :math:`\Lambda(t, y)`, and is therefore by handled by concrete subclasses.
     * Setting an internal "evaluation mode", to set the specific numerical methods to use
        when evaluating :math:`\Lambda(t, y)`.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Gets matrix dimension."""

    @property
    @abstractmethod
    def rotating_frame(self) -> RotatingFrame:
        """Get the rotating frame."""

    @rotating_frame.setter
    @abstractmethod
    def rotating_frame(self, rotating_frame: RotatingFrame):
        """Set the rotating frame; either an already instantiated :class:`RotatingFrame` object,
        or a valid argument for the constructor of :class:`RotatingFrame`.
        """

    @property
    @abstractmethod
    def in_frame_basis(self) -> bool:
        """Whether or not the model is evaluated in the basis in which the frame is diagonalized."""

    @in_frame_basis.setter
    @abstractmethod
    def in_frame_basis(self, in_frame_basis: bool):
        """Set whether to evaluate in the basis in which the frame is diagonalized."""

    @property
    @abstractmethod
    def evaluation_mode(self) -> str:
        """Numerical evaluation mode of the model."""

    @evaluation_mode.setter
    @abstractmethod
    def evaluation_mode(self, new_mode: str):
        """Sets evaluation mode of model."""

    @abstractmethod
    def evaluate(self, time: float) -> Array:
        r"""If possible, evaluate the map :math:`\Lambda(t, \cdot)`.

        Args:
            time: Time.
        """

    @abstractmethod
    def evaluate_rhs(self, time: float, y: Array) -> Array:
        r"""Evaluate the right hand side :math:`\dot{y}(t) = \Lambda(t, y)`.

        Args:
            time: Time.
            y: State of the differential equation.
        """

    def copy(self):
        """Return a copy of self."""
        return copy(self)

    def __call__(self, time: float, y: Optional[Array] = None) -> Array:
        r"""Evaluate generator RHS functions. If ``y is None``,
        attemps to evaluate :math:`\Lambda(t, \cdot)`, otherwise, calculates
        :math:`\Lambda(t, y)`

        Args:
            time: Time.
            y: Optional state.

        Returns:
            Array: Either the evaluated model or the RHS for the given y
        """

        if y is None:
            return self.evaluate(time)

        return self.evaluate_rhs(time, y)


class GeneratorModel(BaseGeneratorModel):
    r"""A model for a a linear matrix differential equation in standard form.

    :class:`GeneratorModel` is a concrete instance of :class:`BaseGeneratorModel`, where the
    map :math:`\Lambda(t, y)` is explicitly constructed as:

    .. math::

        \Lambda(t, y) = G(t)y,

        G(t) = \sum_i s_i(t) G_i + G_d

    where the :math:`G_i` are matrices (represented by :class:`Operator`
    or :class:`Array` objects), the :math:`s_i(t)` are signals represented by
    a list of :class:`Signal` objects, and
    :math:`G_d` is the constant-in-time static term of the generator.
    """

    def __init__(
        self,
        static_operator: Optional[Array] = None,
        operators: Optional[Array] = None,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        evaluation_mode: str = "dense",
    ):
        """Initialize.

        Args:
            static_operator: Constant part of the generator.
            operators: A list of operators :math:`G_i`.
            signals: Stores the terms :math:`s_i(t)`. While required for evaluation,
                :class:`GeneratorModel` signals are not required at instantiation.
            rotating_frame: Rotating frame operator.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                frame operator is diagonalized.
            evaluation_mode: Evaluation mode to use.  Supported options are
                ``'dense'`` and ``'sparse'``. Call ``help(GeneratorModel.evaluation_mode)``
                for more details.
        Raises:
            QiskitError: If model not sufficiently specified.
        """
        if static_operator is None and operators is None:
            raise QiskitError(
                self.__class__.__name__
                + """ requires at least one of static_operator or operators to be
                              specified at construction."""
            )

        # initialize internal operator representation
        self._operator_collection = construct_operator_collection(
            evaluation_mode=evaluation_mode, static_operator=static_operator, operators=operators
        )
        self._evaluation_mode = evaluation_mode

        # set frame
        self._rotating_frame = None
        self.rotating_frame = rotating_frame

        # set operation in frame basis
        self._in_frame_basis = None
        self.in_frame_basis = in_frame_basis

        # initialize signal-related attributes
        self.signals = signals

    @property
    def dim(self) -> int:
        if self._operator_collection.static_operator is not None:
            return self._operator_collection.static_operator.shape[-1]
        else:
            return self._operator_collection.operators[0].shape[-1]

    @property
    def evaluation_mode(self) -> str:
        """Numerical evaluation mode of the model.

        Available options:

         * 'dense': Stores/evaluates operators using dense Arrays.
         * 'sparse': Stores/evaluates operators using sparse matrices. If
           the default Array backend is JAX, implemented with JAX BCOO arrays,
           otherwise uses scipy :class:`csr_matrix` sparse type. Note that
           JAX sparse mode is only recommended for use on CPU.
        """
        return self._evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, new_mode: str):
        """Set evaluation mode.

        Args:
            new_mode: String specifying new mode. Available options
                      are 'dense' and 'sparse'. See property doc string for details.

        Raises:
            NotImplementedError: if new_mode is not one of the above
            supported evaluation modes.
        """

        if new_mode != self._evaluation_mode:
            self._operator_collection = construct_operator_collection(
                new_mode,
                self._operator_collection.static_operator,
                self._operator_collection.operators,
            )
            self._evaluation_mode = new_mode

    @property
    def rotating_frame(self) -> RotatingFrame:
        """Return the rotating frame."""
        return self._rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]):
        new_frame = RotatingFrame(rotating_frame)

        new_static_operator = transfer_static_operator_between_frames(
            self._get_static_operator(in_frame_basis=True),
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )
        new_operators = transfer_operators_between_frames(
            self._get_operators(in_frame_basis=True),
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        self._rotating_frame = new_frame

        self._operator_collection = construct_operator_collection(
            self.evaluation_mode, new_static_operator, new_operators
        )

    @property
    def signals(self) -> SignalList:
        """Return the signals in the model."""
        return self._signals

    @signals.setter
    def signals(self, signals: Union[SignalList, List[Signal]]):
        """Set the signals."""

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

    @property
    def in_frame_basis(self) -> bool:
        """Whether to represent the model in the basis in which the frame operator
        is diagonalized.
        """
        return self._in_frame_basis

    @in_frame_basis.setter
    def in_frame_basis(self, in_frame_basis: bool):
        self._in_frame_basis = in_frame_basis

    @property
    def operators(self) -> Array:
        """Get the operators in the model."""
        return self._get_operators(in_frame_basis=self._in_frame_basis)

    @property
    def static_operator(self) -> Array:
        """Get the static operator."""
        return self._get_static_operator(in_frame_basis=self._in_frame_basis)

    @static_operator.setter
    def static_operator(self, static_operator: Array):
        """Set the static operator."""
        self._set_static_operator(
            new_static_operator=static_operator, operator_in_frame_basis=self._in_frame_basis
        )

    def _get_operators(self, in_frame_basis: Optional[bool] = False) -> Array:
        """Get the operators used in the model construction.

        Args:
            in_frame_basis: Flag indicating whether to return the operators
            in the basis in which the rotating frame operator is diagonal.
        Returns:
            The operators in the basis specified by `in_frame_basis`.
        """
        ops = self._operator_collection.operators
        if ops is not None and not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(ops)
        else:
            return ops

    def _get_static_operator(self, in_frame_basis: Optional[bool] = False) -> Array:
        """Get the constant term.

        Args:
            in_frame_basis: Flag for whether the returned static_operator should be
            in the basis in which the frame is diagonal.
        Returns:
            The static operator term.
        """
        op = self._operator_collection.static_operator
        if not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(op)
        else:
            return op

    def _set_static_operator(
        self,
        new_static_operator: Array,
        operator_in_frame_basis: Optional[bool] = False,
    ):
        """Sets static term. Note that if the model has a rotating frame this will override
        any contributions to the static term due to the frame transformation.

        Args:
            new_static_operator: The static operator operator.
            operator_in_frame_basis: Whether `new_static_operator` is already in the rotating
            frame basis.
        """
        if new_static_operator is None:
            self._operator_collection.static_operator = None
        else:
            if not operator_in_frame_basis and self.rotating_frame is not None:
                new_static_operator = self.rotating_frame.operator_into_frame_basis(
                    new_static_operator
                )

            self._operator_collection.static_operator = new_static_operator

    def evaluate(self, time: float) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.

        Args:
            time: Time to evaluate the model.

        Returns:
            Array: the evaluated model as a (n,n) matrix

        Raises:
            QiskitError: If model cannot be evaluated.
        """

        if self._signals is None:
            if self._operator_collection.operators is not None:
                raise QiskitError(
                    self.__class__.__name__
                    + " with non-empty operators cannot be evaluated without signals."
                )

            sig_vals = None
        else:
            sig_vals = self._signals.__call__(time)

        # Evaluated in frame basis, but without rotations
        op_combo = self._operator_collection(sig_vals)

        # Apply rotations e^{-Ft}Ae^{Ft} in frame basis where F = D
        return self.rotating_frame.operator_into_frame(
            time, op_combo, operator_in_frame_basis=True, return_in_frame_basis=self._in_frame_basis
        )

    def evaluate_rhs(self, time: float, y: Array) -> Array:
        r"""Evaluate `G(t) @ y`.

        Args:
            time: Time to evaluate the model.
            y: Array specifying system state.

        Returns:
            Array defined by :math:`G(t) \times y`.

        Raises:
            QiskitError: If model cannot be evaluated.
        """

        if self._signals is None:
            if self._operator_collection.operators is not None:
                raise QiskitError(
                    self.__class__.__name__
                    + " with non-empty operators cannot be evaluated without signals."
                )

            sig_vals = None
        else:
            sig_vals = self._signals.__call__(time)

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
        else:
            return self._operator_collection(sig_vals, y)

        return out


def transfer_static_operator_between_frames(
    static_operator: Union[None, Array, csr_matrix],
    new_frame: Optional[Union[Array, RotatingFrame]] = None,
    old_frame: Optional[Union[Array, RotatingFrame]] = None,
) -> Tuple[Union[None, Array]]:
    """Helper function for transforming the static operator for a model from one
    frame basis into another.

    ``static_operator`` is additionally transformed as a generator: the old frame operator is
    added, and the new one is subtracted.

    Args:
        static_operator: Static operator of the model. None is treated as 0.
        new_frame: New rotating frame.
        old_frame: Old rotating frame.

    Returns:
        static_operator: Transformed as described above.
    """
    new_frame = RotatingFrame(new_frame)
    old_frame = RotatingFrame(old_frame)

    static_operator = to_numeric_matrix_type(static_operator)

    # transform out of old frame basis, and add the old frame operator
    if static_operator is not None:
        static_operator = old_frame.generator_out_of_frame(
            t=0.0,
            operator=static_operator,
            operator_in_frame_basis=True,
            return_in_frame_basis=False,
        )
    else:
        # "add" the frame operator to 0
        if old_frame.frame_operator is not None:
            if issparse(static_operator):
                if old_frame.frame_operator.ndim == 1:
                    static_operator = diags(old_frame.frame_operator, format="csr")
                else:
                    static_operator = csr_matrix(old_frame.frame_operator)
            else:
                if old_frame.frame_operator.ndim == 1:
                    static_operator = np.diag(old_frame.frame_operator)
                else:
                    static_operator = old_frame.frame_operator
    # transform into new frame basis, and add the new frame operator
    if static_operator is not None:
        static_operator = new_frame.generator_into_frame(
            t=0.0,
            operator=static_operator,
            operator_in_frame_basis=False,
            return_in_frame_basis=True,
        )
    else:
        # "subtract" the frame operator from 0
        if new_frame.frame_operator is not None:
            if issparse(static_operator):
                static_operator = -diags(new_frame.frame_diag, format="csr")
            else:
                static_operator = -np.diag(new_frame.frame_diag)

    return static_operator


def transfer_operators_between_frames(
    operators: Union[None, Array, List[csr_matrix]],
    new_frame: Optional[Union[Array, RotatingFrame]] = None,
    old_frame: Optional[Union[Array, RotatingFrame]] = None,
) -> Tuple[Union[None, Array]]:
    """Helper function for transforming a list of operators for a model
    from one frame basis into another.

    Args:
        operators: Operators of the model. If None, remains as None.
        new_frame: New rotating frame.
        old_frame: Old rotating frame.

    Returns:
        operators: Transformed as described above.
    """
    new_frame = RotatingFrame(new_frame)
    old_frame = RotatingFrame(old_frame)

    operators = to_numeric_matrix_type(operators)

    # transform out of old frame basis
    if operators is not None:
        # For sparse case, if list, loop
        if isinstance(operators, list):
            operators = [old_frame.operator_out_of_frame_basis(op) for op in operators]
        else:
            operators = old_frame.operator_out_of_frame_basis(operators)

    # transform into new frame basis
    if operators is not None:
        # For sparse case, if list, loop
        if isinstance(operators, list):
            operators = [new_frame.operator_into_frame_basis(op) for op in operators]
        else:
            operators = new_frame.operator_into_frame_basis(operators)

    return operators


def construct_operator_collection(
    evaluation_mode: str,
    static_operator: Union[None, Array, csr_matrix],
    operators: Union[None, Array, List[csr_matrix]],
) -> BaseOperatorCollection:
    """Construct operator collection for GeneratorModel.

    Args:
        evaluation_mode: Evaluation mode.
        static_operator: Static operator of the model.
        operators: Operators for the model.

    Returns:
        BaseOperatorCollection: The relevant operator collection.

    Raises:
        NotImplementedError: If the evaluation_mode is invalid.
    """

    if evaluation_mode == "dense":
        return DenseOperatorCollection(static_operator=static_operator, operators=operators)
    if evaluation_mode == "sparse" and Array.default_backend() == "jax":
        # warn that sparse mode when using JAX is primarily recommended for use on CPU
        if jax.default_backend() != "cpu":
            warn(
                """Using sparse mode with JAX is primarily recommended for use on CPU.""",
                stacklevel=2,
            )

        return JAXSparseOperatorCollection(static_operator=static_operator, operators=operators)
    if evaluation_mode == "sparse":
        return SparseOperatorCollection(static_operator=static_operator, operators=operators)

    raise NotImplementedError(
        "Evaluation mode '"
        + str(evaluation_mode)
        + "' is not supported. Call help(GeneratorModel.evaluation_mode) for available options."
    )
