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
from typing import Callable, Tuple, Union, List, Optional
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
)
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.type_utils import to_array, to_csr, to_numeric_matrix_type
from .rotating_frame import RotatingFrame


class BaseGeneratorModel(ABC):
    r"""Defines an interface for a time-dependent
    linear differential equation of the form :math:`\dot{y}(t) = \Lambda(t, y)`,
    where :math:`\Lambda` is linear in :math:`y`. The core functionality is
    evaluation of :math:`\Lambda(t, y)`, as well as,
    if possible, evaluation of the map :math:`\Lambda(t, \cdot)`.

    Additionally, the class defines interfaces for:

        - Setting a "rotating frame", specified either directly as a :class:`RotatingFrame`
        instance, or an operator from which a :class:`RotatingFrame` instance can be constructed.
        The exact meaning of this transformation is determined by the structure of
        :math:`\Lambda(t, y)`, and is therefore by handled by concrete subclasses.
        - Setting an internal "evaluation mode", to set the specific numerical methods to use
        when evaluating :math:`\Lambda(t, y)`.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Gets matrix dimension."""
        pass

    @property
    @abstractmethod
    def rotating_frame(self) -> RotatingFrame:
        """Get the rotating frame."""
        pass

    @rotating_frame.setter
    @abstractmethod
    def rotating_frame(self, rotating_frame: RotatingFrame):
        """Set the rotating frame; either an already instantiated :class:`RotatingFrame` object,
        or a valid argument for the constructor of :class:`RotatingFrame`.
        """
        pass

    @property
    @abstractmethod
    def evaluation_mode(self) -> str:
        """Numerical evaluation mode of the model."""
        pass

    @evaluation_mode.setter
    @abstractmethod
    def evaluation_mode(self, new_mode: str):
        """Sets evaluation mode of model."""
        pass

    @abstractmethod
    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        r"""If possible, evaluate the map :math:`\Lambda(t, \cdot)`.

        Args:
            time: Time.
            in_frame_basis: Whether to return the result in the rotating frame basis.
        """
        pass

    @abstractmethod
    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        r"""Evaluate the right hand side :math:`\dot{y}(t) = \Lambda(t, y)`.

        Args:
            time: Time.
            y: State of the differential equation.
            in_frame_basis: Whether `y` is in the rotating frame basis and the results should be
                returned in the rotatign frame basis.
        """
        pass

    def copy(self):
        """Return a copy of self."""
        return copy(self)

    def __call__(
        self, time: float, y: Optional[Array] = None, in_frame_basis: Optional[bool] = False
    ) -> Array:
        r"""Evaluate generator RHS functions. If ``y is None``,
        attemps to evaluate :math:`\Lambda(t, \cdot)`, otherwise, calculates
        :math:`\Lambda(t, y)`

        Args:
            time: Time.
            y: Optional state.
            in_frame_basis: Whether or not to evaluate in the rotating frame basis.

        Returns:
            Array: Either the evaluated model or the RHS for the given y
        """

        if y is None:
            return self.evaluate(time, in_frame_basis=in_frame_basis)

        return self.evaluate_rhs(time, y, in_frame_basis=in_frame_basis)


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
        operators: Array,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        static_operator: Optional[Array] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        evaluation_mode: str = "dense",
    ):
        """Initialize.

        Args:
            operators: A list of operators :math:`G_i`.
            signals: Stores the terms :math:`s_i(t)`. While required for evaluation,
                     :class:`GeneratorModel` signals are not required at instantiation.
            static_operator: Constant part of the generator.
            rotating_frame: Rotating frame operator.
            evaluation_mode: Evaluation mode to use. See ``GeneratorModel.evaluation_mode``
                             for more details. Supported options are:

                                - 'dense' (DenseOperatorCollection)
                                - 'sparse' (SparseOperatorCollection)

        """

        operators = to_array(operators)
        static_operator = to_array(static_operator)

        # initialize internal operator representation
        self._operator_collection = self.construct_operator_collection(
            evaluation_mode, static_operator, operators
        )
        self._evaluation_mode = evaluation_mode

        # set frame
        self._rotating_frame = None
        self.rotating_frame = RotatingFrame(rotating_frame)

        # initialize signal-related attributes
        self.signals = signals

    @property
    def dim(self) -> int:
        if self._operator_collection.static_operator is not None:
            return self._operator_collection.static_operator.shape[-1]
        else:
            return self._operator_collection.operators.shape[-1]

    @property
    def evaluation_mode(self) -> str:
        """Numerical evaluation mode of the model.

        Available options:

            - 'dense': Stores/evaluates operators using dense Arrays.
            - 'sparse': stores/evaluates operators using scipy
            :class:`csr_matrix` types. Not compatible with JAX.
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

        if new_mode != self.evaluation_mode:
            self._operator_collection = self.construct_operator_collection(
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
        new_static_operator, new_operators = self.transfer_operators_between_frames(
            self.get_static_operator(in_frame_basis=True),
            self.get_operators(in_frame_basis=True),
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        self._rotating_frame = new_frame

        self._operator_collection = self.construct_operator_collection(
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
        else:
            # if signals is a list, instantiate a SignalList
            if isinstance(signals, list):
                signals = SignalList(signals)

            # if it isn't a SignalList by now, raise an error
            if not isinstance(signals, SignalList):
                raise QiskitError("Signals specified in unaccepted format.")

            # verify signal length is same as operators
            if len(signals) != self.get_operators().shape[0]:
                raise QiskitError("Signals needs to have the same length as operators.")

            self._signals = signals

    def get_operators(self, in_frame_basis: Optional[bool] = False) -> Array:
        """Get the operators used in the model construction.

        Args:
            in_frame_basis: Flag indicating whether to return the operators
            in the basis in which the rotating frame operator is diagonal.
        Returns:
            The operators in the basis specified by `in_frame_basis`.
        """
        ops = self._operator_collection.operators
        if not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(ops)
        else:
            return ops

    def get_static_operator(self, in_frame_basis: Optional[bool] = False) -> Array:
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

    def set_static_operator(
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

    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.

        Args:
            time: Time to evaluate the model.
            in_frame_basis: Whether to evaluate in the basis in which the rotating frame
            operator is diagonal.

        Returns:
            Array: the evaluated model as a (n,n) matrix

        Raises:
            QiskitError: If model cannot be evaluated.

        """

        if self._signals is None:
            raise QiskitError("GeneratorModel cannot be evaluated without signals.")

        sig_vals = self._signals.__call__(time)

        # Evaluated in frame basis, but without rotations
        op_combo = self._operator_collection(sig_vals)

        # Apply rotations e^{-Ft}Ae^{Ft} in frame basis where F = D
        return self.rotating_frame.operator_into_frame(
            time, op_combo, operator_in_frame_basis=True, return_in_frame_basis=in_frame_basis
        )

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        r"""Evaluate `G(t) @ y`.

        Args:
            time: Time to evaluate the model.
            y: Array specifying system state, in basis specified by
            in_frame_basis.
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal.

        Returns:
            Array defined by :math:`G(t) \times y`.

        Raises:
            QiskitError: If model cannot be evaluated.

        """

        if self._signals is None:
            raise QiskitError("""GeneratorModel cannot be evaluated without signals.""")

        sig_vals = self._signals.__call__(time)

        # Evaluated in frame basis, but without rotations e^{\pm Ft}
        op_combo = self._operator_collection(sig_vals)

        if self.rotating_frame is not None:
            # First, compute e^{tF}y as a pre-rotation in the frame basis
            out = self.rotating_frame.state_out_of_frame(
                time, y, y_in_frame_basis=in_frame_basis, return_in_frame_basis=True
            )
            # Then, compute the product Ae^{tF}y
            out = op_combo @ out
            # Finally, we have the full operator e^{-tF}Ae^{tF}y
            out = self.rotating_frame.state_into_frame(
                time, out, y_in_frame_basis=True, return_in_frame_basis=in_frame_basis
            )
        else:
            return op_combo @ y

        return out

    @classmethod
    def transfer_operators_between_frames(
        cls,
        static_operator: Union[None, Array, csr_matrix],
        operators: Union[None, Array, List[csr_matrix]],
        new_frame: Optional[Union[Array, RotatingFrame]] = None,
        old_frame: Optional[Union[Array, RotatingFrame]] = None,
    ) -> Tuple[Union[None, Array]]:
        """Transform operator data for a GeneratorModel from one frame basis into another.

        This transformation converts ``operators`` from the frame basis of the old frame
        into the frame basis of the new frame, and ``static_operator`` is additionally
        transformed as a generator: the old frame operator is added, and the new one is
        subtracted.

        Args:
            static_operator: Static operator of the model. None is treated as 0.
            operators: Operators of the model. If None, remains as None.
            new_frame: New rotating frame.
            old_frame: Old rotating frame.

        Returns:
            static_operator, operators: Transformed as described above.
        """
        old_frame = RotatingFrame(old_frame)
        new_frame = RotatingFrame(new_frame)

        static_operator = to_numeric_matrix_type(static_operator)
        operators = to_numeric_matrix_type(operators)

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
                        static_operator = diags(old_frame.frame_operator, format='csr')
                    else:
                        static_operator = csr_matrix(old_frame.frame_operator)
                else:
                    if old_frame.frame_operator.ndim == 1:
                        static_operator = np.diag(old_frame.frame_operator)
                    else:
                        static_operator = old_frame.frame_operator

        if operators is not None:
            # If list, loop
            if isinstance(operators, List):
                operators = [old_frame.operator_out_of_frame_basis(op) for op in operators]
            else:
                operators = old_frame.operator_out_of_frame_basis(operators)

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
                    static_operator = diags(-new_frame.frame_diag, format='csr')
                else:
                    static_operator = np.diag(-new_frame.frame_diag)

        if operators is not None:
            # If list, loop
            if isinstance(operators, List):
                operators = [new_frame.operator_into_frame_basis(op) for op in operators]
            else:
                operators = new_frame.operator_into_frame_basis(operators)

        return static_operator, operators

    @classmethod
    def construct_operator_collection(
        cls,
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
            return DenseOperatorCollection(operators=operators, static_operator=static_operator)
        if evaluation_mode == "sparse":
            return SparseOperatorCollection(operators=operators, static_operator=static_operator)

        raise NotImplementedError(
            "Evaluation mode '"
            + str(evaluation_mode)
            + "' is not supported. Call help("
            + str(self.__class__.__name__)
            + ".evaluation_mode) for available options."
        )
