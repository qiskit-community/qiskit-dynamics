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
from typing import Callable, Union, List, Optional
from copy import deepcopy
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.operator_collections import DenseOperatorCollection
from qiskit_dynamics import dispatch
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from .frame import BaseFrame, Frame


class BaseGeneratorModel(ABC):
    r"""BaseGeneratorModel is an abstract interface for a time-dependent operator
    :math:`G(t)`, with functionality of relevance for differential
    equations of the form :math:`\dot{y}(t) = G(t)y(t)`.

    The core functionality is evaluation of :math:`G(t)` and the products
    :math:`AG(t)` and :math:`G(t)A`, for operators :math:`A` of suitable
    shape.

    Additionally, this abstract class requires implementation of 3 properties
    to facilitate the use of this object in solving differential equations:
        - A "drift", which is meant to return the "time-independent" part of
          :math:`G(t)`
        - A "frame", here specified as a :class:`BaseFrame` object, which
          represents an anti-Hermitian operator :math:`F`, specifying
          the transformation :math:`G(t) \mapsto G'(t) = e^{-tF}G(t)e^{tF} - F`.

          If a frame is set, the evaluation functions are modified to work
          with G'(t). Furthermore, all evaluation functions have the option
          to return the results in a basis in which :math:`F` is diagonalized,
          to save on the cost of computing :math:`e^{\pm tF}`.
    """

    @property
    @abstractmethod
    def frame(self) -> BaseFrame:
        """Get the frame."""
        pass

    @frame.setter
    @abstractmethod
    def frame(self, frame: BaseFrame):
        """Set the frame; either an already instantiated :class:`Frame` object
        a valid argument for the constructor of :class:`Frame`, or `None`. 
        Takes care of putting all operators into the basis in which the frame 
        matrix F is diagonal.
        """
        pass

    @abstractmethod
    def evaluate_with_state(
        self, time: float, y: Array, in_frame_basis: Optional[bool] = True
    ) -> Array:
        r"""Given some representation y of the system's state,
        evaluate the RHS of the model y'(t) = \Lambda(y,t)
        at the time t.
        Args:
            time: Time
            y: State in the same basis as the model is
            being evaluated.
            in_frame_basis: boolean flag; True if the
                result should be in the frame basis
                or in the lab basis."""
        pass

    @abstractmethod
    def evaluate_without_state(self, time: float, in_frame_basis: Optional[bool] = True):
        """If possible, expresses the model at time t
        without reference to the state of the system.
        Args:
            time: Time
            in_frame_basis: boolean flag; True if the
                result should be in the frame basis
                or in the lab basis."""
        pass

    def copy(self):
        """Return a copy of self."""
        return deepcopy(self)

    def __call__(
        self, time: float, y: Optional[Array] = None, in_frame_basis: Optional[bool] = True
    ):
        """Evaluate generator RHS functions. If ``y is None``,
        evaluates the model, and otherwise evaluates ``G(t) @ y``.

        Args:
            time: Time.
            y: Optional state.
            in_frame_basis: Whether or not to evaluate in the frame basis.

        Returns:
            Array: Either the evaluated model or the RHS for the given y
        """

        if y is None:
            return self.evaluate_without_state(time, in_frame_basis=in_frame_basis)

        return self.evaluate_with_state(time, y, in_frame_basis=in_frame_basis)


class CallableGenerator(BaseGeneratorModel):
    """Generator specified as a callable"""

    def __init__(
        self,
        generator: Callable,
        frame: Optional[Union[Operator, Array, BaseFrame]] = None,
        drift: Optional[Union[Operator, Array]] = None,
    ):

        self._generator = dispatch.wrap(generator)
        self.frame = frame
        self._drift = drift

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, Array, Frame]):
        """Set the frame; either an already instantiated :class:`Frame` object
        a valid argument for the constructor of :class:`Frame`, or `None`.
        """
        self._frame = Frame(frame)

    @property
    def drift(self) -> Array:
        return self._drift

    def evaluate_with_state(
        self, time: float, y: Array, in_frame_basis: Optional[bool] = True
    ) -> Array:
        return self.evaluate_without_state(time, in_frame_basis) @ y

    def evaluate_without_state(self, time: float, in_frame_basis: Optional[bool] = True) -> Array:
        """Evaluate the model in array format.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            Array: the evaluated model

        Raises:
            QiskitError: If model cannot be evaluated.
        """

        # evaluate generator and map it into the frame
        gen = self._generator(time)
        return self.frame.generator_into_frame(
            time, gen, operator_in_frame_basis=False, return_in_frame_basis=in_frame_basis
        )


class GeneratorModel(BaseGeneratorModel):
    r"""GeneratorModel is a concrete instance of BaseGeneratorModel, where the
    operator :math:`G(t)` is explicitly constructed as:

    .. math::

        G(t) = \sum_{i=0}^{k-1} s_i(t) G_i,

    where the :math:`G_i` are matrices (represented by :class:`Operator`
    objects), and the :math:`s_i(t)` given by signals represented by a
    :class:`SignalList` object, or a list of :class:`Signal` objects.

    The signals in the model can be specified at instantiation, or afterwards
    by setting the ``signals`` attribute, by giving a
    list of :class:`Signal` objects or a :class:`SignalList`.

    For specifying a frame, this object works with the concrete
    :class:`Frame`, a subclass of :class:`BaseFrame`.

    To do:
        insert mathematical description of frame/cutoff_freq handling
    """

    def __init__(
        self,
        operators: Array,
        drift: Optional[Array] = None,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        frame: Optional[Union[Operator, Array, BaseFrame]] = None,
    ):
        """Initialize.

        Args:
            operators: A rank-3 Array of operator components. If 
                a frame object is provided, each operator is assumed
                to be in the basis in which the frame operator is 
                diagonal. 
            signals: Specifiable as either a SignalList, a list of
                Signal objects, or as the inputs to signal_mapping.
                GeneratorModel can be instantiated without specifying
                signals, but it can not perform any actions without them.
            frame: Rotating frame operator. If specified with a 1d
                array, it is interpreted as the diagonal of a
                diagonal matrix.
        """

        # initialize internal operator representation in the frame basis
        self._operator_collection = DenseOperatorCollection(operators,drift)

        # set frame. 
        self._frame = None
        self.frame = frame

        # initialize signal-related attributes
        self._signals = None
        self.signals = signals

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
            if len(signals) != self._operator_collection.num_operators:
                raise QiskitError(
                    """Signals needs to have the same length as
                                    operators."""
                )

            self._signals = signals

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, Array, Frame]):
        if self._frame is not None and self._frame.frame_diag is not None:
            self._operator_collection.drift = self._operator_collection.drift + Array(np.diag(self._frame.frame_diag))
            self._operator_collection.apply_function_to_operators(self.frame.operator_out_of_frame_basis)

        self._frame = Frame(frame)
        if self._frame.frame_diag is not None:
            self._operator_collection.apply_function_to_operators(self.frame.operator_into_frame_basis)
            self._operator_collection.drift = self._operator_collection.drift - Array(np.diag(self._frame.frame_diag))

    def evaluate_without_state(self, time: float, in_frame_basis: Optional[bool] = True) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.
        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal
        Returns:
            Array: the evaluated model as a (n,n) matrix
        Raises:
            QiskitError: If model cannot be evaluated."""

        if self._signals is None:
            raise QiskitError("""GeneratorModel cannot be evaluated without signals.""")

        sig_vals = np.real(self._signals.complex_value(time))

        # Evaluated in frame basis, but without rotations e^{\pm Ft}
        op_combo = self._operator_collection(sig_vals)

        if in_frame_basis:
            return op_combo
        else:
            return self.frame.operator_out_of_frame_basis(op_combo)

    def evaluate_with_state(
        self, time: Union[float, int], y: Array, in_frame_basis: Optional[bool] = True
    ) -> Array:
        """Evaluate the model in array format as a vector, given the current state.
        Args:
            time: Time to evaluate the model
            y: (n) Array specifying system state, in basis choice specified by
                in_frame_basis. If not in frame basis, assumed to not include
                the rotating term e^{-Ft}. If in the frame basis, assumed to
                include the rotating term e^{-Ft}.
            in_frame_basis: Whether to evaluate in the basis in which the frame
                operator is diagonal
        Returns:
            Array: the evaluated model as (n) vector
        Raises:
            QiskitError: If model cannot be evaluated.
        """

        if not in_frame_basis:
            y = self.frame.state_into_frame(
                time, y, y_in_frame_basis=False, return_in_frame_basis=True
            )

        if self._signals is None:
            raise QiskitError("""GeneratorModel cannot be evaluated without signals.""")

        sig_vals = np.real(self._signals.complex_value(time))

        # Evaluated in frame basis, but without rotations e^{\pm Ft}
        op_combo = self._operator_collection(sig_vals)

        if self.frame.frame_diag is None:
            return np.dot(op_combo, y)
        else:
            # perform pre-rotation
            out = np.exp(time * self.frame.frame_diag) * y
            # apply operator
            out = np.dot(op_combo, out)
            # apply post-rotation
            out = np.exp(-time * self.frame.frame_diag) * out

        if not in_frame_basis:
            out = self.frame.state_out_of_frame(time, out, y_in_frame_basis=True)

        return out

    def _reset_internal_ops(self):
        """Helper function to be used by various setters whose value changes
        require reconstruction of the internal operators.
        """
        self.frame = self._frame