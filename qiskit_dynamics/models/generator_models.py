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
from qiskit_dynamics.type_utils import to_array
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
        - A `cutoff_freq`: a cutoff frequency for further modifying the
          evaluation of :math:`G'(t)` to remove terms above a given frequency.
          In this abstract class the exact meaning of this is left unspecified;
          it is left to concrete implementations to define this.
    """

    @property
    @abstractmethod
    def frame(self) -> BaseFrame:
        """Get the frame."""
        pass

    @frame.setter
    @abstractmethod
    def frame(self, frame: BaseFrame):
        """Set the frame."""
        pass

    @property
    @abstractmethod
    def cutoff_freq(self) -> float:
        """Get cutoff frequency."""
        pass

    @cutoff_freq.setter
    @abstractmethod
    def cutoff_freq(self, cutoff_freq: float):
        """Set cutoff frequency."""
        pass

    @property
    @abstractmethod
    def carrier_freqs(self) -> list[float]:
        """Gets the list of signal frequencies"""

    @abstractmethod
    def evaluate(self, time: float, in_frame_basis: bool = False) -> Array:
        """Evaluate the model at a given time."""
        pass

    @property
    @abstractmethod
    def drift(self) -> Array:
        """Evaluate the constant part of the model."""
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
    def cutoff_freq(self) -> float:
        """Return the cutoff frequency."""
        return None

    @cutoff_freq.setter
    def cutoff_freq(self, cutoff_freq: float):
        """Cutoff frequency not supported for generic."""
        if cutoff_freq is not None:
            raise QiskitError("""Cutoff frequency is not supported for function-based generator.""")

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
        cutoff_freq: Optional[float] = None,
    ):
        """Initialize.

        Args:
            operators: A rank-3 Array of operator components.
            signals: Specifiable as either a SignalList, a list of
                Signal objects, or as the inputs to signal_mapping.
                GeneratorModel can be instantiated without specifying
                signals, but it can not perform any actions without them.
            frame: Rotating frame operator. If specified with a 1d
                array, it is interpreted as the diagonal of a
                diagonal matrix.
            cutoff_freq: Frequency cutoff when evaluating the model.
        """

        # initialize internal operator representation in the frame basis
        self._fb_op_collection = DenseOperatorCollection(operators, drift)
        self._fb_op_conj_collection = DenseOperatorCollection(operators, drift)
        self._band_aid_temporary_operator_collection = DenseOperatorCollection(operators, drift)

        # set cutoff frequency and frame. Must be done in this order.
        self._frame = None
        self._cutoff_freq = None

        self.frame = frame
        self.cutoff_freq = cutoff_freq

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
            if len(signals) != self._fb_op_collection.num_operators:
                raise QiskitError(
                    """Signals needs to have the same length as
                                    operators."""
                )

            if self._signals is not None:
                # compare flattened carrier frequencies
                old_carrier_freqs = [sig.carrier_freq for sig in self._signals.flatten()]
                new_carrier_freqs = [sig.carrier_freq for sig in signals.flatten()]
                if (
                    not np.allclose(old_carrier_freqs, new_carrier_freqs)
                    and self._cutoff_freq is not None
                ):
                    self.apply_cutoff_filtering()

            self._signals = signals

    @property
    def frame(self) -> Frame:
        """Return the frame."""
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, Array, Frame]):
        """Set the frame; either an already instantiated :class:`Frame` object
        a valid argument for the constructor of :class:`Frame`, or `None`.
        """
        if self._frame is not None and self._frame.frame_diag is not None:
            self._undo_frame_basis(Array(np.diag(self._frame.frame_diag)))

        self._frame = Frame(frame)
        if self._frame.frame_diag is not None:
            self._apply_frame_basis(Array(-np.diag(self._frame.frame_diag)))

        if self.cutoff_freq is not None:
            self.apply_cutoff_filtering()

    def _undo_frame_basis(self, drift_term: Optional[Array]):
        self._fb_op_collection.drift = self._fb_op_collection.drift + drift_term
        self._fb_op_collection.apply_function_to_operators(self._frame.operator_out_of_frame_basis)

        self._fb_op_conj_collection.drift = self._fb_op_conj_collection.drift + drift_term
        self._fb_op_conj_collection.apply_function_to_operators(
            self._frame.operator_out_of_frame_basis
        )

    def _apply_frame_basis(self, drift_term: Optional[Array]):
        self._fb_op_collection.apply_function_to_operators(self._frame.operator_into_frame_basis)
        self._fb_op_collection.drift = self._fb_op_collection.drift + drift_term

        self._fb_op_conj_collection.apply_function_to_operators(
            self._frame.operator_into_frame_basis
        )
        self._fb_op_conj_collection.drift = self._fb_op_conj_collection.drift + drift_term

    @property
    def cutoff_freq(self) -> float:
        """Return the cutoff frequency."""
        return self._cutoff_freq

    @cutoff_freq.setter
    def cutoff_freq(self, cutoff_freq: float):
        """Set the cutoff frequency."""
        if cutoff_freq != self._cutoff_freq:
            self._cutoff_freq = cutoff_freq
        if self._cutoff_freq is not None:
            self.apply_cutoff_filtering()

    def apply_cutoff_filtering(self):
        """Filters the two stored operator collections
        according to the stored cutoff frequency"""

        cutoff_filter = self.frame.calculate_cutoff_filter(
            self.carrier_freqs, self._cutoff_freq, self._fb_op_collection.hilbert_space_dimension
        )

        self._fb_op_collection.filter_arrays(cutoff_filter)
        self._fb_op_conj_collection.filter_arrays(cutoff_filter.transpose([0, 2, 1]))

    def evaluate_without_state(self, time: float, in_frame_basis: Optional[bool] = True) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            Array: the evaluated model as a (n,n) matrix

        Raises:
            QiskitError: If model cannot be evaluated.
        """

        if self._signals is None:
            raise QiskitError("""GeneratorModel cannot be evaluated without signals.""")

        sig_vals = self._signals.complex_value(time)

        # evaluate the linear combination in the frame basis with cutoffs,
        # then map into the frame

        # Evaluated in frame basis, but without rotations e^{\pm Ft}
        op_combo = self._evaluate_in_frame_basis_with_cutoffs(sig_vals)

        return self.frame.generator_into_frame(
            time, op_combo, operator_in_frame_basis=True, return_in_frame_basis=in_frame_basis
        )

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

        sig_vals = self._signals.complex_value(time)

        # evaluate the linear combination in the frame basis with cutoffs,
        # then map into the frame

        # Evaluated in frame basis, but without rotations e^{\pm Ft}
        op_combo = self._evaluate_in_frame_basis_with_cutoffs(sig_vals)

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

    @property
    def carrier_freqs(self) -> list[float]:
        """Returns the list of frequencies used by the model's SignalList"""
        carrier_freqs = None
        if self._signals is None:
            carrier_freqs = np.zeros(self._fb_op_collection.num_operators)
        else:
            carrier_freqs = [sig.carrier_freq for sig in self._signals.flatten()]
        return carrier_freqs

    def _reset_internal_ops(self):
        """Helper function to be used by various setters whose value changes
        require reconstruction of the internal operators.
        """
        self.frame = self._frame
        self.cutoff_freq = self._cutoff_freq

    def _evaluate_in_frame_basis_with_cutoffs(self, sig_vals: Array):
        """Evaluate the operator in the frame basis with frequency cutoffs.
        The computation here corresponds to that prescribed in
        `Frame.operators_into_frame_basis_with_cutoff`.

        Args:
            sig_vals: Signals evaluated at some time.

        Returns:
            Array: operator model evaluated for a given list of signal values
        """

        return 0.5 * (
            self._fb_op_collection(sig_vals) + self._fb_op_conj_collection(sig_vals.conj())
        )
