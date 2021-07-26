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
from qiskit_dynamics.signals import Signal, SignalList,SignalSum
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
    def hilbert_space_dimension(self) -> int:
        """Gets Hilbert space dimension."""
        pass

    @property
    @abstractmethod
    def operators(self) -> Union[Array, List[Array]]:
        """Get the originally passed operators by the user"""
        pass

    @property
    def drift(self) -> Array:
        """Gets the originally passed drift term"""
        return self._drift

    @drift.setter
    def drift(self, new_drift: Array):
        """Sets drift term."""
        if new_drift is None:
            new_drift = np.zeros((self.hilbert_space_dimension, self.hilbert_space_dimension))

        new_drift = Array(np.array(new_drift))
        self._drift = new_drift
        # pylint: disable=no-member
        if self._operator_collection is not None:
            # pylint: disable=no-member
            self._operator_collection.drift = new_drift

    @property
    @abstractmethod
    def evaluation_mode(self) -> str:
        """Returns the current implementation mode,
        e.g. sparse/dense, vectorized/not"""
        return self._evaluation_mode

    @evaluation_mode.setter
    @abstractmethod
    def evaluation_mode(self, new_mode: str):
        """Sets evaluation mode of model.
        Will replace _operator_collection with the
        correct type of operator collection"""
        self._evaluation_mode = new_mode
        pass

    @property
    @abstractmethod
    def carrier_freqs(self) -> list[float]:
        """Gets the list of signal frequencies"""

    @abstractmethod
    def frame(self) -> BaseFrame:
        """Get the frame."""
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
        self._evaluation_mode = "callable_generator"
        self._operator_collection = None

    @property
    def hilbert_space_dimension(self) -> int:
        return self._generator(0).shape[-1]

    @property
    def operators(self) -> Callable:
        return self._generator

    @property
    def evaluation_mode(self) -> str:
        return self._evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, new_mode: str):
        raise NotImplementedError(
            "Setting implementation mode for CallableGenerator is not supported."
        )

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

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        return self.evaluate_generator(time, in_frame_basis=in_frame_basis) @ y

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
            time, gen, operator_in_frame=False, return_in_frame_basis=in_frame_basis
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
        evaluation_mode: str = "dense_operator_collection",
    ):
        """Initialize.

        Args:
            operators: A rank-3 Array of operator components. If
                a frame object is provided, each operator is assumed
                to be in the basis in which the frame operator is
                diagonal.
            drift: Optional, constant terms to add to G. Useful for
                frame transformations. If a frame, but not a drift,
                is provided, will be set to -F. If both are provided,
                the drift will be set to drift - F.
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

    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
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

        if self.frame is not None:
            # First, compute e^{tF}y as a pre-rotation in the frame basis
            out = self.frame.state_out_of_frame(
                time, y, y_in_frame_basis=in_frame_basis, return_in_frame_basis=True
            )
            # Then, compute the product Ae^{tF}y
            out = np.dot(op_combo, out)
            # Finally, we have the full operator e^{-tF}Ae^{tF}y
            out = self.frame.state_into_frame(
                time, out, y_in_frame_basis=True, return_in_frame_basis=in_frame_basis
            )
        else:
            return np.dot(op_combo, y)

        return out

def perform_rotating_wave_approximation(model: GeneratorModel,cutoff_freq: Union[float,int]) -> GeneratorModel:
    r"""Performs RWA on a GeneratorModel so that all terms that 
    rotate with complex frequency larger than cutoff_freq are 
    discarded. In particular, let G(t) = \sum_j s_j(t) G_j + G_d,
    and let us consider it in the rotating frame, so that the 
    actual operator that is applied is given by e^{-tF}(G(t) - F)e^{tF} 
    = \sum_js_j(t)e^{-tF}G_je^{tF} + e^{-tF}(G_d - F)e^{tF}
    Args:
        model: The GeneratorModel to which you 
        wish to apply the RWA
        cutoff_freq: the maximum (magnitude) of 
        frequency you wish to allow.
    Returns 
        GeneratorModel with twice as many terms
        and some signals with negative frequencies
        
    Formalism: When we consider e^{-tF}A e^{tF} in the basis in which F 
    is diagonal, we may write conjugation by e^{\pm tF} as elementwise 
    multiplication s.t. (e^{-tF} A e^{tF})_{jk} = e^{(-d_j+d_k)t}*A_{jk}.
    When we write e^{-tF}(G(t)-F)e^{tF} = \sum_i e^{-tF} s_i(t)G_ie^{tF}
    + e^{-tF}(G_d-F)e^{tF} in order to take the RWA in the rotating frame,
    we must consider the effect of rotations caused by the frame, with 
    frequency Im(-d_j-d_k)/(2*pi), as well as those due to the signal's
    rotation itself, where we write s_i(t) = Re[a_i(t)e^{i(2*pi*nu_i*t+phi_i)}]
    = a_i(t)e^{i(2*pi*nu_i*t-phi_i)}/2 + \bar{a_i(t)}e^{-i(2*pi*nu_i*t+phi_i)}/2. With this
    in mind, consider a term of the form (e^{-tF}s_i(t)G_ie^{tF})_{jk} 
    = e^{(-d_j+d_k)t}(G_i)_{jk}[a_i(t)e^{i(2*pi*nu_i*t+phi_i)}
    +\bar{a_i(t)}e^{-i(2*pi*nu_i*t+phi_i)}]. The \pm i(...) term has 
    effective frequency \pm nu_i + Im(-d_j+d_k)/(2*pi) = \pm nu_i + f_{jk}, 
    (we neglect any oscillation/rotation in a_i for now) and we wish to neglect 
    all terms with effective frequency greater in magnitude than some cutoff 
    frequency nu_*. As such, let us write G_i = A_i + B_i + C_i + D_i, with 
    A_i the matrix representing terms/elements which have 
    1) abs(nu_i+f_{jk})<nu_* and 2) abs(-nu_i+f_{jk})<nu_*, B_i the matrix 
    of terms where 1) but not 2), C_i the matrix of terms where 2) but not 1), 
    and D_i the terms where neither 1) nor 2) hold. Thus, after the RWA, 
    we may write our term as s_i(t) G_i -> a_i(t)e^{i(2*pi*nu_i*t+phi_i)}(A_i+B_i)/2
    + \bar{a_i(t)}e^{-i(2*pi*nu_i*t+phi_i)}(A_i+C_i)/2 = s_i(t) A_i + 
    [Re[a_i]cos(2*pi*nu_i*t+phi_i)-Im[a_i]sin(2*pi*nu_i*t+phi_i)]B_i/2 + 
    [Im[a_i]cos(2*pi*nu_i*t+phi_i)+Re[a_i]sin(2*pi*nu_i*t+phi_i)]iB_i/2 + 
    [Re[a_i]cos(2*pi*nu_i*t+phi_i)-Im[a_i]sin(2*pi*nu_i*t+phi_i)]C_i/2 + 
    [Im[a_i]cos(2*pi*nu_i*t+phi_i)-Re[a_i]sin(2*pi*nu_i*t+phi_i)](-iC_i/2)
    = s_i(t)A_i + Re[a_i(t)e^{i(2*pi*nu_i*t+phi_i)}](B_i+C_i)/2 + 
    Re[a_i(t)e^{i(2*pi*nu_i*t+phi_i-pi/2)}](iB_i-iC_i)/2 = 
    s_i(t)(A_i + B_i/2 + C_i/2) + s'_i(t)(iB_i-iC_i)/2 where s'_i
    is a Signal with frequency nu_i, amplitude a_i, and phase phi_i - pi/2.

    Next, note that the drift terms (G_d)_{jk} have effective frequency 
    Im[-d_j+d_k]/(2*pi). Note that this vanishes on the diagonal, 
    so the -F term may be ignored for any nonzero cutoff_freq.
    
    """

    if model.frame is None or model.frame.frame_diag is None:
        return model

    n = model.hilbert_space_dimension
    diag = model.frame.frame_diag
    
    diff_matrix = np.broadcast_to(diag,(n,n)) - np.broadcast_to(diag,(n,n)).T
    frame_freqs = diff_matrix.imag/(2*np.pi)

    new_drift = model.frame.operator_out_of_frame_basis(model.drift+np.diag(model.frame.frame_diag))
    new_drift = deepcopy(new_drift*(abs(frame_freqs)<cutoff_freq).astype(int))

    num_components = len(model.signals)
    frame_freqs = np.broadcast_to(frame_freqs,(num_components,n,n))
    
    carrier_freqs = []
    for sig in model.signals.components:
        carrier_freqs.append(sig.carrier_freq)
    carrier_freqs = np.array(carrier_freqs).reshape((num_components,1,1))
    
    pos_pass = np.abs(carrier_freqs + frame_freqs) < cutoff_freq
    neg_pass = np.abs(-carrier_freqs+ frame_freqs) < cutoff_freq
    A = model.operators*(pos_pass & neg_pass).astype(int)
    B = model.operators*(pos_pass & np.logical_not(neg_pass)).astype(int)
    C = model.operators*(np.logical_not(pos_pass) & neg_pass).astype(int)
    normal_operators = A + B/2 + C/2
    model_signal_sums = model.signals.components
    abnormal_operators = 1j*B/2-1j*C/2
    normal_signals = []
    abnormal_signals = []
    for sig_sum in model_signal_sums:
        if len(sig_sum.components)>1:
            raise NotImplementedError("RWA where the coefficients s_j are not pure Signal objects is not currently supported.")
        sig = sig_sum.components[0]
        normal_signals.append(sig)
        abnormal_signals.append(SignalSum(Signal(sig.envelope,sig.carrier_freq,sig.phase-np.pi/2)))
    new_signals = deepcopy(SignalList(normal_signals + abnormal_signals))
    new_operators = deepcopy(model.frame.operator_out_of_frame_basis(np.append(normal_operators,abnormal_operators,axis=0)))

    new_model = GeneratorModel(new_operators,drift=new_drift,signals=new_signals,frame=deepcopy(model.frame.frame_operator))
    return new_model

        Returns:
            Array: operator model evaluated for a given list of signal values
        """

        return 0.5 * (
            self._fb_op_collection(sig_vals) + self._fb_op_conj_collection(sig_vals.conj())
        )
