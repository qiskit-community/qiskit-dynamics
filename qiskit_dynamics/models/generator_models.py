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
from copy import deepcopy
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.operator_collections import (
    DenseOperatorCollection,
    SparseOperatorCollection,
)
from qiskit_dynamics import dispatch
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from .rotating_frame import RotatingFrame


class BaseGeneratorModel(ABC):
    r"""BaseGeneratorModel is an abstract interface for a time-dependent
    linear differential equation of the form :math:`\dot{y}(t) = \Lambda(y,t)`,
    where :math:`\Lambda` is linear in :math:`y`.

    The core functionality is evaluation of :math:`\Lambda(y,t)`, as well as,
    if possible, a representation :math:`\Lambda(t)` of the linear map :math:`\Lambda(y,t)`
    independent of :math:`y(t)`.

    Additionally, this abstract class requires implementation of 3
    properties to facilitate the use of this object in solving differential equations:
        - An "operator collection," stored as a subclass of :class:`BaseOperatorCollection`,
        which will handle almost all of the numerical evaluation of :math:`\Lambda`,
        except for frame transformations. Having multiple types of :class:`OperatorCollection`
        supported by a single model can enable multiple evaluation modes, like
        using sparse arrays.
        - A "drift," which stores time-independent parts of :math:`\Lambda`,
        typically terms added to the Hamiltonian of a system.
        - A "rotating frame," here specified as a :class:`RotatingFrame` object, representing
        an antihermitian operator :math:`F`, specifying a transformation law for states
        and operators, typically through multiplication or conjugation by :math:`e^{tF}`.
        If a frame F is specified, all internal calculations should be done in the basis
        in which :math:`F` is diagonal, so as to more quickly calculate :math:`e^{\pm tF}`.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Gets Hilbert space dimension."""
        pass

    @abstractmethod
    def get_operators(
        self, in_frame_basis: Optional[bool] = False
    ) -> Union[Array, Tuple[Array], Callable]:
        """Get the operators used for calculating the model's value.
        Args:
            in_frame_basis: Flag for whether the returned operators should be
            in the basis in which the rotating frame operator is diagonal.
        Returns:
            The operators in the basis specified by in_frame_basis"""
        pass

    def get_drift(self, in_frame_basis: Optional[bool] = False) -> Array:
        """Gets the drift term. If a frame F has been specified, this
        drift will include any contributions (typically -F) from the frame.
        Args:
            in_frame_basis: Flag for whether the returned drift should be
            in the basis in which the frame is diagonal.
        Returns:
            The drift term."""
        if not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(self._drift)
        else:
            return self._drift

    def set_drift(
        self,
        new_drift: Array,
        operator_in_frame_basis: Optional[bool] = False,
    ):
        """Sets drift term. The drift term will be transformed into the rotating
        frame basis. If in a rotating frame, note that this will NOT automatically
        add the -F or -iF term due to the frame shift. 

        Args:
            new_drift: The drift operator.
            operator_in_frame_basis: Whether new_drift is already in the rotating
            frame basis."""
        if new_drift is None:
            new_drift = np.zeros((self.dim, self.dim))

        new_drift = Array(np.array(new_drift))

        if not operator_in_frame_basis and self.rotating_frame is not None:
            new_drift = self.rotating_frame.operator_into_frame_basis(new_drift)
        self._drift = new_drift
        # pylint: disable=no-member
        if self._operator_collection is not None:
            # pylint: disable=no-member
            self._operator_collection.drift = new_drift

    @property
    def evaluation_mode(self) -> str:
        """Returns the current implementation mode,
        e.g. sparse/dense, vectorized/not"""
        # pylint: disable=no-member
        return self._evaluation_mode

    @abstractmethod
    def set_evaluation_mode(self, new_mode: str):
        """Sets evaluation mode of model.
        Will replace _operator_collection with the
        correct type of operator collection.

        Instances of this function should
        include important details about each
        evaluation mode."""
        pass

    @property
    def rotating_frame(self) -> RotatingFrame:
        """Get the rotating frame."""
        return self.rotating_frame

    @rotating_frame.setter
    @abstractmethod
    def rotating_frame(self, rotating_frame: RotatingFrame):
        """Set the rotating frame; either an already instantiated :class:`RotatingFrame` object
        a valid argument for the constructor of :class:`RotatingFrame`, or `None`.
        Takes care of putting all operators into the basis in which the frame
        matrix F is diagonal.
        """
        pass

    @abstractmethod
    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        r"""Given some representation y of the system's state,
        evaluate the RHS of the model :math:`\dot{y}(t) = \Lambda(y,t)`
        at the time t.
        Args:
            time: Time
            y: State in the same basis as the model is
            being evaluated.
            in_frame_basis: boolean flag; True if the
                result should be in the rotating frame basis
                or in the lab basis."""
        pass

    @abstractmethod
    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """If possible, expresses the model at time t
        without reference to the state of the system.
        Args:
            time: Time
            in_frame_basis: boolean flag; True if the
                result should be in the rotating frame basis
                or in the lab basis."""
        pass

    def copy(self):
        """Return a copy of self."""
        return deepcopy(self)

    def __call__(
        self, time: float, y: Optional[Array] = None, in_frame_basis: Optional[bool] = False
    ) -> Array:
        """Evaluate generator RHS functions. If ``y is None``,
        tries to evaluate :math:`\Lambda(t)` using the desired
        representation of the linear map. Otherwise, calculates
        :math:`\Lambda(y,t)`

        Args:
            time: Time.
            y: Optional state.
            in_frame_basis: Whether or not to evaluate in the rotating frame basis.

        Returns:
            Array: Either the evaluated model or the RHS for the given y
        """

        if y is None:
            return self.evaluate_generator(time, in_frame_basis=in_frame_basis)

        return self.evaluate_rhs(time, y, in_frame_basis=in_frame_basis)


class CallableGenerator(BaseGeneratorModel):
    r"""Specifies the model as a purely LMDE, with a callable generator.
    That is to say, define :math:`\dot{y}=\Lambda(y,t)=G(t)y` for some
    callable generator :math:`G(t)`."""

    def __init__(
        self,
        generator: Callable,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        drift: Optional[Union[Operator, Array]] = None,
        dim: Optional[int] = None,
    ):

        self._generator = dispatch.wrap(generator)
        self.rotating_frame = rotating_frame
        self._drift = drift
        self._evaluation_mode = "callable_generator"
        self._operator_collection = None
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        else:
            raise ValueError(
                "Dimension of CallableGenerator object should be specified at initialization."
            )

    def get_drift(self, in_frame_basis: Optional[bool] = False) -> Array:
        if in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_into_frame_basis(self._drift)
        else:
            return self._drift

    def get_operators(self, in_frame_basis: Optional[bool] = False) -> Callable:
        if in_frame_basis and self.rotating_frame is not None:
            # The callable generator is assumed to be in the lab basis. Need to transform into
            # the frame basis.
            f = lambda t: self.rotating_frame.operator_into_frame_basis(self._generator(t))
            return f
        else:
            return self._generator

    def set_drift(
        self,
        new_drift: Array,
        operator_in_frame_basis: Optional[bool] = False,
        includes_frame_contribution: Optional[bool] = False,
    ):
        # Subtracting the frame operator from the generator is handled at evaluation time.
        if operator_in_frame_basis and self.rotating_frame is not None:
            self._drift = self.rotating_frame.operator_out_of_frame_basis(new_drift)
        else:
            self._drift = new_drift

    def get_frame_contribution(self):
        raise ValueError("Frame Contribution for CallableGenerator is not well-defined")

    def set_evaluation_mode(self, new_mode: str):
        """Setting the evaluation mode for CallableGenerator
        is not supported."""
        raise NotImplementedError(
            "Setting implementation mode for CallableGenerator is not supported."
        )

    @property
    def rotating_frame(self) -> RotatingFrame:
        """Return the frame."""
        return self._rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]):
        """Set the frame; either an already instantiated :class:`RotatingFrame` object
        a valid argument for the constructor of :class:`RotatingFrame`, or `None`.
        """
        self._rotating_frame = RotatingFrame(rotating_frame)

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        return self.evaluate_generator(time, in_frame_basis=in_frame_basis) @ y

    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
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

        # evaluate generator and map it into the rotating frame
        gen = self._generator(time)
        if self._drift is not None:
            gen = gen + self._drift
        return self.rotating_frame.generator_into_frame(
            time, gen, operator_in_frame_basis=False, return_in_frame_basis=in_frame_basis
        )


class GeneratorModel(BaseGeneratorModel):
    r"""GeneratorModel is a concrete instance of BaseGeneratorModel, where the
    map :math:`\Lambda(y,t)` is explicitly constructed as:

    .. math::
        \Lambda(y,t) = G(t)y,

        G(t) = \sum_i s_i(t) G_i + G_d

    where the :math:`G_i` are matrices (represented by :class:`Operator`
    or :class:`Array` objects), the :math:`s_i(t)` are signals represented by
    a :class:`SignalList` object, or a list of :class:`Signal` objects, and
    :math:`G_d` is the drift term of the generator, and constant in time.

    The signals in the model can be specified at instantiation, or afterwards
    by setting the ``signals`` attribute, by giving a
    list of :class:`Signal` objects or a :class:`SignalList`. Rotating frames
    should be specified by providing a :class:`RotatingFrame` object.
    """

    def __init__(
        self,
        operators: Array,
        signals: Optional[Union[SignalList, List[Signal]]] = None,
        drift: Optional[Array] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        evaluation_mode: str = "dense",
    ):
        """Initialize.

        Args:
            operators: A rank-3 Array of generator components :math:`G_i`. If
            a rotating frame object is provided, these will be transformed
            into the basis in which the frame operator is diagonal.
            drift: Optional, constant terms to add to G. Useful for
            frame transformations. If a rotating frame is provided, the
            drift term will be decreased by F.
            signals: Stores the terms :math:`s_i(t)`. Specifiable as either a
            SignalList, a list of Signal objects, or as the inputs to
            signal_mapping. GeneratorModel can be instantiated without
            specifying signals, but it can not perform any actions without them.
            rotating_frame: Rotating frame operator. If specified with a 1d
            array, it is interpreted as the diagonal of a
            diagonal matrix.
            evaluation_mode: Flag for what type of evaluation should
            be used. Currently supported options are
                dense (DenseOperatorCollection)
                sparse (SparseOperatorCollection)
            See GeneratorModel.set_evaluation_mode for more details.
        """

        # initialize internal operator representation
        self._operator_collection = None
        self._operators = Array(np.array(operators))
        self._drift = None
        self._evaluation_mode = None
        self.set_drift(drift, operator_in_frame_basis=True)

        # set frame and transform operators into frame basis.
        self._rotating_frame = None
        self.rotating_frame = RotatingFrame(rotating_frame)

        # initialize signal-related attributes
        self._signals = None
        self.signals = signals

        self.set_evaluation_mode(evaluation_mode)

    def get_operators(self, in_frame_basis: Optional[bool] = False) -> Array:
        if not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(self._operators)
        else:
            return self._operators

    def get_frame_contribution(self):
        return Array(np.diag(-1 * self.rotating_frame.frame_diag))

    @property
    def dim(self) -> int:
        return self._operators.shape[-1]

    def set_evaluation_mode(self, new_mode: str):
        """Sets evaluation mode to new_mode.
        Args:
            new_mode: string specifying new mode, with options
                dense: stores/evaluates operators using only
                    dense Array objects.
                sparse: stores/evaluates operators using scipy
                    :class:`csr_matrix` types. Can be faster/less
                    memory intensive than dense if Hamiltonian components
                    are mathematically sparse. If evaluating the generator
                    with a 2d frame operator (non-diagonal), all generators
                    will be returned as dense matrices. Not compatible
                    with jax.
        Raises:
            NotImplementedError: if new_mode is not one of the above
            supported evaluation modes."""
        if new_mode == "dense":
            self._operator_collection = DenseOperatorCollection(
                self.get_operators(True), drift=self.get_drift(True)
            )
        elif new_mode == "sparse":
            self._operator_collection = SparseOperatorCollection(
                self.get_operators(True), self.get_drift(True)
            )
        elif new_mode is None:
            self._operator_collection = None
        else:
            raise NotImplementedError("Evaluation Mode " + str(new_mode) + " is not supported.")
        self._evaluation_mode = new_mode

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
            if len(signals) != self.get_operators(True).shape[0]:
                raise QiskitError(
                    """Signals needs to have the same length as
                                    operators."""
                )

            self._signals = signals

    @property
    def rotating_frame(self) -> RotatingFrame:
        """Return the rotating frame."""
        return self._rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]):
        if self._rotating_frame is not None and self._rotating_frame.frame_diag is not None:
            self._drift = self._drift + Array(np.diag(self._rotating_frame.frame_diag))
            self._operators = self.rotating_frame.operator_out_of_frame_basis(self._operators)
            self._drift = self.rotating_frame.operator_out_of_frame_basis(self._drift)

        self._rotating_frame = RotatingFrame(rotating_frame)

        if self._rotating_frame.frame_diag is not None:
            self._operators = self.rotating_frame.operator_into_frame_basis(self._operators)
            self._drift = self.rotating_frame.operator_into_frame_basis(self._drift)
            self._drift = self._drift - Array(np.diag(self._rotating_frame.frame_diag))

        # Reset internal operator collection
        self.set_evaluation_mode(self.evaluation_mode)

    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluate the model in array format as a matrix, independent of state.
        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the rotating frame
                            operator is diagonal
        Returns:
            Array: the evaluated model as a (n,n) matrix
        Raises:
            QiskitError: If model cannot be evaluated."""

        if self._signals is None:
            raise QiskitError("GeneratorModel cannot be evaluated without signals.")

        # pylint: disable=not-callable
        sig_vals = self._signals(time)

        # Evaluated in frame basis, but without rotations
        op_combo = self._operator_collection(sig_vals)

        # Apply rotations e^{-Ft}Ae^{Ft} in frame basis where F = D
        return self.rotating_frame.operator_into_frame(
            time, op_combo, operator_in_frame_basis=True, return_in_frame_basis=in_frame_basis
        )

    def evaluate_rhs(self, time: float, y: Array, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluate the model in array format using left multiplication, as
        G(t) @ y, given the current state y.
        Args:
            time: Time to evaluate the model
            y: Array specifying system state, in basis specified by
                in_frame_basis.
            in_frame_basis: Whether to evaluate in the basis in which the frame
                operator is diagonal
        Returns:
            Array defined by :math:`G(t)y`.
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
