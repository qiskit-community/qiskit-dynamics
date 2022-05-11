# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

r"""
Solver classes.
"""


from typing import Optional, Union, Tuple, Any, Type, List, Callable
from copy import copy
import warnings

import numpy as np

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator, Statevector, DensityMatrix

from qiskit_dynamics.models import (
    BaseGeneratorModel,
    HamiltonianModel,
    LindbladModel,
    RotatingFrame,
    rotating_wave_approximation,
)
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.array import Array

from .solver_functions import solve_lmde
from .solver_utils import is_lindblad_model_vectorized, is_lindblad_model_not_vectorized

try:
    import jax.numpy as jnp
    from jax import jit
    from jax.lax import switch, scan
except ImportError:
    pass

class Solver:
    ###################################################################################################
    # Update doc!
    ###################################################################################################

    """Solver class for simulating both Hamiltonian and Lindblad dynamics, with high
    level type-handling of input states.

    Given the components of a Hamiltonian and optional dissipators, this class will
    internally construct either a :class:`HamiltonianModel` or :class:`LindbladModel`
    instance.

    Transformations on the model can be specified via the optional arguments:

    * ``rotating_frame``: Transforms the model into a rotating frame. Note that
      operator specifying the frame will be substracted from the static_hamiltonian.
      If supplied as a 1d array, ``rotating_frame`` is interpreted as the diagonal
      elements of a diagonal matrix. See :class:`~qiskit_dynamics.models.RotatingFrame` for details.
    * ``in_frame_basis``: Whether to represent the model in the basis in which the frame
      operator is diagonal, henceforth called the "frame basis".
      If ``rotating_frame`` is ``None`` or was supplied as a 1d array,
      this kwarg has no effect. If ``rotating_frame`` was specified as a 2d array,
      the frame basis is hte diagonalizing basis supplied by ``np.linalg.eigh``.
      If ``in_frame_basis==True``, calls to ``solve``, this objects behaves as if all
      operators were supplied in the frame basis: calls to ``solve`` will assume the initial
      state is supplied in the frame basis, and the results will be returned in the frame basis.
      If ``in_frame_basis==False``, the system will still be solved in the frame basis for
      efficiency, however the initial state (and final output states) will automatically be
      transformed into (and, respectively, out of) the frame basis.
    * ``rwa_cutoff_freq``: Performs a rotating wave approximation (RWA) on the model
      with cutoff frequency ``rwa_cutoff_freq``. See
      :func:`~qiskit_dynamics.models.rotating_wave_approximation`
      for details.

      .. note::
            When using the ``rwa_cutoff_freq`` optional argument,
            :class:`~qiskit_dynamics.solvers.solver_classes.Solver` cannot be instantiated within
            a JAX-transformable function. However, after construction, instances can
            still be used within JAX-transformable functions regardless of whether an
            ``rwa_cutoff_freq`` is set.

    .. note::
        Modifications to the underlying model after instantiation may be made
        directly via the ``model`` property of this class. However,
        the getting and setting of model signals should be done via the ``signals`` property
        of this class, which manages signal transformations required in
        the case that a rotating wave approximation is made.

    The evolution given by the model can be simulated by calling
    :meth:`~qiskit_dynamics.solvers.Solver.solve`, which calls
    calls :func:`~qiskit_dynamics.solve.solve_lmde`, and does various automatic
    type handling operations for :mod:`qiskit.quantum_info` state and super operator types.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[Array] = None,
        hamiltonian_operators: Optional[Array] = None,
        hamiltonian_signals: Optional[Union[List[Signal], SignalList]] = None,
        static_dissipators: Optional[Array] = None,
        dissipator_operators: Optional[Array] = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        rwa_carrier_freqs: Optional[Union[Array, Tuple[Array, Array]]] = None,
        validate: bool = True,
    ):
        """Initialize solver with model information.

        Args:
            static_hamiltonian: Constant Hamiltonian term. If a ``rotating_frame``
                                is specified, the ``frame_operator`` will be subtracted from
                                the static_hamiltonian.
            hamiltonian_operators: Hamiltonian operators.
            hamiltonian_signals: (Deprecated) Coefficients for the Hamiltonian operators.
                                 This argument has been deprecated, signals should be passed
                                 to the solve method.
            static_dissipators: Constant dissipation operators.
            dissipator_operators: Dissipation operators with time-dependent coefficients.
            dissipator_signals: (Deprecated) Optional time-dependent coefficients for the
                                dissipators. If ``None``, coefficients are assumed to be the
                                constant ``1.``. This argument has been deprecated, signals
                                should be passed to the solve method.
            rotating_frame: Rotating frame to transform the model into. Rotating frames which
                            are diagonal can be supplied as a 1d array of the diagonal elements,
                            to explicitly indicate that they are diagonal.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                            frame operator is diagonalized. See class documentation for a more
                            detailed explanation on how this argument affects object behaviour.
            evaluation_mode: Method for model evaluation. See documentation for
                             ``HamiltonianModel.evaluation_mode`` or
                             ``LindbladModel.evaluation_mode``.
                             (if dissipators in model) for valid modes.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency. If ``None``, no
                             approximation is made.
            rwa_carrier_freqs: Carrier frequencies to use for rotating wave approximation.
            validate: Whether or not to validate Hamiltonian operators as being Hermitian.
        """

        if hamiltonian_signals or dissipator_signals:
            warnings.warn(
                """hamiltonian_signals and dissipator_signals are deprecated arguments
                and will be removed in a subsequent release.
                Signals should be passed directly to the solve method.""",
                DeprecationWarning,
                stacklevel=2
            )

        model = None
        if static_dissipators is None and dissipator_operators is None:
            model = HamiltonianModel(
                static_operator=static_hamiltonian,
                operators=hamiltonian_operators,
                signals=hamiltonian_signals,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )
            self._signals = hamiltonian_signals
        else:
            model = LindbladModel(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                hamiltonian_signals=hamiltonian_signals,
                static_dissipators=static_dissipators,
                dissipator_operators=dissipator_operators,
                dissipator_signals=dissipator_signals,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )
            self._signals = (hamiltonian_signals, dissipator_signals)

        ###############################################################################################
        # fix this to use new argument!
        ##################################################################################################

        self._rwa_signal_map = None
        if rwa_cutoff_freq is not None:

            original_signals = model.signals

            if rwa_carrier_freqs:
                #######################################################################################
                # Coudl validate here based on model type as well
                #######################################################################################

                if isinstance(rwa_carrier_freqs, tuple):
                    rwa_ham_sigs = None
                    rwa_lindblad_sigs = None
                    if rwa_carrier_freqs[0]:
                        rwa_ham_sigs = [Signal(1., carrier_freq=freq) for freq in rwa_carrier_freqs[0]]
                    if rwa_carrier_freqs[1]:
                        rwa_lindblad_sigs = [Signal(1., carrier_freq=freq) for freq in rwa_carrier_freqs[1]]

                    model.signals = (rwa_ham_sigs, rwa_lindblad_sigs)

                else:
                    model.signals = [Signal(1., carrier_freq=freq) for freq in rwa_carrier_freqs]

            model, rwa_signal_map = rotating_wave_approximation(
                model, rwa_cutoff_freq, return_signal_map=True
            )
            self._rwa_signal_map = rwa_signal_map

            if hamiltonian_signals or dissipator_signals:
                model.signals = self._rwa_signal_map(original_signals)

        self._model = model

    @property
    def model(self) -> Union[HamiltonianModel, LindbladModel]:
        """The model of the system, either a Hamiltonian or Lindblad model."""
        return self._model

    @property
    def signals(self) -> SignalList:
        """The signals used in the solver."""
        warnings.warn(
            """The signals property is deprecated.
            Signals should be passed directly to the solve method.""",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._signals

    @signals.setter
    def signals(
        self, new_signals: Union[List[Signal], SignalList, Tuple[List[Signal]], Tuple[SignalList]]
    ):
        """Set signals for the solver, and pass to the model."""
        warnings.warn(
            """The signals property is deprecated.
            Signals should be passed directly to the solve method.""",
            DeprecationWarning,
            stacklevel=2,
        )

        self._signals = new_signals
        if self._rwa_signal_map is not None:
            new_signals = self._rwa_signal_map(new_signals)
        self.model.signals = new_signals

    def copy(self) -> "Solver":
        """Return a copy of self."""
        ##############################################################################################
        # Deprecate this?
        # Not really necessary with pure version of solve
        ##############################################################################################

        return copy(self)

    def solve(
        self,
        signals: Union[List[Signal], Tuple[List[Signal], List[Signal]]],
        t_span: Union[Array, List[Array]],
        y0: Union[Array, QuantumState, BaseOperator],
        wrap_results: Optional[bool] = True,
        **kwargs
    ) -> OdeResult:
        ##############################################################################################
        # Update this doc
        # should signals be a single argument combining hamiltonian/dissipator signals, or should
        # we break these into two separate kwargs?
        ##############################################################################################

        r"""Solve the dynamical problem.

        Calls :func:`~qiskit_dynamics.solvers.solve_lmde`, and returns an `OdeResult`
        object in the style of `scipy.integrate.solve_ivp`, with results
        formatted to be the same types as the input. See Additional Information
        for special handling of various input types.

        Args:
            signals: Specification of time-dependent coefficients to simulate.
            t_span: Time interval to integrate over.
            y0: Initial state.
            wrap_results: Whether or not to wrap the result arrays in the same class as y0.
            control_flow: Whether to use standard python or other loops.
            **kwargs: Keyword args passed to :func:`~qiskit_dynamics.solvers.solve_lmde`.

        Returns:
            OdeResult: object with formatted output types.

        Raises:
            QiskitError: Initial state ``y0`` is of invalid shape.

        Additional Information:

            The behaviour of this method is impacted by the input type of ``y0``:

             * If ``y0`` is an ``Array``, it is passed directly to
                :func:`~qiskit_dynamics.solve_lmde` as is. Acceptable array shapes are
                determined by the model type and evaluation mode.
             * If ``y0`` is a subclass of :class:`qiskit.quantum_info.QuantumState`:

                 * If ``self.model`` is a :class:`~qiskit_dynamics.models.LindbladModel`,
                    ``y0`` is converted to a :class:`DensityMatrix`. Further, if the model
                    evaluation mode is vectorized ``y0`` will be suitably reshaped for solving.
                 * If ``self.model`` is a :class:`~qiskit_dynamics.models.HamiltonianModel`,
                    and ``y0`` a :class:`DensityMatrix`, the full unitary will be simulated,
                    and the evolution of ``y0`` is attained via conjugation.

             * If ``y0`` is a subclass of :class`qiskit.quantum_info.QuantumChannel`, the full
                evolution map will be computed and composed with ``y0``; either the unitary if
                ``self.model`` is a :class:`~qiskit_dynamics.models.HamiltonianModel`, or the full
                Lindbladian ``SuperOp`` if the model is a
                :class:`~qiskit_dynamics.models.LindbladModel`.

        """


        signals_list, t_span_list, y0_list, multiple_sims = setup_simulation_lists(signals, t_span, y0)

        # hold copy of signals in model for deprecated behavior
        original_signals = self.model.signals

        all_results = []
        ##################################################################################################
        # for now assume Hamiltonian, can handle lindblad properly later
        #################################################################################################
        for signals, t_span, y0 in zip(signals_list, t_span_list, y0_list):

            # convert types
            if isinstance(y0, QuantumState) and isinstance(self.model, LindbladModel):
                y0 = DensityMatrix(y0)

            y0, y0_cls, state_type_wrapper = initial_state_converter(y0)

            # validate types
            if (y0_cls is SuperOp) and is_lindblad_model_not_vectorized(self.model):
                raise QiskitError(
                    """Simulating SuperOp for a LindbladModel requires setting
                    vectorized evaluation. Set LindbladModel.evaluation_mode to a vectorized option.
                    """
                )

            # modify initial state for some custom handling of certain scenarios
            y_input = y0

            # if Simulating density matrix or SuperOp with a HamiltonianModel, simulate the unitary
            if y0_cls in [DensityMatrix, SuperOp] and isinstance(self.model, HamiltonianModel):
                y0 = np.eye(self.model.dim, dtype=complex)
            # if LindbladModel is vectorized and simulating a density matrix, flatten
            elif (
                (y0_cls is DensityMatrix)
                and isinstance(self.model, LindbladModel)
                and "vectorized" in self.model.evaluation_mode
            ):
                y0 = y0.flatten(order="F")

            # validate y0 shape before passing to solve_lmde
            if isinstance(self.model, HamiltonianModel) and (
                y0.shape[0] != self.model.dim or y0.ndim > 2
            ):
                raise QiskitError("""Shape mismatch for initial state y0 and HamiltonianModel.""")
            if is_lindblad_model_vectorized(self.model) and (
                y0.shape[0] != self.model.dim**2 or y0.ndim > 2
            ):
                raise QiskitError(
                    """Shape mismatch for initial state y0 and LindbladModel
                                     in vectorized evaluation mode."""
                )
            if is_lindblad_model_not_vectorized(self.model) and y0.shape[-2:] != (
                self.model.dim,
                self.model.dim,
            ):
                raise QiskitError("""Shape mismatch for initial state y0 and LindbladModel.""")

            if signals is not None:
                if self._rwa_signal_map:
                    signals = self._rwa_signal_map(signals)
                self.model.signals = signals

            results = solve_lmde(generator=self.model, t_span=t_span, y0=y0, **kwargs)
            results.y = format_final_states(results.y, self.model, y_input, y0_cls)

            if y0_cls is not None and wrap_results:
                results.y = [state_type_wrapper(yi) for yi in results.y]

            all_results.append(results)


        # replace copy of original signals for deprecated behavior
        self.model.signals = original_signals

        # do we want to do this?
        ##############################################################################################
        if multiple_sims is False:
            return all_results[0]

        return all_results


def initial_state_converter(obj: Any) -> Tuple[Array, Type, Callable]:
    """Convert initial state object to an Array, the type of the initial input, and return
    function for constructing a state of the same type.

    Args:
        obj: An initial state.

    Returns:
        tuple: (Array, Type, Callable)
    """
    # pylint: disable=invalid-name
    y0_cls = None
    if isinstance(obj, Array):
        y0, y0_cls, wrapper = obj, None, lambda x: x
    if isinstance(obj, QuantumState):
        y0, y0_cls = Array(obj.data), obj.__class__
        wrapper = lambda x: y0_cls(np.array(x), dims=obj.dims())
    elif isinstance(obj, QuantumChannel):
        y0, y0_cls = Array(SuperOp(obj).data), SuperOp
        wrapper = lambda x: SuperOp(
            np.array(x), input_dims=obj.input_dims(), output_dims=obj.output_dims()
        )
    elif isinstance(obj, (BaseOperator, Gate, QuantumCircuit)):
        y0, y0_cls = Array(Operator(obj.data)), Operator
        wrapper = lambda x: Operator(
            np.array(x), input_dims=obj.input_dims(), output_dims=obj.output_dims()
        )
    else:
        y0, y0_cls, wrapper = Array(obj), None, lambda x: x

    return y0, y0_cls, wrapper


def format_final_states(y, model, y_input, y0_cls):
    """Format final states for a single simulation."""

    y = Array(y)

    new_y = None

    if y0_cls is DensityMatrix and isinstance(model, HamiltonianModel):
        # conjugate by unitary
        new_y = y @ y_input @ y.conj().transpose((0, 2, 1))
    elif y0_cls is SuperOp and isinstance(model, HamiltonianModel):
        # convert to SuperOp and compose
        new_y = (
            np.einsum("nka,nlb->nklab", y.conj(), y).reshape(
                y.shape[0], y.shape[1] ** 2, y.shape[1] ** 2
            )
            @ y_input
        )
    elif (y0_cls is DensityMatrix) and is_lindblad_model_vectorized(model):
        new_y = y.reshape((len(y),) + y_input.shape, order="F")
    else:
        new_y = y

    return new_y


def setup_simulation_lists(signals, t_span, y0):
    """Setup args to do a list of simulations.

    This can probably be done more generally, involving some generic singleton type
    checking/variable name arguments.

    ***************************************************************************************************
    fill this out
    """

    multiple_sims = False

    if signals is None:
        # for deprecated behavior
        signals = [signals]
    elif isinstance(signals, tuple):
        # single Lindblad simulation
        signals = [signals]
    elif isinstance(signals, list) and isinstance(signals[0], tuple):
        # multiple lindblad simulation
        multiple_sims = True
    elif isinstance(signals, list) and isinstance(signals[0], (list, SignalList)):
        # multiple Hamiltonian simulation
        multiple_sims = True
    elif isinstance(signals, SignalList) or (isinstance(signals, list) and isinstance(signals[0], Signal)):
        # single round of Hamiltonian simulation
        signals = [signals]
    else:
        raise QiskitError("Signals specified in invalid format.")

    if not isinstance(y0, list):
        y0 = [y0]
    else:
        multiple_sims = True


    t_span = Array(t_span)

    # setup t_span to have the same "shape" as signals
    if t_span.ndim > 2:
        raise QiskitError("t_span must be either 1d or 2d.")
    elif t_span.ndim == 1:
        t_span = [t_span]
    else:
        multiple_sims = True

    # consolidate lengths and raise error if incompatible
    args = [signals, t_span, y0]
    arg_lens = [len(x) for x in args]
    max_len = max(arg_lens)
    if any(x != 1 and x != max_len for x in arg_lens):
        raise QiskitError("signals, y0, and t_span specify an incompatible number of simulations.")

    args = [arg * max_len if arg_len == 1 else arg for arg, arg_len in zip(args, arg_lens)]

    return args[0], args[1], args[2], multiple_sims
