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

from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import

from qiskit import QiskitError
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator, DensityMatrix

from qiskit_dynamics.models import (
    HamiltonianModel,
    LindbladModel,
    RotatingFrame,
    rotating_wave_approximation,
)
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.array import Array
from qiskit_dynamics.dispatch.dispatch import Dispatch

from .solver_functions import solve_lmde
from .solver_utils import is_lindblad_model_vectorized, is_lindblad_model_not_vectorized


class Solver:
    r"""Solver class for simulating both Hamiltonian and Lindblad dynamics, with high
    level type-handling of input states.

    If only Hamiltonian information is provided, this class will internally construct
    a :class:`~qiskit_dynamics.models.HamiltonianModel` instance, and simulate the model
    using the Schrodinger equation :math:`\dot{y}(t) = -iH(t)y(t)`
    (see the :meth:`~qiskit_dynamics.solvers.Solver.solve` method documentation for details
    on how different initial state types are handled).
    :class:`~qiskit_dynamics.models.HamiltonianModel` represents a
    decomposition of the Hamiltonian of the form:

    .. math::

        H(t) = H_0 + \sum_i s_i(t) H_i,

    where :math:`H_0` is the static component, the :math:`H_i` are the operator part
    of the time-dependent part of the Hamiltonian, and the :math:`s_i(t)` are the
    time-dependent signals, specifiable as either :class:`~qiskit_dynamics.signals.Signal`
    objects, or constructed from Qiskit Pulse schedules if :class:`~qiskit_dynamics.solvers.Solver`
    is configured for Pulse simulation (see below).

    If dissipators are specified as part of the model, then a
    :class:`~qiskit_dynamics.models.LindbladModel` is constructed, and simulations are performed
    by solving the Lindblad equation:

    .. math::

        \dot{y}(t) = -i[H(t), y(t)] + \mathcal{D}_0(y(t)) + \mathcal{D}(t)(y(t)),

    where :math:`H(t)` is the Hamiltonian part, specified as above, and :math:`\mathcal{D}_0`
    and :math:`\mathcal{D}(t)` are the static and time-dependent portions of the dissipator,
    given by:

    .. math::

        \mathcal{D}_0(y(t)) = \sum_j N_j y(t) N_j^\dagger
                                      - \frac{1}{2} \{N_j^\dagger N_j, y(t)\},

    and

    .. math::

        \mathcal{D}(t)(y(t)) = \sum_j \gamma_j(t) L_j y(t) L_j^\dagger
                                  - \frac{1}{2} \{L_j^\dagger L_j, y(t)\},

    with :math:`N_j` the static dissipators, :math:`L_j` the time-dependent dissipator
    operators, and :math:`\gamma_j(t)` the time-dependent signals
    specifiable as either :class:`~qiskit_dynamics.signals.Signal`
    objects, or constructed from Qiskit Pulse schedules if :class:`~qiskit_dynamics.solvers.Solver`
    is configured for Pulse simulation (see below).

    Transformations on the model can be specified via the optional arguments:

    * ``rotating_frame``: Transforms the model into a rotating frame. Note that the
      operator specifying the frame will be substracted from the ``static_hamiltonian``.
      If supplied as a 1d array, ``rotating_frame`` is interpreted as the diagonal
      elements of a diagonal matrix. See :class:`~qiskit_dynamics.models.RotatingFrame` for details.
    * ``in_frame_basis``: Whether to represent the model in the basis in which the frame
      operator is diagonal, henceforth called the "frame basis".
      If ``rotating_frame`` is ``None`` or was supplied as a 1d array,
      this kwarg has no effect. If ``rotating_frame`` was specified as a 2d array,
      the frame basis is the diagonalizing basis supplied by ``np.linalg.eigh``.
      If ``in_frame_basis==True``, this objects behaves as if all
      operators were supplied in the frame basis: calls to ``solve`` will assume the initial
      state is supplied in the frame basis, and the results will be returned in the frame basis.
      If ``in_frame_basis==False``, the system will still be solved in the frame basis for
      efficiency, however the initial state (and final output states) will automatically be
      transformed into (and, respectively, out of) the frame basis.
    * ``rwa_cutoff_freq`` and ``rwa_carrier_freqs``: Performs a rotating wave approximation (RWA)
      on the model with cutoff frequency ``rwa_cutoff_freq``, assuming the time-dependent
      coefficients of the model have carrier frequencies specified by ``rwa_carrier_freqs``.
      If ``dissipator_operators is None``, ``rwa_carrier_freqs`` must be a list of floats
      of length equal to ``hamiltonian_operators``, and if ``dissipator_operators is not None``,
      ``rwa_carrier_freqs`` must be a ``tuple`` of lists of floats, with the first entry
      the list of carrier frequencies for ``hamiltonian_operators``, and the second
      entry the list of carrier frequencies for ``dissipator_operators``.
      See :func:`~qiskit_dynamics.models.rotating_wave_approximation` for details on
      the mathematical approximation.

      .. note::
            When using the ``rwa_cutoff_freq`` optional argument,
            :class:`~qiskit_dynamics.solvers.solver_classes.Solver` cannot be instantiated within
            a JAX-transformable function. However, after construction, instances can
            still be used within JAX-transformable functions regardless of whether an
            ``rwa_cutoff_freq`` is set.

    :class:`~qiskit_dynamics.solvers.Solver` can be configured to simulate Qiskit Pulse schedules
    by setting all of the following parameters, which determine how Pulse schedules are
    interpreted:

    * ``hamiltonian_channels``: List of channels in string format corresponding to the
      time-dependent coefficients of ``hamiltonian_operators``.
    * ``dissipator_channels``: List of channels in string format corresponding to time-dependent
      coefficients of ``dissipator_operators``.
    * ``channel_carrier_freqs``: Dictionary mapping channel names to frequencies. A frequency
      must be specified for every channel appearing in ``hamiltonian_channels`` and
      ``dissipator_channels``. When simulating ``schedule``\s, these frequencies are
      interpreted as the analog carrier frequencies associated with the channel; deviations from
      these frequencies due to ``SetFrequency`` or ``ShiftFrequency`` instructions are
      implemented by digitally modulating the samples for the channel envelope.
      If an ``rwa_cutoff_freq`` is specified, and no ``rwa_carrier_freqs`` is specified, these
      frequencies will be used for the RWA.
    * ``dt``: The envelope sample width.

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
        hamiltonian_channels: Optional[List[str]] = None,
        dissipator_channels: Optional[List[str]] = None,
        channel_carrier_freqs: Optional[dict] = None,
        dt: Optional[float] = None,
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
            hamiltonian_channels: List of channel names corresponding to Hamiltonian operators.
            dissipator_channels: List of channel names corresponding to dissipator operators.
            channel_carrier_freqs: Dictionary mapping channel names to floats.
            dt: Sample rate for simulating pulse schedules.
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
                               If no time dependent coefficients in model leave as ``None``,
                               if no time-dependent dissipators specify as a list of frequencies
                               for each Hamiltonian operator, and if time-dependent dissipators
                               present specify as a tuple of lists of frequencies, one for
                               Hamiltonian operators and one for dissipators.
            validate: Whether or not to validate Hamiltonian operators as being Hermitian.

        Raises:
            QiskitError: If arguments concerning pulse-schedule interpretation are insufficiently
            specified.
        """

        # set pulse specific information if specified
        self._hamiltonian_channels = None
        self._dissipator_channels = None
        self._all_channels = None
        self._channel_carrier_freqs = None
        self._dt = None
        self._schedule_converter = None

        if any([dt, channel_carrier_freqs, hamiltonian_channels, dissipator_channels]):
            all_channels = []

            if hamiltonian_channels is not None:
                hamiltonian_channels = [chan.lower() for chan in hamiltonian_channels]
                for chan in hamiltonian_channels:
                    if chan not in all_channels:
                        all_channels.append(chan)
                if hamiltonian_operators is None or len(hamiltonian_operators) != len(
                    hamiltonian_channels
                ):
                    raise QiskitError(
                        """hamiltonian_channels must have same length as hamiltonian_operators"""
                    )

            self._hamiltonian_channels = hamiltonian_channels

            if dissipator_channels is not None:
                dissipator_channels = [chan.lower() for chan in dissipator_channels]
                for chan in dissipator_channels:
                    if chan not in all_channels:
                        all_channels.append(chan)
                if dissipator_operators is None or len(dissipator_operators) != len(
                    dissipator_channels
                ):
                    raise QiskitError(
                        """dissipator_channels must have same length as dissipator_operators"""
                    )

            self._dissipator_channels = dissipator_channels
            self._all_channels = all_channels

            if channel_carrier_freqs is None:
                channel_carrier_freqs = {}
            else:
                channel_carrier_freqs = {
                    key.lower(): val for key, val in channel_carrier_freqs.items()
                }

            for chan in all_channels:
                if chan not in channel_carrier_freqs:
                    raise QiskitError(
                        f"""Channel '{chan}' does not have carrier frequency specified in
                        channel_carrier_freqs."""
                    )

            if len(channel_carrier_freqs) == 0:
                channel_carrier_freqs = None
            self._channel_carrier_freqs = channel_carrier_freqs

            if dt is not None:
                self._dt = dt

                self._schedule_converter = InstructionToSignals(
                    dt=self._dt, carriers=self._channel_carrier_freqs, channels=self._all_channels
                )
            else:
                raise QiskitError("dt must be specified if channel information provided.")

        if hamiltonian_signals or dissipator_signals:
            warnings.warn(
                """hamiltonian_signals and dissipator_signals are deprecated arguments
                and will be removed in a subsequent release.
                Signals should be passed directly to the solve method.""",
                DeprecationWarning,
                stacklevel=2,
            )

        # setup model
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

        self._rwa_signal_map = None
        if rwa_cutoff_freq:

            original_signals = model.signals
            # if configured in pulse mode and rwa_carrier_freqs is None, use the channel
            # carrier freqs
            if rwa_carrier_freqs is None and self._channel_carrier_freqs is not None:
                rwa_carrier_freqs = None
                if self._hamiltonian_channels is not None:
                    rwa_carrier_freqs = [
                        self._channel_carrier_freqs[c] for c in self._hamiltonian_channels
                    ]

                if self._dissipator_channels is not None:
                    rwa_carrier_freqs = (
                        rwa_carrier_freqs,
                        [self._channel_carrier_freqs[c] for c in self._dissipator_channels],
                    )

            if rwa_carrier_freqs is not None:
                if isinstance(rwa_carrier_freqs, tuple):
                    rwa_ham_sigs = None
                    rwa_lindblad_sigs = None
                    if rwa_carrier_freqs[0]:
                        rwa_ham_sigs = [
                            Signal(1.0, carrier_freq=freq) for freq in rwa_carrier_freqs[0]
                        ]
                    if rwa_carrier_freqs[1]:
                        rwa_lindblad_sigs = [
                            Signal(1.0, carrier_freq=freq) for freq in rwa_carrier_freqs[1]
                        ]

                    model.signals = (rwa_ham_sigs, rwa_lindblad_sigs)
                else:
                    rwa_sigs = [Signal(1.0, carrier_freq=freq) for freq in rwa_carrier_freqs]

                    if isinstance(model, LindbladModel):
                        rwa_sigs = (rwa_sigs, None)

                    model.signals = rwa_sigs

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
        """(Deprecated) The signals used in the solver."""
        warnings.warn(
            """The signals property is deprecated and will be removed in the next release.
            Signals should be passed directly to the solve method.""",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._signals

    @signals.setter
    def signals(
        self, new_signals: Union[List[Signal], SignalList, Tuple[List[Signal]], Tuple[SignalList]]
    ):
        """(Deprecated) Set signals for the solver, and pass to the model."""
        warnings.warn(
            """The signals property is deprecated and will be removed in the next release.
            Signals should be passed directly to the solve method.""",
            DeprecationWarning,
            stacklevel=2,
        )

        self._signals = new_signals
        if self._rwa_signal_map is not None:
            new_signals = self._rwa_signal_map(new_signals)
        self.model.signals = new_signals

    def copy(self) -> "Solver":
        """(Deprecated) Return a copy of self."""
        warnings.warn(
            """The copy method is deprecated and will be removed in the next release.
            This deprecation is associated with the deprecation of the signals property;
            the copy method will no longer be needed once the signal property is removed.""",
            DeprecationWarning,
            stacklevel=2,
        )

        return copy(self)

    def solve(
        self,
        t_span: Array,
        y0: Union[Array, QuantumState, BaseOperator],
        signals: Optional[
            Union[List[Schedule], List[Signal], Tuple[List[Signal], List[Signal]]]
        ] = None,
        convert_results: bool = True,
        **kwargs,
    ) -> Union[OdeResult, List[OdeResult]]:
        r"""Solve a dynamical problem, or a set of dynamical problems.

        Calls :func:`~qiskit_dynamics.solvers.solve_lmde`, and returns an ``OdeResult``
        object in the style of ``scipy.integrate.solve_ivp``, with results
        formatted to be the same types as the input. See Additional Information
        for special handling of various input types, and for specifying multiple
        simulations at once.

        Args:
            t_span: Time interval to integrate over.
            y0: Initial state.
            signals: Specification of time-dependent coefficients to simulate, either in
                     Signal format or as Qiskit Pulse Pulse schedules.
                     If specifying in Signal format, if ``dissipator_operators is None``,
                     specify as a list of signals for the Hamiltonian component, otherwise
                     specify as a tuple of two lists, one for Hamiltonian components, and
                     one for the ``dissipator_operators`` coefficients.
            convert_results: If ``True``, convert returned solver state results to the same class
                             as y0. If ``False``, states will be returned in the native array type
                             used by the specified solver method.
            **kwargs: Keyword args passed to :func:`~qiskit_dynamics.solvers.solve_lmde`.

        Returns:
            OdeResult: object with formatted output types.

        Raises:
            QiskitError: Initial state ``y0`` is of invalid shape. If ``signals`` specifies
                         ``Schedule`` simulation but ``Solver`` hasn't been configured to
                         simulate pulse schedules.

        Additional Information:

        The behaviour of this method is impacted by the input type of ``y0``
        and the internal model, summarized in the following table:

        .. list-table:: Type-based behaviour
           :widths: 10 10 10 70
           :header-rows: 1

           * - ``y0`` type
             - Model type
             - ``yf`` type
             - Description
           * - ``Array``, ``np.ndarray``, ``Operator``
             - Any
             - Same as ``y0``
             - Solves either the Schrodinger equation or Lindblad equation
               with initial state ``y0`` as specified.
           * - ``Statevector``
             - ``HamiltonianModel``
             - ``Statevector``
             - Solves the Schrodinger equation with initial state ``y0``.
           * - ``DensityMatrix``
             - ``HamiltonianModel``
             - ``DensityMatrix``
             - Solves the Schrodinger equation with initial state the identity matrix to compute
               the unitary, then conjugates ``y0`` with the result to solve for the density matrix.
           * - ``Statevector``, ``DensityMatrix``
             - ``LindbladModel``
             - ``DensityMatrix``
             - Solve the Lindblad equation with initial state ``y0``, converting to a
               ``DensityMatrix`` first if ``y0`` is a ``Statevector``.
           * - ``QuantumChannel``
             - ``HamiltonianModel``
             - ``SuperOp``
             - Converts ``y0`` to a ``SuperOp`` representation, then solves the Schrodinger
               equation with initial state the identity matrix to compute the unitary and
               composes with ``y0``.
           * - ``QuantumChannel``
             - ``LindbladModel``
             - ``SuperOp``
             - Solves the vectorized Lindblad equation with initial state ``y0``.
               ``evaluation_mode`` must be set to a vectorized option.

        In some cases (e.g. if using JAX), wrapping the returned states in the type
        given in the ``yf`` type column above may be undesirable. Setting
        ``convert_results=False`` prevents this wrapping, while still allowing
        usage of the automatic type-handling for the input state.

        In addition to the above, this method can be used to specify multiple simulations
        simultaneously. This can be done by specifying one or more of the arguments
        ``t_span``, ``y0``, or ``signals`` as a list of valid inputs.
        For this mode of operation, all of these arguments must be either lists of the same
        length, or a single valid input, which will be used repeatedly.

        For example the following code runs three simulations, returning results in a list:

        .. code-block:: python

            t_span = [span1, span2, span3]
            y0 = [state1, state2, state3]
            signals = [signals1, signals2, signals3]

            results = solver.solve(t_span=t_span, y0=y0, signals=signals)

        The following code block runs three simulations, for different sets of signals,
        repeatedly using the same ``t_span`` and ``y0``:

        .. code-block:: python

            t_span = [t0, tf]
            y0 = state1
            signals = [signals1, signals2, signal3]

            results = solver.solve(t_span=t_span, y0=y0, signals=signals)
        """
        # hold copy of signals in model for deprecated behavior
        original_signals = self.model.signals

        # raise deprecation warning if signals is None and non-trivial signals to fall back on
        if signals is None and not original_signals in (None, (None, None)):
            warnings.warn(
                """No signals specified to solve, falling back on signals stored in model.
                Passing signals to Solver at instantiation and setting Solver.signals have been
                deprecated and will be removed in the next release. Instead pass signals
                directly to the solve method.""",
                DeprecationWarning,
                stacklevel=2,
            )

        t_span_list, y0_list, signals_list, multiple_sims = setup_simulation_lists(
            t_span, y0, signals
        )

        # if simulating schedules, convert to Signal objects at this point
        if isinstance(signals_list[0], (Schedule, ScheduleBlock)):
            if self._schedule_converter is None:
                raise QiskitError("Solver instance not configured for pulse Schedule simulation.")

            new_signals_list = []
            for sched in signals_list:
                if isinstance(sched, ScheduleBlock):
                    sched = block_to_schedule(sched)

                all_signals = self._schedule_converter.get_signals(sched)
                if isinstance(self.model, HamiltonianModel):
                    if self._hamiltonian_channels is not None:
                        new_signals_list.append(
                            [
                                all_signals[self._all_channels.index(chan)]
                                for chan in self._hamiltonian_channels
                            ]
                        )
                    else:
                        new_signals_list.append(None)
                else:
                    hamiltonian_signals = None
                    dissipator_signals = None
                    if self._hamiltonian_channels is not None:
                        hamiltonian_signals = [
                            all_signals[self._all_channels.index(chan)]
                            for chan in self._hamiltonian_channels
                        ]
                    if self._dissipator_channels is not None:
                        dissipator_signals = [
                            all_signals[self._all_channels.index(chan)]
                            for chan in self._dissipator_channels
                        ]
                    new_signals_list.append((hamiltonian_signals, dissipator_signals))

            signals_list = new_signals_list

        # run simulations
        all_results = [
            self._solve(
                t_span=current_t_span,
                y0=current_y0,
                signals=current_signals,
                convert_results=convert_results,
                **kwargs,
            )
            for current_t_span, current_y0, current_signals in zip(
                t_span_list, y0_list, signals_list
            )
        ]

        # replace copy of original signals for deprecated behavior
        self.model.signals = original_signals

        if multiple_sims is False:
            return all_results[0]

        return all_results

    def _solve(
        self,
        t_span: Array,
        y0: Union[Array, QuantumState, BaseOperator],
        signals: Optional[Union[List[Signal], Tuple[List[Signal], List[Signal]]]] = None,
        convert_results: Optional[bool] = True,
        **kwargs,
    ) -> OdeResult:
        """Helper function solve for running a single simulation."""
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
            # if Lindblad model and signals are given as a list
            # set as just the Hamiltonian part of the signals
            if isinstance(self.model, LindbladModel) and isinstance(signals, (list, SignalList)):
                signals = (signals, None)

            if self._rwa_signal_map:
                signals = self._rwa_signal_map(signals)
            self.model.signals = signals

        results = solve_lmde(generator=self.model, t_span=t_span, y0=y0, **kwargs)
        results.y = format_final_states(results.y, self.model, y_input, y0_cls)

        if y0_cls is not None and convert_results:
            results.y = [state_type_wrapper(yi) for yi in results.y]

        return results


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

    if y0_cls is DensityMatrix and isinstance(model, HamiltonianModel):
        # conjugate by unitary
        return y @ y_input @ y.conj().transpose((0, 2, 1))
    elif y0_cls is SuperOp and isinstance(model, HamiltonianModel):
        # convert to SuperOp and compose
        return (
            np.einsum("nka,nlb->nklab", y.conj(), y).reshape(
                y.shape[0], y.shape[1] ** 2, y.shape[1] ** 2
            )
            @ y_input
        )
    elif (y0_cls is DensityMatrix) and is_lindblad_model_vectorized(model):
        return y.reshape((len(y),) + y_input.shape, order="F")

    return y


def setup_simulation_lists(
    t_span: Array,
    y0: Union[Array, QuantumState, BaseOperator],
    signals: Optional[Union[List[Schedule], List[Signal], Tuple[List[Signal], List[Signal]]]],
) -> Tuple[List, List, List, bool]:
    """Helper function for setting up lists of simulations.

    Transform input signals, t_span, and y0 into lists of the same length.
    Arguments are given as either lists of valid specifications, or as a singleton of a valid
    specification. Singletons are transformed into a list of length one, then all arguments
    are expanded to be the same length as the longest argument max_len:
        - If len(arg) == 1, it will be repeated max_len times
        - if len(arg) == max_len, nothing is done
        - If len(arg) not in (1, max_len), an error is raised

    Args:
        t_span: Time interval specification.
        y0: Initial state specification.
        signals: Signal specification.

    Returns:
        Tuple: tuple of lists of arguments of the same length, along with a bool specifying whether
        the arguments specified multiple simulations or not.

    Raises:
        QiskitError: If the length of any arguments are incompatible, or if any singleton
        is an invalid shape.
    """

    multiple_sims = False

    if signals is None:
        signals = [signals]
    elif isinstance(signals, tuple):
        # single Lindblad
        signals = [signals]
    elif isinstance(signals, list) and isinstance(signals[0], tuple):
        # multiple lindblad
        multiple_sims = True
    elif isinstance(signals, (Schedule, ScheduleBlock)):
        # pulse simulation
        signals = [signals]
    elif isinstance(signals, list) and isinstance(signals[0], (Schedule, ScheduleBlock)):
        # multiple pulse simulation
        multiple_sims = True
    elif isinstance(signals, list) and isinstance(signals[0], (list, SignalList)):
        # multiple Hamiltonian signals lists
        multiple_sims = True
    elif isinstance(signals, SignalList) or (
        isinstance(signals, list) and not isinstance(signals[0], (list, SignalList))
    ):
        # single Hamiltonian signals list
        signals = [signals]
    else:
        raise QiskitError("Signals specified in invalid format.")

    if not isinstance(y0, list):
        y0 = [y0]
    else:
        multiple_sims = True

    t_span_ndim = nested_ndim(t_span)

    if t_span_ndim > 2:
        raise QiskitError("t_span must be either 1d or 2d.")
    if t_span_ndim == 1:
        t_span = [t_span]
    else:
        multiple_sims = True

    # consolidate lengths and raise error if incompatible
    args = [t_span, y0, signals]
    arg_names = ["t_span", "y0", "signals"]
    arg_lens = [len(x) for x in args]
    max_len = max(arg_lens)
    for idx, arg_len in enumerate(arg_lens):
        if arg_len not in (1, max_len):
            max_name = arg_names[arg_lens.index(max_len)]
            raise QiskitError(
                f"""If one of signals, y0, and t_span is given as a list of valid inputs,
                then the others must specify only a single input, or a list of the same length.
                {max_name} specifies {max_len} inputs, but {arg_names[idx]} is of length {arg_len},
                which is incompatible."""
            )

    args = [arg * max_len if arg_len == 1 else arg for arg, arg_len in zip(args, arg_lens)]

    return args[0], args[1], args[2], multiple_sims


def nested_ndim(x):
    """Determine the 'ndim' of x, which could be composed of nested lists and array types."""
    if isinstance(x, (list, tuple)):
        return 1 + nested_ndim(x[0])
    elif issubclass(type(x), Dispatch.REGISTERED_TYPES) or isinstance(x, Array):
        return x.ndim

    # assume scalar
    return 0
