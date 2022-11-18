# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
Pulse-enabled simulator backend.
"""

import datetime
import uuid

from typing import List, Optional, Union
import copy
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import

from qiskit import pulse
from qiskit.qobj.utils import MeasLevel
from qiskit.qobj.common import QobjHeader
from qiskit.transpiler import Target
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.providers.options import Options
from qiskit.providers.backend import BackendV2
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData

from qiskit import QiskitError, QuantumCircuit
from qiskit import schedule as build_schedule
from qiskit.quantum_info import Statevector, DensityMatrix

from qiskit_dynamics.solvers.solver_classes import Solver

from .dynamics_job import DynamicsJob
from .pulse_simulator_utils import (
    _get_dressed_state_decomposition,
    _get_lab_frame_static_hamiltonian,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
)


class PulseSimulator(BackendV2):
    r"""Pulse enabled simulator backend.

    **Supported options**

    * ``shots``: Number of shots per experiment. Defaults to ``1024``.
    * ``solver``: The Qiskit Dynamics :class:`.Solver` instance used for simulation.
    * ``solver_options``: Dictionary containing optional kwargs for passing to :meth:`Solver.solve`,
      indicating solver methods and options. Defaults to the empty dictionary ``{}``.
    * ``subsystem_dims``: Dimensions of subsystems making up the system in ``solver``. Defaults
      to ``[solver.model.dim]``.
    * ``subsystem_labels``: Integer labels for subsystems. Defaults to
      ``[0, ..., len(subsystem_dims) - 1]``.
    * ``meas_map``: Measurement map. Defaults to ``[[idx] for idx in subsystem_labels]``.
    * ``initial_state``: Initial state for simulation, either the string ``"ground_state"``,
      indicating that the ground state for the system Hamiltonian should be used, or an arbitrary
      ``Statevector`` or ``DensityMatrix``. Defaults to ``"ground_state"``.
    * ``normalize_states``: Boolean indicating whether to normalize states before computing
      outcome probabilities. Defaults to ``True``. Setting to ``False`` can result in errors if
      the solution tolerance results in probabilities with significant numerical deviation from
      proper probability distributions.
    * ``meas_level``: Form of measurement return. Only supported value is ``2``, indicating that
      counts should be returned. Defaults to ``meas_level==2``.
    * ``max_outcome_level``: For ``meas_level==2``, the maximum outcome for each subsystem.
      Values will be rounded down to be no larger than ``max_outcome_level``. Must be a positive
      integer or ``None``. If ``None``, no rounding occurs. Defaults to ``1``.
    * ``memory``: Boolean indicating whether to return a list of explicit measurement outcomes for
      every experimental shot. Defaults to ``True``.
    * ``seed_simulator``: Seed to use in random sampling. Defaults to ``None``.
    * ``experiment_result_function``: Function for computing the ``ExperimentResult``
      for each simulated experiment. This option defaults to
      :func:`default_experiment_result_function`, and any other function set to this option
      must have the same signature. Note that the default utilizes various other options that
      control results computation, and hence changing it will impact the meaning of other options.
    """

    def __init__(
        self,
        solver: Solver,
        target: Optional[Target] = None,
        **options,
    ):
        """Instantiate with a :class:`.Solver` instance and additional options.

        Args:
            solver: Solver instance configured for pulse simulation.
            target: Target object.
            options: Additional configuration options for the simulator.

        Raises:
            QiskitError: If any instantiation arguments fail validation checks.
        """

        super().__init__(
            name="PulseSimulator",
            description="Pulse enabled simulator backend.",
            backend_version="0.1",
        )

        # Dressed states of solver, will be calculated when solver option is set
        self._dressed_evals = None
        self._dressed_states = None
        self._dressed_states_adjoint = None

        # add subsystem_dims to options so set_options validation works
        if "subsystem_dims" not in options:
            options["subsystem_dims"] = [solver.model.dim]

        # Set simulator options
        self.set_options(solver=solver, **options)

        if self.options.subsystem_labels is None:
            labels = list(range(len(self.options.subsystem_dims)))
            self.set_options(subsystem_labels=labels)

        if self.options.meas_map is None:
            meas_map = [[idx] for idx in self.options.subsystem_labels]
            self.set_options(meas_map=meas_map)

        # self._target = target or Target() doesn't work as bool(target) can be False
        if target is None:
            target = Target()
        else:
            target = copy.copy(target)

        # add default simulator measure instructions
        instruction_schedule_map = target.instruction_schedule_map()
        for qubit in self.options.subsystem_labels:
            if not instruction_schedule_map.has(instruction="measure", qubits=qubit):
                with pulse.build() as meas_sched:
                    pulse.acquire(
                        duration=1, qubit_or_channel=qubit, register=pulse.MemorySlot(qubit)
                    )

            instruction_schedule_map.add(instruction="measure", qubits=qubit, schedule=meas_sched)

        self._target = target

    def _default_options(self):
        return Options(
            shots=1024,
            solver=None,
            solver_options={},
            subsystem_dims=None,
            subsystem_labels=None,
            meas_map=None,
            normalize_states=True,
            initial_state="ground_state",
            meas_level=MeasLevel.CLASSIFIED,
            max_outcome_level=1,
            memory=True,
            seed_simulator=None,
            experiment_result_function=default_experiment_result_function,
        )

    def set_options(self, **fields):
        """Set options for PulseSimulator."""

        validate_subsystem_dims = False

        for key, value in fields.items():
            if not hasattr(self._options, key):
                raise AttributeError(f"Invalid option {key}")

            if key == "initial_state":
                if value != "ground_state" and not isinstance(value, (Statevector, DensityMatrix)):
                    raise QiskitError(
                        """initial_state must be either "ground_state",
                        or a Statevector or DensityMatrix instance."""
                    )
            elif key == "meas_level" and value != 2:
                raise QiskitError("Only meas_level == 2 is supported by PulseSimulator.")
            elif key == "max_outcome_level":
                if (value is not None) and (not isinstance(value, int) or (value <= 0)):
                    raise QiskitError("max_outcome_level must be a positive integer or None.")
            elif key == "experiment_result_function" and not callable(value):
                raise QiskitError("experiment_result_function must be callable.")

            if key == "solver":
                self._set_solver(value)
                validate_subsystem_dims = True
            else:
                if key == "subsystem_dims":
                    validate_subsystem_dims = True
                self._options.update_options(**{key: value})

        # perform additional validation if certain options were modified
        if (
            validate_subsystem_dims
            and np.prod(self._options.subsystem_dims) != self._options.solver.model.dim
        ):
            raise QiskitError(
                "PulseSimulator options subsystem_dims and solver.model.dim are inconsistent."
            )

    def _set_solver(self, solver):
        """Configure simulator based on provided solver."""
        if solver._dt is None:
            raise QiskitError(
                "Solver passed to PulseSimulator is not configured for Pulse simulation."
            )

        self._options.update_options(solver=solver)
        # Get dressed states
        static_hamiltonian = _get_lab_frame_static_hamiltonian(solver.model)
        dressed_evals, dressed_states = _get_dressed_state_decomposition(static_hamiltonian)
        self._dressed_evals = dressed_evals
        self._dressed_states = dressed_states
        self._dressed_states_adjoint = self._dressed_states.conj().transpose()

    # pylint: disable=arguments-differ
    def run(
        self,
        run_input: List[Union[QuantumCircuit, Schedule, ScheduleBlock]],
        validate: Optional[bool] = True,
        **options,
    ) -> DynamicsJob:
        """Run a list of simulations.

        Args:
            run_input: A list of simulations, specified by ``QuantumCircuit``, ``Schedule``, or
                       ``ScheduleBlock`` instances.
            validate: Whether or not to run validation checks on the input.
            **options: Additional run options to temporarily override current backend options.

        Returns:
            DynamicsJob object containing results and status.

        Raises:
            QiskitError: If invalid options are set.
        """

        if validate:
            _validate_run_input(run_input)

        # Configure run options for simulation
        if options:
            backend = copy.copy(self)
            backend.set_options(**options)
        else:
            backend = self

        schedules, num_memory_slots_list = _to_schedule_list(run_input, backend=backend)

        # get the acquires instructions and simulation times
        t_span, measurement_subsystems_list, memory_slot_indices_list = _get_acquire_data(
            schedules, backend.options.subsystem_labels
        )

        # Build and submit job
        job_id = str(uuid.uuid4())
        dynamics_job = DynamicsJob(
            backend=backend,
            job_id=job_id,
            fn=backend._run,
            fn_kwargs={
                "t_span": t_span,
                "schedules": schedules,
                "measurement_subsystems_list": measurement_subsystems_list,
                "memory_slot_indices_list": memory_slot_indices_list,
                "num_memory_slots_list": num_memory_slots_list,
            },
        )
        dynamics_job.submit()

        return dynamics_job

    def _run(
        self,
        job_id,
        t_span,
        schedules,
        measurement_subsystems_list,
        memory_slot_indices_list,
        num_memory_slots_list,
    ) -> Result:
        """Simulate a list of schedules."""

        # simulate all schedules
        y0 = self.options.initial_state
        if y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **self.options.solver_options
        )

        # compute results for each experiment
        experiment_names = [schedule.name for schedule in schedules]
        rng = np.random.default_rng(self.options.seed_simulator)
        experiment_results = []
        for (
            experiment_name,
            solver_result,
            measurement_subsystems,
            memory_slot_indices,
            num_memory_slots,
        ) in zip(
            experiment_names,
            solver_results,
            measurement_subsystems_list,
            memory_slot_indices_list,
            num_memory_slots_list,
        ):
            experiment_results.append(
                self.options.experiment_result_function(
                    experiment_name,
                    solver_result,
                    measurement_subsystems,
                    memory_slot_indices,
                    num_memory_slots,
                    self,
                    seed=rng.integers(low=0, high=9223372036854775807),
                )
            )

        # Construct full result object
        return Result(
            backend_name=self.name,
            backend_version=self.backend_version,
            qobj_id="",
            job_id=job_id,
            success=True,
            results=experiment_results,
            date=datetime.datetime.now().isoformat(),
        )

    @property
    def max_circuits(self):
        return None

    @property
    def target(self) -> Target:
        return self._target

    @property
    def meas_map(self) -> List[List[int]]:
        return self.options.meas_map


def default_experiment_result_function(
    experiment_name: str,
    solver_result: OdeResult,
    measurement_subsystems: List[int],
    memory_slot_indices: List[int],
    num_memory_slots: Union[None, int],
    backend: PulseSimulator,
    seed: Optional[int] = None,
) -> ExperimentResult:
    """Default routine for generating ExperimentResult object.

    Transforms state out of rotating frame into lab frame using ``backend.options.solver``,
    normalizes if ``backend.options.normalize_states==True``, and computes measurement results
    in the dressed basis based on measurement-related options in ``backend.options`` along with
    the measurement specification extracted from the experiments, passed as args to this function.

    Args:
        experiment_name: Name of experiment.
        solver_result: Result object from :class:`Solver.solve`.
        measurement_subsystems: Labels of subsystems in the model being measured.
        memory_slot_indices: Indices of memory slots to store the results in for each subsystem.
        num_memory_slots: Total number of memory slots in the returned output. If ``None``,
            ``max(memory_slot_indices)`` will be used.
        backend: The backend instance that ran the simulation. Various options and properties
            are utilized.
        seed: Seed for any random number generation involved (e.g. when computing outcome samples).

    Returns:
        ExperimentResult object containing results.

    Raises:
        QiskitError: If a specified option is unsupported.
    """

    yf = solver_result.y[-1]
    tf = solver_result.t[-1]

    # Take state out of frame, put in dressed basis, and normalize
    if isinstance(yf, Statevector):
        yf = np.array(backend.options.solver.model.rotating_frame.state_out_of_frame(t=tf, y=yf))
        yf = backend._dressed_states_adjoint @ yf
        yf = Statevector(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.linalg.norm(yf.data)
    elif isinstance(yf, DensityMatrix):
        yf = np.array(
            backend.options.solver.model.rotating_frame.operator_out_of_frame(t=tf, operator=yf)
        )
        yf = backend._dressed_states_adjoint @ yf @ backend._dressed_states
        yf = DensityMatrix(yf, dims=backend.options.subsystem_dims)

        if backend.options.normalize_states:
            yf = yf / np.diag(yf.data).sum()

    if backend.options.meas_level == MeasLevel.CLASSIFIED:

        # compute probabilities for measurement slot values
        measurement_subsystems = [
            backend.options.subsystem_labels.index(x) for x in measurement_subsystems
        ]
        memory_slot_probabilities = _get_memory_slot_probabilities(
            probability_dict=yf.probabilities_dict(qargs=measurement_subsystems),
            memory_slot_indices=memory_slot_indices,
            num_memory_slots=num_memory_slots,
            max_outcome_value=backend.options.max_outcome_level,
        )

        # sample
        memory_samples = _sample_probability_dict(
            memory_slot_probabilities, shots=backend.options.shots, seed=seed
        )
        counts = _get_counts_from_samples(memory_samples)

        # construct results object
        exp_data = ExperimentResultData(
            counts=counts, memory=memory_samples if backend.options.memory else None
        )
        return ExperimentResult(
            shots=backend.options.shots,
            success=True,
            data=exp_data,
            meas_level=MeasLevel.CLASSIFIED,
            seed=seed,
            header=QobjHeader(name=experiment_name),
        )
    else:
        raise QiskitError(f"meas_level=={backend.options.meas_level} not implemented.")


def _validate_run_input(run_input, accept_list=True):
    """Raise errors if the run_input is not one of QuantumCircuit, Schedule, ScheduleBlock, or
    a list of these.
    """
    if isinstance(run_input, list) and accept_list:
        # if list apply recursively, but no longer accept lists
        for x in run_input:
            _validate_run_input(x, accept_list=False)
    elif not isinstance(run_input, (QuantumCircuit, Schedule, ScheduleBlock)):
        raise QiskitError(f"Input type {type(run_input)} not supported by PulseSimulator.run.")


def _get_acquire_data(schedules, valid_subsystem_labels):
    """Get the required data from the acquire commands in each schedule.

    Additionally validates that each schedule has acquire instructions occuring at one time,
    at least one memory slot is being listed, and all measured subsystems exist in
    subsystem_labels.
    """
    t_span_list = []
    measurement_subsystems_list = []
    memory_slot_indices_list = []
    for schedule in schedules:
        schedule_acquires = []
        schedule_acquire_times = []
        for start_time, inst in schedule.instructions:
            # only track acquires saving in a memory slot
            if isinstance(inst, pulse.Acquire) and inst.mem_slot is not None:
                schedule_acquires.append(inst)
                schedule_acquire_times.append(start_time)

        # validate
        if len(schedule_acquire_times) == 0:
            raise QiskitError(
                """At least one measurement saving a a result in a MemorySlot
                must be present in each schedule."""
            )

        for acquire_time in schedule_acquire_times[1:]:
            if acquire_time != schedule_acquire_times[0]:
                raise QiskitError("PulseSimulator.run only supports measurements at one time.")

        t_span_list.append([0, schedule_acquire_times[0]])
        measurement_subsystems = []
        memory_slot_indices = []
        for inst in schedule_acquires:
            if inst.channel.index in valid_subsystem_labels:
                measurement_subsystems.append(inst.channel.index)
            else:
                raise QiskitError(
                    f"""Attempted to measure subsystem {inst.channel.index},
                    but it is not in subsystem_list."""
                )

            memory_slot_indices.append(inst.mem_slot.index)

        measurement_subsystems_list.append(measurement_subsystems)
        memory_slot_indices_list.append(memory_slot_indices)

    return t_span_list, measurement_subsystems_list, memory_slot_indices_list


def _to_schedule_list(
    run_input: List[Union[QuantumCircuit, Schedule, ScheduleBlock]], backend: BackendV2
):
    """Convert all inputs to schedules, and store the number of classical registers present
    in any circuits.
    """
    if not isinstance(run_input, list):
        run_input = [run_input]

    schedules = []
    num_memslots = []
    for sched in run_input:
        num_memslots.append(None)
        if isinstance(sched, ScheduleBlock):
            schedules.append(block_to_schedule(sched))
        elif isinstance(sched, Schedule):
            schedules.append(sched)
        elif isinstance(sched, QuantumCircuit):
            num_memslots[-1] = sched.cregs[0].size
            schedules.append(build_schedule(sched, backend, dt=backend.options.solver._dt))
        else:
            raise QiskitError(f"Type {type(sched)} cannot be converted to Schedule.")
    return schedules, num_memslots
