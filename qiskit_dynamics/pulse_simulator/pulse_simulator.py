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

from qiskit import pulse
from qiskit.qobj.utils import MeasLevel
from qiskit.qobj.common import QobjHeader
from qiskit.transpiler import Target
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.providers import Provider
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
    """Pulse enabled simulator backend."""

    def __init__(
        self,
        solver: Solver,
        target: Optional[Target] = None,
        provider: Optional[Provider] = None,
        **options,
    ):
        """Instantiate with a :class:`.Solver` instance and additional options.

        Args:
            solver: Solver instance configured for pulse simulation.
            target: Target object.
            provider: Provider of the backend.
            options: Additional configuration options for the simulator.
        Raises:
            QiskitError: If any instantiation arguments fail validation checks.
        """

        super().__init__(
            name="PulseSimulator",
            description="Pulse enabled simulator backend.",
            backend_version=0.1,
            provider=provider,
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

        # add default simulator measure instructions
        instruction_schedule_map = target.instruction_schedule_map()
        for qubit in self.options.subsystem_labels:
            if not instruction_schedule_map.has(instruction="measure", qubits=qubit):
                with pulse.build() as meas_sched:
                    pulse.acquire(
                        duration=1, qubit_or_channel=qubit, register=pulse.MemorySlot(qubit)
                    )

            instruction_schedule_map.add(
                instruction="measure", qubits=qubit, schedule=meas_sched
            )

        self._target = target

    def _default_options(self):
        return Options(
            shots=1024,
            solver=None,
            solver_options={},
            subsystem_labels=None,
            subsystem_dims=None,
            meas_map=None,
            normalize_states=True,
            initial_state="ground_state",
            meas_level=MeasLevel.CLASSIFIED,
            memory=True,
            seed_simulator=None,
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
            elif key == "meas_level":
                if value != 2:
                    raise QiskitError("Only meas_level == 2 is supported by PulseSimulator.")

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
            # TODO: We might need to implement a copy or __copy__ method
            # to make sure this copying works correctly without a deepcopy
            backend = copy.copy(self)
            backend.set_options(**options)
        else:
            backend = self

        initial_state = backend.options.initial_state
        if initial_state == "ground_state":
            initial_state = Statevector(backend._dressed_states[:, 0])

        schedules, schedules_memslot_nums = _to_schedule_list(run_input, backend=self)

        # get the acquires instructions and simulation times
        t_span, measurement_subsystems_list, memory_slot_indices_list = _get_acquire_data(
            schedules, self.options.subsystem_labels
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
                "schedules_memslot_nums": schedules_memslot_nums,
                "memory_slot_indices_list": memory_slot_indices_list,
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
        schedules_memslot_nums,
        memory_slot_indices_list,
    ) -> Result:
        """Simulate a list of schedules."""

        y0 = self.options.initial_state
        if y0 == "ground_state":
            y0 = Statevector(self._dressed_states[:, 0])

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **self.options.solver_options
        )

        schedule_names = [schedule.name for schedule in schedules]

        # build random number generator for count sampling
        rng = np.random.default_rng(self.options.seed_simulator)

        # construct outputs for each experiment
        experiment_results = []
        for (
            ts,
            result,
            measurement_subsystems,
            memory_slot_indices,
            num_memory_slots,
            schedule_name,
        ) in zip(
            t_span,
            solver_results,
            measurement_subsystems_list,
            memory_slot_indices_list,
            schedules_memslot_nums,
            schedule_names,
        ):
            yf = result.y[-1]

            # Take state out of frame, put in dressed basis, and normalize
            if isinstance(yf, Statevector):
                yf = np.array(
                    self.options.solver.model.rotating_frame.state_out_of_frame(t=ts[-1], y=yf)
                )
                yf = self._dressed_states_adjoint @ yf
                yf = Statevector(yf, dims=self.options.subsystem_dims)

                if self.options.normalize_states:
                    yf = yf / np.linalg.norm(yf.data)
            elif isinstance(yf, DensityMatrix):
                yf = np.array(
                    self.options.solver.model.rotating_frame.operator_out_of_frame(
                        t=ts[-1], operator=yf
                    )
                )
                yf = self._dressed_states_adjoint @ yf @ self._dressed_states
                yf = DensityMatrix(yf, dims=self.options.subsystem_dims)

                if self.options.normalize_states:
                    yf = yf / np.diag(yf.data).sum()

            # construct experiment results
            if self.options.meas_level == MeasLevel.CLASSIFIED:

                # compute probabilities for measurement slot values
                measurement_subsystems = [
                    self.options.subsystem_labels.index(x) for x in measurement_subsystems
                ]
                memory_slot_probabilities = _get_memory_slot_probabilities(
                    probability_dict=yf.probabilities_dict(qargs=measurement_subsystems),
                    memory_slot_indices=memory_slot_indices,
                    num_memory_slots=num_memory_slots,
                    max_outcome_value=1,
                )

                # sample
                seed = rng.integers(low=0, high=9223372036854775807)
                memory_samples = _sample_probability_dict(
                    memory_slot_probabilities, shots=self.options.shots, seed=seed
                )
                counts = _get_counts_from_samples(memory_samples)

                exp_data = ExperimentResultData(
                    counts=counts, memory=memory_samples if self.options.memory else None
                )
                experiment_results.append(
                    ExperimentResult(
                        shots=self.options.shots,
                        success=True,
                        data=exp_data,
                        meas_level=MeasLevel.CLASSIFIED,
                        seed=seed,
                        header=QobjHeader(name=schedule_name),
                    )
                )
            else:
                raise QiskitError(
                    f"Only meas_level=={MeasLevel.CLASSIFIED} is supported by PulseSimulator."
                )

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
