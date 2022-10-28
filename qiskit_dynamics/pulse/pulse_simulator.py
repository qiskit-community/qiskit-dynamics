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

import time
import datetime
import uuid

from typing import Dict, Iterable, List, Optional, Union
import copy
import numpy as np

from qiskit import pulse
from qiskit.transpiler import Target
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.channels import AcquireChannel, DriveChannel, MeasureChannel, ControlChannel
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.providers.options import Options
from qiskit.providers.backend import BackendV2
from qiskit.result import Result

from qiskit import QiskitError
from qiskit.quantum_info import Statevector, DensityMatrix

from qiskit_dynamics import Solver
from qiskit_dynamics.array import Array
from qiskit_dynamics.models import HamiltonianModel

from .dynamics_job import DynamicsJob
from .pulse_utils import _get_dressed_state_decomposition, _get_lab_frame_static_hamiltonian


class PulseSimulator(BackendV2):
    def __init__(
        self,
        solver: Solver,
        name: Optional[str] = "PulseSimulator",
        target: Optional[Target] = None,
        **options,
    ):
        """This init needs fleshing out. Need to determine all that is necessary for each use case.

        Assumptions
            - Solver is well-formed.
            - The no-rotating frame Solver Hamiltonian operators are specified in undressed
              basis using standard tensor-product convention (whether dense or sparse evaluation)

        Design questions
            - Simulating measurement requires subsystem dims and labels, should we allow this
              to be constructed without them, and then measurement is just not possible?
            - Should we add the ability to do custom measurements?
            - Should we fill out defaults by extracting them from Solver?
            - If no set frequency commands are set in the schedule, what should the channel
              frequencies be?
                - How are control channel frequencies handled these days?
            - Fill out properties

        Notes:
            - Add validation of the Solver object, verifying that its configured to simulate pulse
              schedules
        """
        ##############################################################################################
        # what to put for provider?
        super().__init__(
            provider=None,
            name=name,
            description="Pulse enabled simulator backend.",
            backend_version=0.1,
        )
        # TODO: We should have a default configuration of a target if
        # none is provided, and a modification of a provided one to change any
        # simulator specific attrbiutes to make it compatible
        self._target = target or Target()
        self._options_target = {}

        # Dressed states of solver, will be calculated when solver option is set
        self._dressed_evals = None
        self._dressed_states = None
        self._dressed_states_adjoint = None

        # Set simulator options
        self.set_options(solver=solver, **options)

    def _default_options(self):
        return Options(
            shots=1024,
            solver=None,
            solver_options={},
            subsystem_labels=None,
            subsystem_dims=None,  # Is this needed as explicit option?
            normalize_states=True,
            initial_state=None,  # Do we need this? (y0)
        )

    def set_options(self, **fields):
        for key, value in fields.items():
            if hasattr(self._target, key):
                self._set_target_option(key, value)
            elif not hasattr(self._options, key):
                raise AttributeError("Invalid option %s" % key)
            elif key == "solver":
                # Special handling for solver setting
                self._set_solver(value)
            else:
                setattr(self._options, key, value)

    def _set_solver(self, solver):
        """Configure simulator based on provided solver."""
        self._options.solver = solver
        # Get dressed states
        static_hamiltonian = _get_lab_frame_static_hamiltonian(solver.model)
        dressed_evals, dressed_states = _get_dressed_state_decomposition(static_hamiltonian)
        self._dressed_evals = dressed_evals
        self._dressed_states = dressed_states
        self._dressed_states_adjoint = self._dressed_states.conj().transpose()

    def _set_target_option(self, key: str, value: any):
        """"""
        if value is not None:
            self._options_target[key] = value
        elif key in self._options_target:
            # if value is None reset to default target value
            self._options_target.pop(key)

    def run(
        self,
        run_input: Union[Schedule, ScheduleBlock],
        validate: Optional[bool] = False,
        **options,
    ) -> Result:
        """Run on the backend.

        normalize_states ensures that, when sampling, the state is normalized, to avoid errors
        being raised due to numerical tolerance issues.

        Notes/questions:
        - Should we force y0 to be a quantum_info state? This currently assumes that
        - Should we provide optional arguments to run that allow the user to specify different
          modes of simulation? E.g.
            - Just simulate the state
            - Simulate the unitary or process
            - Return probabilities
            - Return counts
          For now, assuming just counts
        - Validate which channels are available on device before running? Can the BackendV2 be
          set up to do this automatically somehow?
        - What is the formatting for the results when measuring a subset of qubits? E.g. if
          you measure qubits [0, 2, 3], do you get counts for 3-bit strings out?
        - Add validation that only qubits present in the model are being measured. This may
          happen automatically as the measurement code will fail, but would be good to raise
          a proper error.
        - How to handle binning of higher level measurements? E.g. should everything above 0
          count as 1?
        - What to do with memory and register slots?

        To test:
        - Measuring a subset of qubits when more than one present
        - If t_span=[0, 0] it seems that NaNs appear for JAX simulation - should solve this.
        - Normalizing the state.
        """

        if validate:
            _validate_run_input(run_input)

        # Configure run options for simulation
        run_options = copy.copy(self.options.__dict__)
        for key, value in options.items():
            run_options[key] = value
        if run_options["initial_state"] is None:
            run_options["initial_state"] = Statevector(self._dressed_states[:, 0])

        # to do: add handling of circuits
        schedules = _to_schedule_list(run_input)

        # get the acquires instructions and simulation times
        t_span = []
        measurement_subsystems_list = []
        for schedule in schedules:
            schedule_acquires = []
            schedule_acquire_times = []
            for start_time, inst in schedule.instructions:
                if isinstance(inst, pulse.Acquire):
                    schedule_acquires.append(inst)
                    schedule_acquire_times.append(start_time)

            # maybe need to validate more here
            _validate_acquires(schedule_acquire_times, schedule_acquires)

            t_span.append([0, schedule_acquire_times[0]])
            measurement_subsystems_list.append([inst.channel.index for inst in schedule_acquires])
            measurement_subsystems_list[-1].sort()

        # Build and submit job
        job_id = str(uuid.uuid4())
        dynamics_job = DynamicsJob(
            backend=self,
            job_id=job_id,
            fn=self._run,
            fn_kwargs={
                "t_span": t_span,
                "y0": run_options["initial_state"],
                "schedules": schedules,
                "solver_options": run_options["solver_options"],
                "measurement_subsystems_list": measurement_subsystems_list,
                "normalize_states": run_options["normalize_states"],
                "shots": run_options["shots"],
            },
        )
        dynamics_job.submit()

        return dynamics_job

    def _run(
        self,
        job_id,
        t_span,
        y0,
        schedules,
        solver_options,
        normalize_states,
        measurement_subsystems_list,
        shots,
    ) -> Result:
        """Not sure here what the right delineation of arguments is to put in _run.
        This feels somewhat hacky/arbitrary.
        """
        start = time.time()
        subsystem_dims = self.options.subsystem_dims or [self.options.solver.model.dim]
        subsystem_labels = self.options.subsystem_labels or list(range(len(subsystem_dims)))

        # map measurement subsystems from labels to correct index
        if subsystem_labels:
            new_measurement_subsystems_list = []
            for measurement_subsystems in measurement_subsystems_list:
                new_measurement_subsystems = []
                for subsystem in measurement_subsystems:
                    if subsystem in subsystem_labels:
                        new_measurement_subsystems.append(subsystem_labels.index(subsystem))
                    else:
                        raise QiskitError(
                            f"Attempted to measure subsystem {subsystem}, but it is not in subsystem_list."
                        )
                new_measurement_subsystems_list.append(new_measurement_subsystems)

            measurement_subsystems_list = new_measurement_subsystems_list

        solver_results = self.options.solver.solve(
            t_span=t_span, y0=y0, signals=schedules, **solver_options
        )

        # construct counts for each experiment
        ##############################################################################################
        # Change to if statement depending on types of outputs, e.g. counts vs IQ data
        outputs = []
        for ts, result, measurement_subsystems in zip(
            t_span, solver_results, measurement_subsystems_list
        ):
            yf = result.y[-1]

            # Put state in dressed basis and sample counts
            if isinstance(yf, Statevector):
                yf = np.array(
                    self.options.solver.model.rotating_frame.state_out_of_frame(t=ts[-1], y=yf)
                )
                yf = self._dressed_states_adjoint @ yf
                yf = Statevector(yf, dims=subsystem_dims)

                if normalize_states:
                    yf = yf / np.linalg.norm(yf.data)
            elif isinstance(yf, DensityMatrix):
                yf = np.array(
                    self.options.solver.model.rotating_frame.operator_out_of_frame(t=ts[-1], y=yf)
                )
                yf = self._dressed_states_adjoint @ yf @ self.dressed_states
                yf = DensityMatrix(yf, dims=subsystem_dims)

                if normalize_states:
                    yf = yf / np.diag(yf.data).sum()

            outputs.append({"counts": yf.sample_counts(shots=shots, qargs=measurement_subsystems)})

        results_list = []
        for schedule, output in zip(schedules, outputs):
            results_list.append(
                Result(
                    backend_name=self.name,
                    backend_version=self.backend_version,
                    qobj_id=None,  # Should we subclass result or something because qobj_id doesn't exist?
                    job_id=job_id,
                    success=True,
                    results=output,
                    date=datetime.datetime.now().isoformat(),
                )
            )

        return results_list

    @property
    def meas_map(self) -> List[List[int]]:
        pass

    def acquire_channel(self, qubit: Iterable[int]) -> Union[int, AcquireChannel, None]:
        pass

    def drive_channel(self, qubit: int) -> Union[int, DriveChannel, None]:
        pass

    def control_channel(self, qubit: int) -> Union[int, ControlChannel, None]:
        pass

    def measure_channel(self, qubit: int) -> Union[int, MeasureChannel, None]:
        pass

    def max_circuits(self):
        return None

    def target(self) -> Target:
        target = copy.copy(self._target)
        # Override default target with any currently set option values
        for key, val in self._options_target.items():
            setattr(target, key, val)
        # Any extra custom simulator handling of target can be done here
        return target


def _validate_run_input(run_input, accept_list=True):
    """Raise errors if the run_input is invalid."""
    if isinstance(run_input, list) and accept_list:
        # if list apply recursively, but no longer accept lists
        for x in run_input:
            _validate_run_input(x, accept_list=False)
    elif not isinstance(run_input, (Schedule, ScheduleBlock)):
        raise QiskitError(f"Input type {type(run_input)} not supported by PulseSimulator.run.")


#######################################################################################################
# this should be rolled into _validate_experiments?
def _validate_acquires(schedule_acquire_times, schedule_acquires):
    """Validate the acquire instructions.
    For now, make sure all acquires happen at one time.
    """

    if len(schedule_acquire_times) == 0:
        raise QiskitError("At least one measurement must be present in each schedule.")

    start_time = schedule_acquire_times[0]
    for time in schedule_acquire_times[1:]:
        if time != start_time:
            raise QiskitError("PulseSimulator.run only supports measurements at one time.")


def _to_schedule_list(schedules):
    if not isinstance(schedules, list):
        schedules = [schedules]

    new_schedules = []
    for sched in schedules:
        if isinstance(sched, pulse.ScheduleBlock):
            new_schedules.append(block_to_schedule(sched))
        elif isinstance(sched, pulse.Schedule):
            new_schedules.append(sched)
        else:
            raise QiskitError("Invalid Schedule type.")
    return new_schedules
