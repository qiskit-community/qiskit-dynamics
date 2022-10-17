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
import numpy as np

from qiskit import pulse
from qiskit.pulse import Schedule, ScheduleBlock
from qiskit.pulse.channels import AcquireChannel, DriveChannel, MeasureChannel, ControlChannel
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit.providers.backend import BackendV2
from qiskit.result import Result

from qiskit import QiskitError
from qiskit.quantum_info import Statevector, DensityMatrix

from qiskit_dynamics import Solver
from qiskit_dynamics.array import Array
from qiskit_dynamics.models import HamiltonianModel

from .dynamics_job import DynamicsJob
from .pulse_utils import _get_dressed_state_decomposition


class PulseSimulator(BackendV2):
    def __init__(
        self,
        solver: Solver,
        subsystem_dims,
        subsystem_labels: Optional[List[int]] = None,
        name: Optional[str] = 'PulseSimulator',
    ):
        """This init needs fleshing out. Need to determine all that is necessary for each use case.

        Assumptions
            - Solver is well-formed.
            - Solver Hamiltonian operators are specified in undressed basis using standard
              tensor-product convention (whether dense or sparse evaluation)

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
            description='Pulse enabled simulator backend.',
            backend_version=0.1
        )

        ##############################################################################################
        # Make these into properties?
        self.solver = solver
        self.subsystem_dims = subsystem_dims or [solver.model.dim]
        self.subsystem_labels = subsystem_labels or np.arange(len(subsystem_dims), dtype=int)

        # get the static hamiltonian in the lab frame and undressed basis
        # assumes that the solver was constructed with operators specified in lab frame
        # using standard tensor product structure
        static_hamiltonian = None
        if isinstance(self.solver.model, HamiltonianModel):
            static_hamiltonian = self.solver.model.static_operator
        else:
            static_hamiltonian = self.solver.model.static_hamiltonian

        rotating_frame = self.solver.model.rotating_frame
        static_hamiltonian = 1j * rotating_frame.generator_out_of_frame(
            t=0., operator=-1j * static_hamiltonian
        )

        # convert to numpy array
        static_hamiltonian = np.array(Array(static_hamiltonian).data)

        # get the dressed states
        ##############################################################################################
        # Make these into properties?
        dressed_evals, dressed_states = _get_dressed_state_decomposition(static_hamiltonian)
        self._dressed_evals = dressed_evals
        self._dressed_states = dressed_states
        self._dressed_states_adjoint = self._dressed_states.conj().transpose()

    def run(
        self,
        run_input: Union[Schedule, ScheduleBlock],
        shots: int = 1,
        y0 = None,
        validate: Optional[bool] = False,
        solver_options: Optional[dict] = None,
        **options
    ) -> Result:
        """Run on the backend.

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
        """

        if validate:
            _validate_run_input(run_input)

        if solver_options is None:
            solver_options = {}

        if y0 is None:
            y0 = Statevector(self._dressed_states[:, 0])

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
                't_span': t_span, 'y0': y0, 'schedules': schedules,
                'solver_options': solver_options,
                'measurement_subsystems_list': measurement_subsystems_list,
                'shots': shots
            }
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
        measurement_subsystems_list,
        shots
    ) -> Result:
        """Not sure here what the right delineation of arguments is to put in _run.
        This feels somewhat hacky/arbitrary.
        """
        start = time.time()

        # map measurement subsystems from labels to correct index
        if self.subsystem_labels:
            new_measurement_subsystems_list = []
            for measurement_subsystems in measurement_subsystems_list:
                new_measurement_subsystems = []
                for subsystem in measurement_subsystems:
                    if subsystem in self.subsystem_labels:
                        new_measurement_subsystems.append(self.subsystem_labels.index(subsystem))
                    else:
                        raise QiskitError(f"Attempted to measure subsystem {subsystem}, but it is not in subsystem_list.")
                new_measurement_subsystems_list.append(new_measurement_subsystems)

            measurement_subsystems_list = new_measurement_subsystems_list

        solver_results = self.solver.solve(t_span=t_span, y0=y0, signals=schedules, **solver_options)

        # construct counts for each experiment
        ##############################################################################################
        # Change to if statement depending on types of outputs, e.g. counts vs IQ data
        outputs = []
        for ts, result, measurement_subsystems in zip(t_span, solver_results, measurement_subsystems_list):
            yf = result.y[-1]

            # Put state in dressed basis and sample counts
            if isinstance(yf, Statevector):
                yf = np.array(self.solver.model.rotating_frame.state_out_of_frame(t=ts[-1], y=yf))
                yf = self._dressed_states_adjoint @ yf
                yf = Statevector(yf, dims=self.subsystem_dims)
            elif isinstance(yf, DensityMatrix):
                yf = np.array(self.solver.model.rotating_frame.operator_out_of_frame(t=ts[-1], y=yf))
                yf = self._dressed_states_adjoint @ yf @ self.dressed_states
                yf = DensityMatrix(yf, dims=self.subsystem_dims)

            outputs.append({'counts': yf.sample_counts(shots=shots, qargs=measurement_subsystems)})

        results_list = []
        for schedule, output in zip(schedules, outputs):
            results_list.append(
                Result(
                    backend_name=self.name,
                    backend_version=self.backend_version,
                    qobj_id=None, # Should we subclass result or something because qobj_id doesn't exist?
                    job_id=job_id,
                    success=True,
                    results=output,
                    date=datetime.datetime.now().isoformat()
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

    def _default_options(self):
        pass

    def max_circuits(self):
        pass

    def target(self):
        pass


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
            raise Exception('invalid Schedule type')
    return new_schedules
