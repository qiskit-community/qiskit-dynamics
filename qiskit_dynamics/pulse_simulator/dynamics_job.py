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

"""This module implements the job class used for PulseSimulator objects."""

from typing import Callable, Dict

from qiskit.providers.backend import Backend
from qiskit.providers import JobV1 as Job
from qiskit.providers import JobStatus, JobError


class DynamicsJob(Job):
    """DynamicsJob class for PulseSimulator."""

    _async = False

    def __init__(
        self, backend: Backend, job_id: str, fn: Callable, fn_kwargs: Dict, **kwargs
    ) -> None:
        """Initializes the job.

        Args:
            backend: The backend used to run the job.
            job_id: A unique id in the context of the backend used to run the job.
            fn: Function to run the simulation.
            fn_kwargs: Kwargs for fn.
            kwargs: Any key value metadata to associate with this job.
        """
        super().__init__(backend, job_id, **kwargs)
        self._fn = fn
        self._fn_kwargs = fn_kwargs
        self._result = None

    def submit(self):
        """Run the simulation.

        Raises:
            JobError: if trying to re-submit the job.
        """
        if self._result is not None:
            raise JobError("Dynamics job has already been submitted.")
        self._result = self._fn(job_id=self.job_id(), **self._fn_kwargs)

    def result(self):
        """Get job result.

        Returns:
            qiskit.Result: Result object
        Raises:
            JobError: If job has not been submitted.
        """
        if self._result is None:
            raise JobError("Job has not been submitted.")
        return self._result

    def status(self):
        """Gets the status of the job.

        Returns:
            JobStatus: The current JobStatus.
        """
        if self._result is None:
            return JobStatus.INITIALIZING

        return JobStatus.DONE
