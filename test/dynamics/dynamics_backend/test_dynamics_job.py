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
"""
Test DynamicsJob object.
"""

from qiskit.providers import JobStatus, JobError

from qiskit_dynamics.dynamics_backend.dynamics_job import DynamicsJob
from ..common import QiskitDynamicsTestCase


class TestDynamicsJob(QiskitDynamicsTestCase):
    """Tests DynamicsJob."""

    def setUp(self):
        # pylint: disable=unused-argument
        def eval_func(job_id, x):
            return x**2

        self.simple_job = DynamicsJob(
            backend="", job_id="123", fn=eval_func, fn_kwargs={"x": 2}, other_kwarg="for testing"
        )

    def test_submit_and_get_result(self):
        """Test basis job submission and results gathering."""
        self.simple_job.submit()
        self.assertTrue(self.simple_job.result() == 4)

    def test_no_result_error(self):
        """Test error is raised if job not initially submitted."""
        with self.assertRaisesRegex(JobError, "Job has not been submitted."):
            self.simple_job.result()

    def test_double_submit_error(self):
        """Test error is raised if job not initially submitted."""
        with self.assertRaisesRegex(JobError, "Dynamics job has already been submitted."):
            self.simple_job.submit()
            self.simple_job.submit()

    def test_status(self):
        """Test correct status return."""
        self.assertTrue(self.simple_job.status() == JobStatus.INITIALIZING)
        self.simple_job.submit()
        self.assertTrue(self.simple_job.status() == JobStatus.DONE)

    def test_metadata(self):
        """Test metadata storage."""
        self.assertTrue(self.simple_job.metadata == {"other_kwarg": "for testing"})
