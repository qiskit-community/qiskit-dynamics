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
Test PulseSimulator.
"""

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics import Solver, PulseSimulator

from qiskit_dynamics.pulse_simulator.pulse_simulator import (
    _validate_run_input,
    _get_acquire_data,
    _to_schedule_list,
)
from ..common import QiskitDynamicsTestCase


class TestPulseSimulatorValidation(QiskitDynamicsTestCase):
    """Test validation checks."""

    def setUp(self):
        """Build simple simulator for multiple tests."""

        solver = Solver(
            static_hamiltonian=np.array([[1., 0.], [0. -1.]]),
            hamiltonian_operators=[np.array([[0., 1.], [1., 0.]])],
            hamiltonian_channels=['d0'],
            channel_carrier_freqs={'d0': 1.},
            dt=1.
        )

        self.simple_simulator = PulseSimulator(solver=solver)

    def test_solver_not_configured_for_pulse(self):
        """Test error is raised if solver not configured for pulse simulation."""

        solver = Solver(
            static_hamiltonian=np.array([[1., 0.], [0. -1.]]),
            hamiltonian_operators=[np.array([[0., 1.], [1., 0.]])],
        )

        with self.assertRaisesRegex(QiskitError, "not configured for Pulse"):
            PulseSimulator(solver=solver)
