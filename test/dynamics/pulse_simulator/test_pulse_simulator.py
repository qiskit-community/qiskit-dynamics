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

from qiskit import QiskitError, pulse

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
            static_hamiltonian=np.array([[1., 0.], [0., -1.]]),
            hamiltonian_operators=[np.array([[0., 1.], [1., 0.]])],
            hamiltonian_channels=['d0'],
            channel_carrier_freqs={'d0': 1.},
            dt=1.
        )

        self.simple_simulator = PulseSimulator(solver=solver)

    def test_solver_not_configured_for_pulse(self):
        """Test error is raised if solver not configured for pulse simulation."""

        solver = Solver(
            static_hamiltonian=np.array([[1., 0.], [0., -1.]]),
            hamiltonian_operators=[np.array([[0., 1.], [1., 0.]])],
        )

        with self.assertRaisesRegex(QiskitError, "not configured for Pulse"):
            PulseSimulator(solver=solver)

    def test_run_input_error(self):
        """Test submission of invalid run input."""

        with self.assertRaisesRegex(QiskitError, "not supported by PulseSimulator.run."):
            self.simple_simulator.run([1])

    def test_subsystem_dims_inconsistency(self):
        """Test that setting subsystem_dims inconsistently with solver.model.dim raises error."""

        with self.assertRaisesRegex(QiskitError, "inconsistent"):
            self.simple_simulator.set_options(subsystem_dims=[4])

    def test_no_measurements_in_schedule(self):
        """Test that running a schedule with no measurements raises an error."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))

        with self.assertRaisesRegex(QiskitError, "At least one measurement"):
            self.simple_simulator.run(schedule)

    def test_multiple_measurements_in_schedule(self):
        """Test error raising when attempting to run a schedule with multiple measurements."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))
                pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        with self.assertRaisesRegex(QiskitError, "only supports measurements at one time"):
            self.simple_simulator.run(schedule)

    def test_measure_nonexistant_subsystem(self):
        """Attempt to measure subsystem that doesn't exist."""

        with pulse.build() as schedule:
            pulse.play(pulse.Waveform([0.5, 0.5, 0.5]), pulse.DriveChannel(0))
            pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(0))

        with self.assertRaisesRegex(QiskitError, "Attempted to measure subsystem 1"):
            self.simple_simulator.run(schedule)


class TestPulseSimulator(QiskitDynamicsTestCase):
    """Basic tests for PulseSimulator."""

    def setUp(self):
        """Build reusable models."""

        static_ham = 2 * np.pi * 5 * np.array([[-1., 0.], [0., 1.]]) / 2
        drive_op = 2 * np.pi * 0.1 * np.array([[0., 1.], [1., 0.]]) / 2

        solver = Solver(
            static_hamiltonian=static_ham,
            hamiltonian_operators=[drive_op],
            hamiltonian_channels=['d0'],
            channel_carrier_freqs={'d0': 5.},
            dt=0.1,
            rotating_frame=static_ham
        )

        self.simple_simulator = PulseSimulator(solver=solver)

    def test_pi_pulse(self):
        """Test simulation of a pi pulse."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 100), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_simulator.run(schedule, seed_simulator=1234567).result()
        self.assertDictEqual(result.get_counts(), {"1": 1024})
        self.assertTrue(result.get_memory() == ["1"] * 1024)

    def test_pi_half_pulse(self):
        """Test simulation of a pi/2 pulse."""

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))

        result = self.simple_simulator.run(schedule, seed_simulator=398472).result()
        self.assertDictEqual(result.get_counts(), {'0': 505, '1': 519})

    def test_pi_half_pulse_relabelled(self):
        """Test simulation of a pi/2 pulse with qubit relabelled."""

        self.simple_simulator.set_options(subsystem_labels=[1])

        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(pulse.Waveform([1.0] * 50), pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=1, register=pulse.MemorySlot(1))

        result = self.simple_simulator.run(schedule, seed_simulator=398472).result()
        self.assertDictEqual(result.get_counts(), {'00': 505, '10': 519})
