# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Tests for subsystem_mapping.py."""

from itertools import product

import numpy as np

from qiskit_dynamics.systems import Subsystem, SubsystemMapping, QuantumSystemModel
from qiskit_dynamics.systems.subsystem_operators import X, Y, Z

from ..common import QiskitDynamicsTestCase


def swap(d0, d1):
    """Generate swap operator mapping C^d0 x C^d1 -> C^d1 x C^d0."""
    W = np.zeros((d0 * d1, d0 * d1), dtype=float)

    for idx0, idx1 in product(range(d1), range(d0)):
        Eab = np.zeros((d1, d0), dtype=float)
        Eab[idx0, idx1] = 1
        W = W + np.kron(Eab, Eab.transpose())

    return W


class TestSubsystemMapping(QiskitDynamicsTestCase):
    """Tests for SubsystemMapping class."""

    def test_basic_mapping(self):
        """Test simple example."""

        q0 = Subsystem("Q0", dim=2)
        q1 = Subsystem("Q1", dim=3)
        A = np.eye(3, 2, dtype=complex)

        mapping = SubsystemMapping(matrix=A, in_subsystems=[q0], out_subsystems=[q1])
        op = mapping(X(q0))
        self.assertEqual(op.unmapped_subsystems, [])
        expected_mat = A @ X(q0).matrix() @ A.conj().transpose()
        self.assertAllClose(op.matrix([q1]), expected_mat)

        q2 = Subsystem("Q2", dim=4)
        expected_mat = np.kron(np.eye(4), expected_mat)
        self.assertAllClose(op.matrix([q1, q2]), expected_mat)

    def test_map_with_unmapped_subsystems(self):
        """Mapping on an unspecified subsystem."""
        q0 = Subsystem("Q0", dim=2)
        q1 = Subsystem("Q1", dim=3)
        q2 = Subsystem("Q2", dim=2)
        A = np.eye(3, 2, dtype=complex)

        mapping = SubsystemMapping(matrix=A, in_subsystems=[q0], out_subsystems=[q1])
        op = mapping(X(q2))
        self.assertEqual(op.unmapped_subsystems, [q2])
        expected_mat = np.kron(X(q2).matrix(), A @ np.eye(2) @ A.conj().transpose())
        self.assertAllClose(op.matrix([q1, q2]), expected_mat)

    def test_multiple_subsystem_map(self):
        """Mapping with multiple subsystems."""
        q0 = Subsystem("Q0", dim=2)
        q1 = Subsystem("Q1", dim=3)
        q2 = Subsystem("Q2", dim=2)

        rng = np.random.default_rng(523421)
        A = rng.random((2, 6)) + 1j * rng.random((2, 6))
        mapping = SubsystemMapping(matrix=A, in_subsystems=[q0, q1], out_subsystems=[q2])

        op = mapping(X(q0) @ Y(q1))
        self.assertEqual(op.unmapped_subsystems, [])
        expected_mat = A @ np.kron(Y(q1).matrix(), X(q0).matrix()) @ A.conj().transpose()
        self.assertAllClose(op.matrix([q2]), expected_mat)

        q3 = Subsystem("Q3", dim=4)
        op = mapping(X(q0) @ Y(q1) @ Z(q3))
        self.assertEqual(op.unmapped_subsystems, [q3])
        expected_mat = np.kron(
            Z(q3).matrix(), A @ np.kron(Y(q1).matrix(), X(q0).matrix()) @ A.conj().transpose()
        )
        self.assertAllClose(op.matrix([q2, q3]), expected_mat)

    def test_disjoint_outputs(self):
        """Mapping with multiple input and output subsystems that get output in weird order."""
        q0 = Subsystem("Q0", dim=2)
        q1 = Subsystem("Q1", dim=3)
        q2 = Subsystem("Q2", dim=2)
        q3 = Subsystem("Q3", dim=4)
        q4 = Subsystem("Q4", dim=5)

        rng = np.random.default_rng(234223)
        A = rng.random((8, 6)) + 1j * rng.random((8, 6))
        mapping = SubsystemMapping(matrix=A, in_subsystems=[q0, q1], out_subsystems=[q2, q3])

        op = mapping(X(q0) @ Y(q1) @ Z(q4))
        self.assertEqual(op.unmapped_subsystems, [q4])
        expected_mat = np.kron(
            Z(q4).matrix(), A @ np.kron(Y(q1).matrix(), X(q0).matrix()) @ A.conj().transpose()
        )
        expected_mat = (
            np.kron(swap(5, 4), np.eye(2)) @ expected_mat @ np.kron(swap(4, 5), np.eye(2))
        )
        self.assertAllClose(op.matrix([q2, q4, q3]), expected_mat)

    def test_QuantumSystemModel_mapping(self):
        """Test simple example."""

        q0 = Subsystem("Q0", dim=2)
        q1 = Subsystem("Q1", dim=3)
        A = np.eye(3, 2, dtype=complex)

        mapping = SubsystemMapping(matrix=A, in_subsystems=[q0], out_subsystems=[q1])
        model = QuantumSystemModel(
            static_hamiltonian=Z(q0),
            drive_hamiltonian_coefficients=["d0"],
            drive_hamiltonians=[X(q0)],
        )

        out_model = mapping(model)

        self.assertEqual(out_model.subsystems, [q1])
        self.assertAllClose(
            out_model.static_hamiltonian.matrix(), A @ np.diag([1, -1]) @ A.conj().transpose()
        )
        self.assertAllClose(
            out_model.drive_hamiltonians[0].matrix(),
            A @ np.array([[0, 1], [1, 0]]) @ A.conj().transpose(),
        )
