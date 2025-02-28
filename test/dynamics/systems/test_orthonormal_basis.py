# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,no-member

"""Tests for orthonormal_basis.py."""

from itertools import product
from functools import partial

import numpy as np

from qiskit_dynamics.systems import Subsystem, ONBasis, DressedBasis
from qiskit_dynamics.systems.orthonormal_basis import _sorted_eigh

from ..common import QiskitDynamicsTestCase, test_array_backends


class TestONBasis(QiskitDynamicsTestCase):
    """Tests for Subsystem class."""

    def test_default_standard_basis(self):
        """Test construction of default standard basis."""
        s = Subsystem(name="Q0", dim=5)
        basis = ONBasis(subsystems=[s])
        self.assertAllClose(basis.basis_vectors, np.eye(5, dtype=complex))

    def test_default_labels(self):
        """Test construction of default labels."""

        s0 = Subsystem(name="Q0", dim=5)
        basis = ONBasis(basis_vectors=np.eye(5, dtype=complex), subsystems=[s0])
        self.assertEqual(basis.labels, [(idx,) for idx in range(5)])

        s1 = Subsystem(name="Q1", dim=3)
        basis = ONBasis(basis_vectors=np.eye(15, dtype=complex), subsystems=[s0, s1])
        self.assertEqual(basis.labels, [(y, x) for x, y in product(range(5), range(3))])

    def test_subset(self):
        """Test restricting to a subset."""

        s0 = Subsystem(name="Q0", dim=5)
        s1 = Subsystem(name="Q1", dim=3)
        basis = ONBasis(basis_vectors=np.eye(15, dtype=complex), subsystems=[s0, s1])

        sub_basis = basis.subset(condition=lambda label: all(x <= 1 for x in label))

        self.assertEqual(sub_basis.labels, [(0, 0), (1, 0), (0, 1), (1, 1)])
        expected_basis = np.zeros((15, 4), dtype=complex)
        expected_basis[0, 0] = 1.0  # (0, 0)
        expected_basis[1, 1] = 1.0  # (1, 0)
        expected_basis[3, 2] = 1.0  # (0, 1)
        expected_basis[4, 3] = 1.0  # (1, 1)
        self.assertAllClose(sub_basis.basis_vectors, expected_basis)


class TestDressedBasis(QiskitDynamicsTestCase):
    """Test the DressedBasis class."""

    def test_label_construction(self):
        """Test correct construction of labels."""
        s0 = Subsystem("Q0", dim=3)
        s1 = Subsystem("Q1", dim=3)
        basis = DressedBasis(
            subsystems=[s0, s1], basis_vectors=np.eye(9, dtype=complex), evals=np.arange(9)
        )
        expected_labels = [
            {"index": (0, 0), "eval": 0},
            {"index": (1, 0), "eval": 1},
            {"index": (2, 0), "eval": 2},
            {"index": (0, 1), "eval": 3},
            {"index": (1, 1), "eval": 4},
            {"index": (2, 1), "eval": 5},
            {"index": (0, 2), "eval": 6},
            {"index": (1, 2), "eval": 7},
            {"index": (2, 2), "eval": 8},
        ]
        self.assertEqual(basis.labels, expected_labels)

    def test_ground_state(self):
        """Test ground state property."""

        s0 = Subsystem("Q0", dim=3)
        s1 = Subsystem("Q1", dim=3)
        basis = DressedBasis(
            subsystems=[s0, s1], basis_vectors=np.eye(9, dtype=complex), evals=np.arange(9)
        )

        self.assertAllClose(basis.basis_vectors[:, 0], basis.ground_state)

    def test_computational_subspace(self):
        """Test restriction to computational subspace."""

        s0 = Subsystem("Q0", dim=3)
        s1 = Subsystem("Q1", dim=3)
        basis = DressedBasis(
            subsystems=[s0, s1], basis_vectors=np.eye(9, dtype=complex), evals=np.arange(9)
        )

        comp_subspace = basis.computational_states
        self.assertEqual(
            comp_subspace.labels,
            [
                {"index": (0, 0), "eval": 0},
                {"index": (1, 0), "eval": 1},
                {"index": (0, 1), "eval": 3},
                {"index": (1, 1), "eval": 4},
            ],
        )
        expected_basis = np.zeros((9, 4), dtype=complex)
        expected_basis[0, 0] = 1.0  # (0, 0)
        expected_basis[1, 1] = 1.0  # (1, 0)
        expected_basis[3, 2] = 1.0  # (0, 1)
        expected_basis[4, 3] = 1.0  # (1, 1)

        self.assertAllClose(comp_subspace.basis_vectors, expected_basis)

    def test_low_energy_states(self):
        """Test selection of low energy states."""

        s0 = Subsystem("Q0", dim=3)
        s1 = Subsystem("Q1", dim=3)
        basis = DressedBasis(
            subsystems=[s0, s1], basis_vectors=np.eye(9, dtype=complex), evals=np.arange(9)
        )

        low_energy = basis.low_energy_states(cutoff_energy=5.1)

        self.assertEqual(
            low_energy.labels,
            [
                {"index": (0, 0), "eval": 0},
                {"index": (1, 0), "eval": 1},
                {"index": (2, 0), "eval": 2},
                {"index": (0, 1), "eval": 3},
                {"index": (1, 1), "eval": 4},
                {"index": (2, 1), "eval": 5},
            ],
        )

        expected_basis = np.zeros((9, 6), dtype=complex)
        expected_basis[0, 0] = 1.0
        expected_basis[1, 1] = 1.0
        expected_basis[2, 2] = 1.0
        expected_basis[3, 3] = 1.0
        expected_basis[4, 4] = 1.0
        expected_basis[5, 5] = 1.0

        self.assertAllClose(low_energy.basis_vectors, expected_basis)


@partial(test_array_backends, array_libraries=["numpy", "jax"])
class Test_sorted_eigh(QiskitDynamicsTestCase):
    """Tests for _sorted_eigh function."""

    def test_2d_case(self):
        """Test for 2x2 matrix."""

        H = self.asarray([[1.0, 0.1], [0.1, -1.0]])
        evals, evecs = _sorted_eigh(H)

        expected_evals = self.asarray([1.0049876, -1.0049876])
        expected_evecs = self.asarray([[0.99875855, -0.0498137], [0.0498137, 0.99875855]])

        self.assertAllClose(evals, expected_evals, atol=1e-7, rtol=1e-7)
        self.assertAllClose(evecs, expected_evecs, atol=1e-7, rtol=1e-7)

    def test_3d_case(self):
        """Test for 3x3 matrix."""

        H = self.asarray([[1.0, 0.0, 0.01j], [0.0, 0.0, 0.1], [-0.01j, 0.1, 0.9]])
        evals, evecs = _sorted_eigh(H)

        expected_evals = self.asarray([1.0010976, -0.0109784, 0.9098808])
        expected_evecs = self.asarray(
            [
                [0.99397135, 0.00107943j, -0.10963501j],
                [-0.01089778j, 0.9940271, 0.10858816],
                [-0.10909738j, -0.10912828, 0.9880227],
            ]
        )

        self.assertAllClose(evals, expected_evals, atol=1e-7, rtol=1e-7)
        self.assertAllClose(evecs, expected_evecs, atol=1e-7, rtol=1e-7)
