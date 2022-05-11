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

"""Tests for schrieffer_wolff.py."""

import numpy as np

from multiset import Multiset

from qiskit import QiskitError

from qiskit_dynamics.perturbation import schrieffer_wolff
from qiskit_dynamics.perturbation.schrieffer_wolff import solve_commutator_projection, commutator

from ..common import QiskitDynamicsTestCase


class Testschrieffer_wolff_validation(QiskitDynamicsTestCase):
    """Test validation checks for schrieffer_wolff."""

    @classmethod
    def setUpClass(cls):
        """Construct reusable operators."""
        cls.X = np.array(
            [
                [
                    0.0,
                    1.0,
                ],
                [1.0, 0.0],
            ]
        )
        cls.Z = np.array(
            [
                [
                    1.0,
                    0.0,
                ],
                [0.0, -1.0],
            ]
        )
        cls.P = np.array([[0.0, 1.0], [0.0, 0.0]])

    def test_H0_non_diagonal(self):
        """Test detection of non-diagonal."""

        with self.assertRaisesRegex(QiskitError, "diagonal"):
            schrieffer_wolff(self.P, perturbations=[self.Z], expansion_order=1)

    def test_H0_non_hermitian(self):
        """Test detection of diagonal but non-hermitian."""

        with self.assertRaisesRegex(QiskitError, "Hermitian"):
            schrieffer_wolff(1j * self.Z, perturbations=[self.Z], expansion_order=1)

    def test_H0_degenerate(self):
        """Test detection of degenerate H0."""

        with self.assertRaisesRegex(QiskitError, "degenerate"):
            schrieffer_wolff(np.eye(2), perturbations=[self.Z], expansion_order=1)

        # test for non-adjacent degeneracies
        with self.assertRaisesRegex(QiskitError, "degenerate"):
            schrieffer_wolff(
                np.diag([1, 0, 1]), perturbations=[np.diag([1, 0, 1])], expansion_order=1
            )

    def test_perturbation_non_hermitian(self):
        """Test perturbations being non-Hermitian."""

        with self.assertRaisesRegex(QiskitError, "Hermitian"):
            schrieffer_wolff(self.Z, perturbations=[1j * self.Z], expansion_order=1)


class Testschrieffer_wolff(QiskitDynamicsTestCase):
    """Test schrieffer_wolff function."""

    def test_simple_case(self):
        """Test a simple case with one perturbation using Pauli operators."""
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        Z = np.array([[1.0, 0.0], [0.0, -1.0]])

        H0 = Z
        H1 = X

        # manually compute SW transformation up to 5th order
        rhs1 = H1
        S1 = solve_commutator_projection(H0, rhs1)

        rhs2 = commutator(S1, 0.5 * commutator(S1, H0) + H1)
        S2 = solve_commutator_projection(H0, rhs2)

        rhs3 = (
            commutator(S2, 0.5 * commutator(S1, H0) + H1)
            + commutator(S1, 0.5 * commutator(S2, H0))
            + commutator(S1, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
        )
        S3 = solve_commutator_projection(H0, rhs3)

        rhs4 = (
            commutator(S3, 0.5 * commutator(S1, H0) + H1)
            + commutator(S2, 0.5 * commutator(S2, H0))
            + commutator(S1, 0.5 * commutator(S3, H0))
            + commutator(S2, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S2, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S1, commutator(S2, H0) / 6))
            + commutator(S1, commutator(S1, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
        )
        S4 = solve_commutator_projection(H0, rhs4)

        rhs5 = (
            commutator(S4, 0.5 * commutator(S1, H0) + H1)
            + commutator(S1, 0.5 * commutator(S4, H0))
            + commutator(S3, 0.5 * commutator(S2, H0))
            + commutator(S2, 0.5 * commutator(S3, H0))
            + commutator(S3, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S3, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S1, commutator(S3, H0) / 6))
            + commutator(S2, commutator(S2, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S2, commutator(S1, commutator(S2, H0) / 6))
            + commutator(S1, commutator(S2, commutator(S2, H0) / 6))
            + commutator(S2, commutator(S1, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
            + commutator(S1, commutator(S2, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
            + commutator(S1, commutator(S1, commutator(S2, commutator(S1, H0) / 24 + H1 / 6)))
            + commutator(S1, commutator(S1, commutator(S1, commutator(S2, H0) / 24)))
            + commutator(
                S1,
                commutator(
                    S1, commutator(S1, commutator(S1, commutator(S1, H0) / (24 * 5) + H1 / 24))
                ),
            )
        )
        S5 = solve_commutator_projection(H0, rhs5)

        expected = np.array([S1, S2, S3, S4, S5])
        output = schrieffer_wolff(H0, perturbations=[H1], expansion_order=5).array_coefficients
        self.assertAllClose(expected, output)

    def test_simple_single_variable_polynomial_case(self):
        """Test a single variable polynomial case."""
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        Y = np.array([[0.0, -1j], [1j, 0.0]])
        Z = np.array([[1.0, 0.0], [0.0, -1.0]])

        H0 = Z
        H1 = X
        H11 = Y
        H111 = (Y + X + Z) / np.sqrt(3)

        # manually compute SW transformation up to 5th order
        rhs1 = H1
        S1 = solve_commutator_projection(H0, rhs1)

        rhs2 = H11 + commutator(S1, 0.5 * commutator(S1, H0) + H1)
        S2 = solve_commutator_projection(H0, rhs2)

        rhs3 = H111 + (
            commutator(S2, 0.5 * commutator(S1, H0) + H1)
            + commutator(S1, 0.5 * commutator(S2, H0) + H11)
            + commutator(S1, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
        )
        S3 = solve_commutator_projection(H0, rhs3)

        rhs4 = (
            commutator(S3, 0.5 * commutator(S1, H0) + H1)
            + commutator(S2, 0.5 * commutator(S2, H0) + H11)
            + commutator(S1, 0.5 * commutator(S3, H0) + H111)
            + commutator(S2, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S2, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S1, commutator(S2, H0) / 6 + H11 / 2))
            + commutator(S1, commutator(S1, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
        )
        S4 = solve_commutator_projection(H0, rhs4)

        rhs5 = (
            commutator(S4, 0.5 * commutator(S1, H0) + H1)
            + commutator(S1, 0.5 * commutator(S4, H0))
            + commutator(S3, 0.5 * commutator(S2, H0) + H11)
            + commutator(S2, 0.5 * commutator(S3, H0) + H111)
            + commutator(S3, commutator(S1, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S3, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S1, commutator(S1, commutator(S3, H0) / 6 + H111 / 2))
            + commutator(S2, commutator(S2, commutator(S1, H0) / 6 + H1 / 2))
            + commutator(S2, commutator(S1, commutator(S2, H0) / 6 + H11 / 2))
            + commutator(S1, commutator(S2, commutator(S2, H0) / 6 + H11 / 2))
            + commutator(S2, commutator(S1, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
            + commutator(S1, commutator(S2, commutator(S1, commutator(S1, H0) / 24 + H1 / 6)))
            + commutator(S1, commutator(S1, commutator(S2, commutator(S1, H0) / 24 + H1 / 6)))
            + commutator(S1, commutator(S1, commutator(S1, commutator(S2, H0) / 24 + H11 / 6)))
            + commutator(
                S1,
                commutator(
                    S1, commutator(S1, commutator(S1, commutator(S1, H0) / (24 * 5) + H1 / 24))
                ),
            )
        )
        S5 = solve_commutator_projection(H0, rhs5)

        expected = np.array([S1, S2, S3, S4, S5])
        output = schrieffer_wolff(
            H0,
            perturbations=[H1, H11, H111],
            perturbation_labels=[Multiset({0: k}) for k in range(1, 4)],
            expansion_order=5,
        ).array_coefficients
        self.assertAllClose(expected, output)

    def test_random_multiple_perturbations(self):
        """Test with higher dimensional matrices and multiple perturbations."""

        rng = np.random.default_rng(30493)
        dim = 5
        b = 1.0  # bound on size of random terms

        Hd = np.diag(rng.uniform(low=-b, high=b, size=dim))

        # random hermitian ops
        rand_ops = rng.uniform(low=-b, high=b, size=(4, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(4, dim, dim)
        )
        rand_ops = 0.5 * (rand_ops + rand_ops.conj().transpose(0, 2, 1))

        H0 = rand_ops[0]
        H1 = rand_ops[1]
        H00 = rand_ops[2]
        H01 = rand_ops[3]

        # manually compute SW transformation up to 5th order
        rhs0 = H0
        S0 = solve_commutator_projection(Hd, rhs0)

        rhs1 = H1
        S1 = solve_commutator_projection(Hd, rhs1)

        rhs00 = H00 + commutator(S0, 0.5 * commutator(S0, Hd) + H0)
        S00 = solve_commutator_projection(Hd, rhs00)

        rhs01 = (
            H01
            + commutator(S0, 0.5 * commutator(S1, Hd) + H1)
            + commutator(S1, 0.5 * commutator(S0, Hd) + H0)
        )
        S01 = solve_commutator_projection(Hd, rhs01)

        rhs11 = commutator(S1, 0.5 * commutator(S1, Hd) + H1)
        S11 = solve_commutator_projection(Hd, rhs11)

        rhs000 = (
            commutator(S0, 0.5 * commutator(S00, Hd) + H00)
            + commutator(S00, 0.5 * commutator(S0, Hd) + H0)
            + commutator(S0, commutator(S0, commutator(S0, Hd) / 6 + H0 / 2))
        )
        S000 = solve_commutator_projection(Hd, rhs000)

        rhs001 = (
            commutator(S0, 0.5 * commutator(S01, Hd) + H01)
            + commutator(S1, 0.5 * commutator(S00, Hd) + H00)
            + commutator(S00, 0.5 * commutator(S1, Hd) + H1)
            + commutator(S01, 0.5 * commutator(S0, Hd) + H0)
            + commutator(S0, commutator(S0, commutator(S1, Hd) / 6 + H1 / 2))
            + commutator(S0, commutator(S1, commutator(S0, Hd) / 6 + H0 / 2))
            + commutator(S1, commutator(S0, commutator(S0, Hd) / 6 + H0 / 2))
        )
        S001 = solve_commutator_projection(Hd, rhs001)

        rhs011 = (
            commutator(S0, 0.5 * commutator(S11, Hd))
            + commutator(S1, 0.5 * commutator(S01, Hd) + H01)
            + commutator(S01, 0.5 * commutator(S1, Hd) + H1)
            + commutator(S11, 0.5 * commutator(S0, Hd) + H0)
            + commutator(S0, commutator(S1, commutator(S1, Hd) / 6 + H1 / 2))
            + commutator(S1, commutator(S0, commutator(S1, Hd) / 6 + H1 / 2))
            + commutator(S1, commutator(S1, commutator(S0, Hd) / 6 + H0 / 2))
        )
        S011 = solve_commutator_projection(Hd, rhs011)

        rhs111 = (
            commutator(S1, 0.5 * commutator(S11, Hd))
            + commutator(S11, 0.5 * commutator(S1, Hd) + H1)
            + commutator(S1, commutator(S1, commutator(S1, Hd) / 6 + H1 / 2))
        )
        S111 = solve_commutator_projection(Hd, rhs111)

        rhs0000 = (
            commutator(S0, 0.5 * commutator(S000, Hd))
            + commutator(S00, 0.5 * commutator(S00, Hd) + H00)
            + commutator(S000, 0.5 * commutator(S0, Hd) + H0)
            + commutator(S0, commutator(S0, commutator(S00, Hd) / 6 + H00 / 2))
            + commutator(S0, commutator(S00, commutator(S0, Hd) / 6 + H0 / 2))
            + commutator(S00, commutator(S0, commutator(S0, Hd) / 6 + H0 / 2))
            + commutator(S0, commutator(S0, commutator(S0, commutator(S0, Hd) / 24 + H0 / 6)))
        )
        S0000 = solve_commutator_projection(Hd, rhs0000)

        expected = np.array([S0, S1, S00, S01, S11, S000, S001, S011, S111, S0000])
        output = schrieffer_wolff(
            Hd,
            perturbations=[H0, H1, H00, H01],
            perturbation_labels=[
                Multiset({0: 1}),
                Multiset({1: 1}),
                Multiset({0: 2}),
                Multiset({0: 1, 1: 1}),
            ],
            expansion_order=3,
            expansion_labels=[Multiset({0: 4})],
        ).array_coefficients
        self.assertAllClose(expected, output)
