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

"""Tests for orthonormal_basis.py."""

from itertools import product

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics.systems import Subsystem
from qiskit_dynamics.systems.subsystem_operators import A, Adag, I, N, X, Y, Z, SubsystemOperator
from qiskit_dynamics.systems.abstract_subsystem_operators import ScalarOperator, ZeroOperator

from ..common import QiskitDynamicsTestCase


pauliX = np.array([[0.0, 1.0], [1.0, 0.0]])
pauliY = np.array([[0.0, -1j], [1j, 0.0]])


def swap(d0, d1):
    """Generate swap operator mapping C^d0 x C^d1 -> C^d1 x C^d0."""
    W = np.zeros((d0 * d1, d0 * d1), dtype=float)

    for idx0, idx1 in product(range(d1), range(d0)):
        Eab = np.zeros((d1, d0), dtype=float)
        Eab[idx0, idx1] = 1
        W = W + np.kron(Eab, Eab.transpose())

    return W


class TestSubsystemOperator(QiskitDynamicsTestCase):
    """Tests for SubsystemOperator class."""

    def test_single_system_matrix(self):
        """Test correct construction of matrix for a single system."""

        rng = np.random.default_rng(98578373)
        mat = rng.random((10, 10)) + 1j * rng.random((10, 10))

        s0 = Subsystem("Q0", dim=10)
        s1 = Subsystem("Q1", dim=3)
        s2 = Subsystem("Q2", dim=2)

        op = SubsystemOperator(matrix=mat, subsystems=[s0])

        self.assertAllClose(op.matrix(), mat)

        self.assertAllClose(op.matrix([s0, s1]), np.kron(np.eye(3), mat))
        self.assertAllClose(op.matrix([s1, s0]), np.kron(mat, np.eye(3)))
        self.assertAllClose(op.matrix([s2, s0, s1]), np.kron(np.kron(np.eye(3), mat), np.eye(2)))

    def test_two_system_matrix(self):
        """Test correct construction of matrix with two subsystems."""

        rng = np.random.default_rng(523421)
        mat = rng.random((10, 10)) + 1j * rng.random((10, 10))

        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=5)

        op = SubsystemOperator(matrix=mat, subsystems=[s0, s1])

        self.assertAllClose(op.matrix(), mat)
        W = swap(5, 2)
        self.assertAllClose(op.matrix([s1, s0]), W @ mat @ W.conj().transpose())

        s2 = Subsystem("Q2", dim=6)
        self.assertAllClose(op.matrix([s0, s1, s2]), np.kron(np.eye(6), mat))

        W = np.kron(swap(6, 5), np.eye(2))
        expected_mat = W @ np.kron(np.eye(6), mat) @ W.conj().transpose()
        self.assertAllClose(op.matrix([s0, s2, s1]), expected_mat)

        s3 = Subsystem("Q3", dim=3)
        W = np.kron(swap(6, 5), np.eye(6))
        expected_mat = W @ np.kron(np.kron(np.eye(6), mat), np.eye(3)) @ W.conj().transpose()
        self.assertAllClose(op.matrix([s3, s0, s2, s1]), expected_mat)

        W = np.kron(swap(6, 5), swap(2, 3))
        expected_mat = W @ np.kron(np.kron(np.eye(6), mat), np.eye(3)) @ W.conj().transpose()
        self.assertAllClose(op.matrix([s0, s3, s2, s1]), expected_mat)

    def test_three_system_matrix(self):
        """Test correct construction of matrix with three subsystems."""

        rng = np.random.default_rng(643243)
        mat = rng.random((30, 30)) + 1j * rng.random((30, 30))

        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=3)
        s2 = Subsystem("Q2", dim=5)

        op = SubsystemOperator(matrix=mat, subsystems=[s0, s1, s2])

        self.assertAllClose(op.matrix(), mat)

        W = np.kron(swap(5, 3), np.eye(2))
        expected_mat = W @ mat @ W.conj().transpose()
        self.assertAllClose(op.matrix([s0, s2, s1]), expected_mat)

        W = np.kron(np.eye(3), swap(5, 2)) @ np.kron(swap(5, 3), np.eye(2))
        expected_mat = W @ mat @ W.conj().transpose()
        self.assertAllClose(op.matrix([s2, s0, s1]), expected_mat)


class TestNamedOperators(QiskitDynamicsTestCase):
    """Tests validating the output of "named" operators."""

    def testA(self):
        """Validate A."""
        op = A(Subsystem("Q0", 2))
        self.assertAllClose(op.matrix(), np.array([[0.0, 1.0], [0.0, 0.0]]))
        op = A(Subsystem("Q0", 3))
        self.assertAllClose(
            op.matrix(), np.array([[0.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2)], [0.0, 0.0, 0.0]])
        )

    def testAdag(self):
        """Validate Adag."""
        op = Adag(Subsystem("Q0", 2))
        self.assertAllClose(op.matrix(), np.array([[0.0, 0.0], [1.0, 0.0]]))
        op = Adag(Subsystem("Q0", 3))
        self.assertAllClose(
            op.matrix(), np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, np.sqrt(2), 0.0]])
        )

    def testN(self):
        """Validate N."""
        op = N(Subsystem("Q0", 2))
        self.assertAllClose(op.matrix(), np.array([[0.0, 0.0], [0.0, 1.0]]))
        op = N(Subsystem("Q0", 3))
        self.assertAllClose(
            op.matrix(), np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
        )

    def testI(self):
        """Validate I."""
        op = I(Subsystem("Q0", 2))
        self.assertAllClose(op.matrix(), np.array([[1.0, 0.0], [0.0, 1.0]]))
        op = I(Subsystem("Q0", 3))
        self.assertAllClose(
            op.matrix(), np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        )

    def testX(self):
        """Validate X."""
        op = X(Subsystem("Q0", 2))
        self.assertAllClose(op.matrix(), np.array([[0.0, 1.0], [1.0, 0.0]]))
        op = X(Subsystem("Q0", 3))
        self.assertAllClose(
            op.matrix(), np.array([[0.0, 1.0, 0.0], [1.0, 0.0, np.sqrt(2)], [0.0, np.sqrt(2), 0.0]])
        )

    def testY(self):
        """Validate Y."""
        op = Y(Subsystem("Q0", 2))
        self.assertAllClose(op.matrix(), np.array([[0.0, -1j], [1j, 0.0]]))
        op = Y(Subsystem("Q0", 3))
        self.assertAllClose(
            op.matrix(),
            np.array([[0.0, -1j, 0.0], [1j, 0.0, -1j * np.sqrt(2)], [0.0, 1j * np.sqrt(2), 0.0]]),
        )

    def testZ(self):
        """Validate Z."""
        op = Z(Subsystem("Q0", 2))
        self.assertAllClose(op.matrix(), np.array([[1.0, 0.0], [0.0, -1.0]]))
        op = Z(Subsystem("Q0", 3))
        self.assertAllClose(
            op.matrix(), np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -3.0]])
        )


class TestScalarOperators(QiskitDynamicsTestCase):
    """Tests for ScalarOperator and ZeroOperator."""

    def test_ScalarOperator(self):
        """Test correct identity construction."""

        s0 = Subsystem("Q0", dim=5)
        op = ScalarOperator(value=3.0, subsystems=[s0])

        self.assertAllClose(op.matrix([s0]), 3.0 * np.eye(5))

    def test_ZeroOperator(self):
        """Test correct identity construction."""

        s0 = Subsystem("Q0", dim=5)
        op = ZeroOperator(subsystems=[s0])

        self.assertAllClose(op.matrix([s0]), np.zeros((5, 5)))


class TestOperatorSum(QiskitDynamicsTestCase):
    """Tests for summation of operations."""

    def test_basic_sum(self):
        """Test addition of operators."""
        s0 = Subsystem("Q0", dim=2)
        op = X(s0) + Y(s0)
        self.assertAllClose(op.matrix(), np.array([[0.0, 1 - 1j], [1 + 1j, 0.0]]))
        self.assertEqual(str(op), "X(Q0) + Y(Q0)")

    def test_non_overlapping_sum(self):
        """Test addition of operators."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = X(s0) + Y(s1)
        ident = np.eye(2, dtype=complex)
        expected_op = np.kron(ident, np.array([[0.0, 1], [1, 0.0]])) + np.kron(
            np.array([[0.0, -1j], [1j, 0.0]]), ident
        )
        self.assertAllClose(op.matrix([s0, s1]), expected_op)
        self.assertEqual(str(op), "X(Q0) + Y(Q1)")

    def test_zero_operator_sum(self):
        """Test addition of an operator with the zero operator."""
        s0 = Subsystem("Q0", dim=2)
        op = X(s0)
        self.assertTrue(isinstance(op + ZeroOperator(s0), X))
        self.assertTrue(isinstance(ZeroOperator(s0) + op, X))


class TestOperatorMatmul(QiskitDynamicsTestCase):
    """Tests for matmul of operations."""

    def test_basic_matmul(self):
        """Test matmul of operators."""
        s0 = Subsystem("Q0", dim=2)
        op = X(s0) @ Y(s0)
        self.assertAllClose(op.matrix(), pauliX @ pauliY)
        self.assertEqual(str(op), "X(Q0) @ Y(Q0)")

    def test_non_overlapping_matmul(self):
        """Test matmul of operators."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = X(s0) @ Y(s1)
        expected_op = np.kron(pauliY, pauliX)
        self.assertAllClose(op.matrix([s0, s1]), expected_op)
        self.assertEqual(str(op), "X(Q0) @ Y(Q1)")

        expected_op = np.kron(pauliX, pauliY)
        self.assertAllClose(op.matrix([s1, s0]), expected_op)

    def test_zero_operator_matmul(self):
        """Test matmul of an operator with the zero operator."""
        s0 = Subsystem("Q0", dim=2)
        op = X(s0)
        self.assertTrue(isinstance(op @ ZeroOperator(s0), ZeroOperator))
        self.assertTrue(isinstance(ZeroOperator(s0) @ op, ZeroOperator))


class TestOperatorMul(QiskitDynamicsTestCase):
    """Tests for mul of operations."""

    def test_basic_mul(self):
        """Test mul of operators."""
        s0 = Subsystem("Q0", dim=2)
        op = X(s0) * Y(s0)
        self.assertAllClose(op.matrix(), pauliX * pauliY)
        self.assertEqual(str(op), "X(Q0) * Y(Q0)")

    def test_non_overlapping_mul(self):
        """Test matmul of operators."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = X(s0) * Y(s1)
        ident = np.eye(2)
        expected_op = np.kron(pauliY, ident) * np.kron(ident, pauliX)
        self.assertAllClose(op.matrix([s0, s1]), expected_op)
        self.assertEqual(str(op), "X(Q0) * Y(Q1)")

        expected_op = np.kron(pauliX, ident) * np.kron(ident, pauliY)
        self.assertAllClose(op.matrix([s1, s0]), expected_op)

    def test_zero_operator_mul(self):
        """Test matmul of an operator with the zero operator."""
        s0 = Subsystem("Q0", dim=2)
        op = X(s0)
        self.assertTrue(isinstance(op * ZeroOperator(s0), ZeroOperator))
        self.assertTrue(isinstance(ZeroOperator(s0) * op, ZeroOperator))

    def test_scalar_operator_mul(self):
        """Test multiplication of scalars with operators."""
        s0 = Subsystem("Q0", dim=2)
        op = 3 * X(s0)
        self.assertAllClose(op.matrix([s0]), 3 * pauliX)
        op = X(s0) * 3
        self.assertAllClose(op.matrix([s0]), 3 * pauliX)
        self.assertEqual(str(op), "3 * X(Q0)")


class TestFilterAndRestrict(QiskitDynamicsTestCase):
    """Tests restricting and removing subsystems."""

    def test_complete_removal(self):
        """Test removing all subsystems."""
        s0 = Subsystem("Q0", dim=2)
        op = X(s0).remove_subsystems([s0])
        self.assertTrue(isinstance(op, ZeroOperator))
        self.assertTrue(op.subsystems == [])
        with self.assertRaisesRegex(QiskitError, "no subsystems"):
            op.matrix()

    def test_sum_removal(self):
        """Test removal of operators."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = (X(s0) + Y(s1)).remove_subsystems([s0])
        self.assertAllClose(op.matrix(), pauliY)
        self.assertEqual(str(op), "Y(Q1)")

    def test_mul_removal(self):
        """Test mul of operators."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = (X(s0) * Y(s1)).remove_subsystems([s0])
        self.assertAllClose(op.matrix([s1]), np.zeros((2, 2)))
        self.assertEqual(str(op), "0.0")
        self.assertTrue(isinstance(op, ZeroOperator))

    def test_orthogonal_removal(self):
        """Test removal of subsystem that isn't in the operator."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = X(s0).remove_subsystems([s1])
        self.assertAllClose(op.matrix([s0]), pauliX)

    def test_complete_restrict(self):
        """Test filtering all subsystems."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = X(s0).restrict_subsystems([s1])
        self.assertTrue(isinstance(op, ZeroOperator))
        self.assertTrue(op.subsystems == [])
        with self.assertRaisesRegex(QiskitError, "no subsystems"):
            op.matrix()

    def test_sum_restrict(self):
        """Test removal of operators."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = (X(s0) + Y(s1)).restrict_subsystems([s0])
        self.assertAllClose(op.matrix(), pauliX)
        self.assertEqual(str(op), "X(Q0)")

    def test_mul_restrict(self):
        """Test restriction of mul of operators."""
        s0 = Subsystem("Q0", dim=2)
        s1 = Subsystem("Q1", dim=2)
        op = (X(s0) * Y(s1)).restrict_subsystems([s0])
        self.assertAllClose(op.matrix([s0]), np.zeros((2, 2)))
        self.assertEqual(str(op), "0.0")
        self.assertTrue(isinstance(op, ZeroOperator))
