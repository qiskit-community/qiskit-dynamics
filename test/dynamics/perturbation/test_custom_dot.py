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

"""Tests for custom_dot.py"""

import numpy as np

from qiskit_dynamics.perturbation.custom_dot import (
    compile_custom_dot_rule,
    custom_dot,
    custom_dot_jax,
)

from ..common import QiskitDynamicsTestCase, TestJaxBase


class Testcustom_dot(QiskitDynamicsTestCase):
    """Test custom_dot."""

    def setUp(self):
        mult_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        self.compiled_rule1 = compile_custom_dot_rule(mult_rule)

        mult_rule = [(np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [0, 0], [0, 0]]))]

        self.compiled_rule2 = compile_custom_dot_rule(mult_rule)

        # for inheritance
        self.custom_dot = custom_dot

    def test_dot1(self):
        """Test correct dot1."""
        rng = np.random.default_rng(9381)
        A = rng.uniform(size=(3, 5, 5))
        B = rng.uniform(size=(3, 5, 5))

        prod02 = A[0] @ B[2]
        prod11 = A[1] @ B[1]
        prod20 = A[2] @ B[0]

        expected = np.array([prod02 + 2 * prod11 + 3 * prod20, prod02, 3 * prod11])
        output = self.custom_dot(A, B, self.compiled_rule1)

        self.assertAllClose(expected, output)

    def test_dot2(self):
        """Test correct dot2."""
        rng = np.random.default_rng(9381)
        A = rng.uniform(size=(1, 10, 10))
        B = rng.uniform(size=(3, 10, 10))

        prod02 = A[0] @ B[2]
        prod00 = A[0] @ B[0]

        expected = np.array([prod02 + 5 * prod00])
        output = self.custom_dot(A, B, self.compiled_rule2)

        self.assertAllClose(expected, output)

    def test_dot1_mult(self):
        """Test correct dot1 operating in multiplication mode."""
        rng = np.random.default_rng(9381)
        A = rng.uniform(size=(3, 5, 5))
        B = rng.uniform(size=(3, 5, 5))

        prod02 = A[0] * B[2]
        prod11 = A[1] * B[1]
        prod20 = A[2] * B[0]

        expected = np.array([prod02 + 2 * prod11 + 3 * prod20, prod02, 3 * prod11])
        output = self.custom_dot(A, B, self.compiled_rule1, "mult")

        self.assertAllClose(expected, output)

    def test_dot2_mult(self):
        """Test correct dot2 operating in multiplication mode."""
        rng = np.random.default_rng(9381)
        A = rng.uniform(size=(1, 10, 10))
        B = rng.uniform(size=(3, 10, 10))

        prod02 = A[0] * B[2]
        prod00 = A[0] * B[0]

        expected = np.array([prod02 + 5 * prod00])
        output = self.custom_dot(A, B, self.compiled_rule2, "mult")

        self.assertAllClose(expected, output)

    def test_identity(self):
        """Test inclusion of -1 indices to represent identities."""

        rule = [
            (np.array([1.0, 1.0]), np.array([[1, 0], [-1, 1]])),
            (np.array([1.0, 2.0]), np.array([[0, 1], [1, -1]])),
        ]

        compiled_rule = compile_custom_dot_rule(rule)

        rng = np.random.default_rng(2342)

        A = rng.uniform(size=(2, 10, 10))
        B = rng.uniform(size=(2, 10, 10))

        expected = [A[1] @ B[0] + B[1], A[0] @ B[1] + 2 * A[1]]
        output = self.custom_dot(A, B, compiled_rule)

        self.assertAllClose(expected, output)

    def test_padding(self):
        """Test correct evaluation of padded rules."""

        rule = [
            (np.array([1.0, 1.0]), np.array([[1, 0], [-1, 1]])),
            (np.array([1.0, 2.0]), np.array([[0, 1], [1, -1]])),
        ]

        compiled_rule = compile_custom_dot_rule(rule, unique_mult_len=5, linear_combo_len=6)

        rng = np.random.default_rng(2342)

        A = rng.uniform(size=(2, 10, 10))
        B = rng.uniform(size=(2, 10, 10))

        expected = [A[1] @ B[0] + B[1], A[0] @ B[1] + 2 * A[1]]
        output = self.custom_dot(A, B, compiled_rule)

        self.assertAllClose(expected, output)

    def test_vectorized_dot(self):
        """Test that custom_dot works for lists of matrices as well."""

        rng = np.random.default_rng(21319)
        A = rng.uniform(size=(3, 4, 5, 5))
        B = rng.uniform(size=(3, 4, 5, 5))

        prod02 = A[0] @ B[2]
        prod11 = A[1] @ B[1]
        prod20 = A[2] @ B[0]

        expected = np.array([prod02 + 2 * prod11 + 3 * prod20, prod02, 3 * prod11])
        output = self.custom_dot(A, B, self.compiled_rule1)

        self.assertAllClose(expected, output)


class Testcustom_dot_jax(Testcustom_dot, TestJaxBase):
    """Test custom_dot_jax."""

    def setUp(self):
        mult_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]
        self.compiled_rule1 = compile_custom_dot_rule(mult_rule)

        mult_rule = [(np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [0, 0], [0, 0]]))]
        self.compiled_rule2 = compile_custom_dot_rule(mult_rule)

        self.custom_dot = custom_dot_jax

    def test_dot_jit(self):
        """Verify jitting works with custom dot."""

        from jax import jit

        rng = np.random.default_rng(9381)

        A = rng.uniform(size=(3, 5, 5))
        B = rng.uniform(size=(3, 5, 5))

        prod02 = A[0] @ B[2]
        prod11 = A[1] @ B[1]
        prod20 = A[2] @ B[0]

        expected = [prod02 + 2 * prod11 + 3 * prod20, prod02, 3 * prod11]

        fast_dot = jit(lambda a, b: self.custom_dot(a, b, self.compiled_rule1))

        self.assertAllClose(expected, fast_dot(A, B))


class Testcompile_custom_dot_rule(QiskitDynamicsTestCase):
    """Tests for multiplication rule compilation."""

    def setUp(self):
        mult_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        self.compiled_rule1 = compile_custom_dot_rule(mult_rule)

        mult_rule = [(np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [0, 0], [0, 0]]))]

        self.compiled_rule2 = compile_custom_dot_rule(mult_rule)

    def test_unique_mult_pairs(self):
        """Test construction of internal unique multiplication pairs."""

        expected = np.array([[0, 2], [1, 1], [2, 0]], dtype=int)
        self.assertAllClose(expected, self.compiled_rule1[0])

        expected = np.array([[0, 2], [0, 0]], dtype=int)
        self.assertAllClose(expected, self.compiled_rule2[0])

    def test_linear_combo_rule(self):
        """Test internal linear combo rule."""

        expected_coeffs = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        expected_indices = np.array([[0, 1, 2], [0, -1, -1], [1, -1, -1]])

        coeffs, indices = self.compiled_rule1[1]
        self.assertAllClose(expected_coeffs, coeffs)
        self.assertAllClose(expected_indices, indices)

        expected_coeffs = np.array([[1.0, 2.0, 3.0]])
        expected_indices = np.array([[0, 1, 1]])

        coeffs, indices = self.compiled_rule2[1]
        self.assertAllClose(expected_coeffs, coeffs)
        self.assertAllClose(expected_indices, indices)

    def test_padding(self):
        """Test padding of compiled rule."""

        mult_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        compiled_rule = compile_custom_dot_rule(mult_rule, unique_mult_len=5, linear_combo_len=6)

        expected_unique_mults = np.array([[0, 2], [1, 1], [2, 0], [-2, -2], [-2, -2]])
        expected_coeffs = np.array(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        expected_unique_mult_indices = np.array(
            [[0, 1, 2, -1, -1, -1], [0, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1]]
        )

        self.assertAllClose(expected_unique_mults, compiled_rule[0])
        self.assertAllClose(expected_coeffs, compiled_rule[1][0])
        self.assertAllClose(expected_unique_mult_indices, compiled_rule[1][1])

    def test_index_offset(self):
        """Test index_offset argument."""
        mult_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        compiled_rule = compile_custom_dot_rule(mult_rule, index_offset=1)

        expected_unique_mults = np.array([[1, 3], [2, 2], [3, 1]])
        expected_coeffs = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        expected_unique_mult_indices = np.array([[0, 1, 2], [0, -1, -1], [1, -1, -1]])
        self.assertAllClose(expected_unique_mults, compiled_rule[0])
        self.assertAllClose(expected_coeffs, compiled_rule[1][0])
        self.assertAllClose(expected_unique_mult_indices, compiled_rule[1][1])
