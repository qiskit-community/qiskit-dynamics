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

"""Tests for custom_binary_op.py"""

import numpy as np

from qiskit_dynamics.perturbation.custom_binary_op import (
    compile_custom_operation_rule,
    CustomMatmul,
    CustomMul
)

from ..common import QiskitDynamicsTestCase, TestJaxBase

class TestCustomMatmul(QiskitDynamicsTestCase):
    """Test CustomMatmul."""

    def setUp(self):
        self.mult_rule1 = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        self.mult_rule2 = [(np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [0, 0], [0, 0]]))]

        # for inheritance
        self.binary_op = lambda A, B: A @ B
        self.CustomOpClass = CustomMatmul

    def test_rule1(self):
        """Test correct evaluation of rule1."""
        rng = np.random.default_rng(9381)
        A = rng.uniform(size=(3, 5, 5))
        B = rng.uniform(size=(3, 5, 5))

        prod02 = self.binary_op(A[0], B[2])
        prod11 = self.binary_op(A[1], B[1])
        prod20 = self.binary_op(A[2], B[0])
        expected = np.array([prod02 + 2 * prod11 + 3 * prod20, prod02, 3 * prod11])

        custom_op = self.CustomOpClass(operation_rule=self.mult_rule1,
                                     A_shape=A.shape[1:],
                                     B_shape=B.shape[1:])
        output = custom_op(A, B)

        self.assertAllClose(expected, output)

    def test_rule2(self):
        """Test correct evaluation of rule 2."""
        rng = np.random.default_rng(9381)
        A = rng.uniform(size=(1, 10, 10))
        B = rng.uniform(size=(3, 10, 10))

        prod02 = self.binary_op(A[0], B[2])
        prod00 = self.binary_op(A[0], B[0])
        expected = np.array([prod02 + 5 * prod00])

        custom_op = self.CustomOpClass(operation_rule=self.mult_rule2,
                                     A_shape=A.shape[1:],
                                     B_shape=B.shape[1:])
        output = custom_op(A, B)

        self.assertAllClose(expected, output)

    def test_vectorized_dot(self):
        """Test that custom_dot works for lists of matrices as well."""

        rng = np.random.default_rng(21319)
        A = rng.uniform(size=(3, 4, 5, 5))
        B = rng.uniform(size=(3, 4, 5, 5))

        prod02 = self.binary_op(A[0], B[2])
        prod11 = self.binary_op(A[1], B[1])
        prod20 = self.binary_op(A[2], B[0])
        expected = np.array([prod02 + 2 * prod11 + 3 * prod20, prod02, 3 * prod11])

        custom_op = self.CustomOpClass(operation_rule=self.mult_rule1,
                                     A_shape=A.shape[1:],
                                     B_shape=B.shape[1:])
        output = custom_op(A, B)

        self.assertAllClose(expected, output)


class TestCustomMatmulJAX(TestCustomMatmul, TestJaxBase):

    def test_jit_grad(self):
        """Verify jitting and gradding works through CustomMatmul."""

        from jax import jit, grad

        def func(A, B):
            custom_matmul = CustomMatmul(self.mult_rule1, A.shape[1:], B.shape[1:])
            return custom_matmul(A, B).real.sum()

        jit_grad_func = jit(grad(func))

        rng = np.random.default_rng(9381)

        A = np.random.rand(3, 5, 5)
        B = np.random.rand(3, 5, 5)

        jit_grad_func(A, B)


class TestCustomMul(TestCustomMatmul):

    def setUp(self):
        super().setUp()
        # for inheritance
        self.binary_op = lambda A, B: A * B
        self.CustomOpClass = CustomMul


class TestCustomMulJax(TestCustomMul, TestJaxBase):

    def test_jit_grad(self):
        """Verify jitting and gradding works through CustomMul."""

        from jax import jit, grad

        def func(A, B):
            custom_mul = CustomMul(self.mult_rule1, A.shape[1:], B.shape[1:])
            return custom_mul(A, B).real.sum()

        jit_grad_func = jit(grad(func))

        rng = np.random.default_rng(9381)

        A = np.random.rand(3, 5, 5)
        B = np.random.rand(3, 5, 5)

        jit_grad_func(A, B)


class Testcompile_custom_operation_rule(QiskitDynamicsTestCase):
    """Tests for custom operation rule compilation."""

    def setUp(self):
        operation_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        self.compiled_rule1 = compile_custom_operation_rule(operation_rule)

        operation_rule = [(np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [0, 0], [0, 0]]))]

        self.compiled_rule2 = compile_custom_operation_rule(operation_rule)

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

        operation_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        compiled_rule = compile_custom_operation_rule(operation_rule, unique_evaluation_len=5, linear_combo_len=6)

        expected_unique_mults = np.array([[0, 2], [1, 1], [2, 0], [-1, -1], [-1, -1]])
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
        operation_rule = [
            (np.array([1.0, 2.0, 3.0]), np.array([[0, 2], [1, 1], [2, 0]])),
            (np.array([1.0]), np.array([[0, 2]])),
            (np.array([3.0]), np.array([[1, 1]])),
        ]

        compiled_rule = compile_custom_operation_rule(operation_rule, index_offset=1)

        expected_unique_mults = np.array([[1, 3], [2, 2], [3, 1]])
        expected_coeffs = np.array([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        expected_unique_mult_indices = np.array([[0, 1, 2], [0, -1, -1], [1, -1, -1]])
        self.assertAllClose(expected_unique_mults, compiled_rule[0])
        self.assertAllClose(expected_coeffs, compiled_rule[1][0])
        self.assertAllClose(expected_unique_mult_indices, compiled_rule[1][1])
