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

"""Tests for functions in ArrayPolynomial."""

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics.perturbation import Multiset, ArrayPolynomial

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit
    from jax import grad, jit
except ImportError:
    pass


class TestArrayPolynomial(QiskitDynamicsTestCase):
    """Test the ArrayPolynomial class."""

    def test_validation_error_no_ops(self):
        """Test validation error when no information specified."""

        with self.assertRaisesRegex(QiskitError, "At least one"):
            ArrayPolynomial()

    def test_only_constant_term(self):
        """Test constant term."""

        poly = ArrayPolynomial(constant_term=3.)
        self.assertAllClose(poly(), 3.)

    def test_call_simple_case(self):
        """Typical expected usage case."""

        rng = np.random.default_rng(18471)
        coeffs = rng.uniform(low=-1, high=1, size=(5, 10, 10))
        monomial_multisets = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
        ]

        ap = ArrayPolynomial(coeffs, monomial_multisets)

        c = np.array([3.0, 4.0])
        output = ap(c)
        expected = (
            c[0] * coeffs[0]
            + c[1] * coeffs[1]
            + c[0] * c[0] * coeffs[2]
            + c[0] * c[1] * coeffs[3]
            + c[1] * c[1] * coeffs[4]
        )
        self.assertAllClose(expected, output)

        c = np.array([3.2123, 4.1])
        output = ap(c)
        expected = (
            c[0] * coeffs[0]
            + c[1] * coeffs[1]
            + c[0] * c[0] * coeffs[2]
            + c[0] * c[1] * coeffs[3]
            + c[1] * c[1] * coeffs[4]
        )
        self.assertAllClose(expected, output)

    def test_compute_monomials_simple_case(self):
        """Simple test case for compute_monomials."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
            Multiset({0: 3}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(938122)
        c = rng.uniform(size=(2,))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [c[0], c[1], c[0] * c[0], c[0] * c[1], c[1] * c[1], c[0] * c[0] * c[0]]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_skipped_variable(self):
        """Test compute monomials case with skipped variable."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 2: 1}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(22321)
        c = rng.uniform(size=(3,))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[2],
                c[0] * c[0],
                c[0] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_medium_case(self):
        """Test compute_monomials medium complexity test case."""
        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 1: 1}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({2: 3}),
            Multiset({0: 3, 1: 1}),
            Multiset({2: 4}),
        ]

        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3,))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[1],
                c[2],
                c[0] * c[0],
                c[0] * c[1],
                c[0] * c[2],
                c[1] * c[1],
                c[1] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[1],
                c[0] * c[1] * c[2],
                c[2] * c[2] * c[2],
                c[0] * c[0] * c[0] * c[1],
                c[2] * c[2] * c[2] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_vectorized(self):
        """Test vectorized evaluation."""
        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 1: 1}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({2: 3}),
            Multiset({0: 3, 1: 1}),
            Multiset({2: 4}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3, 20))

        ap = ArrayPolynomial(coeffs, multiset_list)

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[1],
                c[2],
                c[0] * c[0],
                c[0] * c[1],
                c[0] * c[2],
                c[1] * c[1],
                c[1] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[1],
                c[0] * c[1] * c[2],
                c[2] * c[2] * c[2],
                c[0] * c[0] * c[0] * c[1],
                c[2] * c[2] * c[2] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_only_first_order_terms(self):
        """Test a case with only first order terms."""

        multiset_list = [Multiset({0: 1}), Multiset({1: 1})]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        ap = ArrayPolynomial(coeffs, multiset_list)

        c = np.array([3.0, 2.0])
        self.assertAllClose(ap.compute_monomials(c), c)

    def test_compute_monomials_incomplete_list(self):
        """Test case where the multiset_list is unordered and incomplete."""

        multiset_list = [Multiset({2: 2}), Multiset({0: 1}), Multiset({1: 1, 2: 1})]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        ap = ArrayPolynomial(coeffs, multiset_list)

        c = np.array([3.0, 2.0, 4.0])
        self.assertAllClose(ap.compute_monomials(c), np.array([16.0, 3.0, 8.0]))


class TestArrayPolynomialJax(TestArrayPolynomial, TestJaxBase):
    """JAX version of TestArrayPolynomial."""

    def test_jit_compute_monomials(self):
        """Test jitting works."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
            Multiset({0: 3}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = ArrayPolynomial(coeffs, multiset_list)

        monomial_function_jit = jit(mp.compute_monomials)

        rng = np.random.default_rng(4122)
        c = rng.uniform(size=(2,))

        self.assertAllClose(mp.compute_monomials(c), monomial_function_jit(c))

    def test_compute_monomials_grad(self):
        """Test grad works."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = ArrayPolynomial(coeffs, multiset_list)

        monomial_function_jit_grad = jit(grad(lambda c: mp.compute_monomials(c).sum()))

        c = np.array([2.0, 3.0])
        expected = np.array([1.0 + 0.0 + 4.0 + 3.0 + 0.0, 0.0 + 1.0 + 0.0 + 2.0 + 6.0])

        self.assertAllClose(expected, monomial_function_jit_grad(c))
