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

"""Tests for functions in power_series_utils.py."""

import numpy as np

from qiskit_dynamics.perturbation.power_series_utils import (
    MatrixPolynomial,
    get_monomial_compute_function,
    get_monomial_compute_function_jax,
    clean_index_multisets,
    get_complete_index_multisets,
    submultisets_and_complements,
    is_submultiset,
    multiset_complement,
    submultiset_filter
)

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit
    from jax import grad, jit
except ImportError:
    pass


class TestMatrixPolynomial(QiskitDynamicsTestCase):
    """Test the MatrixPolynomial class."""

    def test_call_simple_case(self):
        """Typical expected usage case."""

        rng = np.random.default_rng(18471)
        coeffs = rng.uniform(low=-1, high=1, size=(5, 10, 10))
        monomial_multisets = [[0], [1], [0, 0], [0, 1], [1, 1]]

        mp = MatrixPolynomial(coeffs, monomial_multisets)

        c = np.array([3., 4.])
        output = mp(c)
        expected = (c[0] * coeffs[0] + c[1] * coeffs[1] + c[0] * c[0] * coeffs[2]
                    + c[0] * c[1] * coeffs[3] + c[1] * c[1] * coeffs[4])
        self.assertAllClose(expected, output)

        c = np.array([3.2123, 4.1])
        output = mp(c)
        expected = (c[0] * coeffs[0] + c[1] * coeffs[1] + c[0] * c[0] * coeffs[2]
                    + c[0] * c[1] * coeffs[3] + c[1] * c[1] * coeffs[4])
        self.assertAllClose(expected, output)

    def test_compute_monomials_simple_case(self):
        """Simple test case for compute_monomials."""

        multiset_list = [[0], [1], [0, 0], [0, 1], [1, 1], [0, 0, 0]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(938122)
        c = rng.uniform(size=(2,))

        mp = MatrixPolynomial(coeffs, multiset_list)

        output_monomials = mp.compute_monomials(c)
        expected_monomials = np.array([c[0], c[1], c[0] * c[0],
                                       c[0] * c[1], c[1] * c[1], c[0] * c[0] * c[0]])
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_skipped_variable(self):
        """Test compute monomials case with skipped variable."""

        multiset_list = [[0], [2], [0, 0], [0, 2], [2, 2], [0, 0, 0], [0, 0, 2]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(22321)
        c = rng.uniform(size=(3,))

        mp = MatrixPolynomial(coeffs, multiset_list)

        output_monomials = mp.compute_monomials(c)
        expected_monomials = np.array([c[0], c[2], c[0] * c[0], c[0] * c[2], c[2] * c[2],
                                       c[0] * c[0] * c[0], c[0] * c[0] * c[2]])
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_medium_case(self):
        """Test compute_monomials medium complexity test case."""
        multiset_list = [[0], [1], [2],
                         [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2],
                         [0, 0, 0], [0, 0, 1], [0, 1, 2], [2, 2, 2],
                         [0, 0, 0, 1], [2, 2, 2, 2]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3,))

        mp = MatrixPolynomial(coeffs, multiset_list)

        output_monomials = mp.compute_monomials(c)
        expected_monomials = np.array([c[0], c[1], c[2],
                                       c[0] * c[0], c[0] * c[1], c[0] * c[2], c[1] * c[1],
                                       c[1] * c[2], c[2] * c[2],
                                       c[0] * c[0] * c[0], c[0] * c[0] * c[1], c[0] * c[1] * c[2],
                                       c[2] * c[2] * c[2],
                                       c[0] * c[0] * c[0] * c[1], c[2] * c[2] * c[2] * c[2]])
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_vectorized(self):
        """Test vectorized evaluation."""

        multiset_list = [[0], [1], [2],
                         [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2],
                         [0, 0, 0], [0, 0, 1], [0, 1, 2], [2, 2, 2],
                         [0, 0, 0, 1], [2, 2, 2, 2]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3, 20))

        mp = MatrixPolynomial(coeffs, multiset_list)

        output_monomials = mp.compute_monomials(c)
        expected_monomials = np.array([c[0], c[1], c[2],
                                       c[0] * c[0], c[0] * c[1], c[0] * c[2], c[1] * c[1],
                                       c[1] * c[2], c[2] * c[2],
                                       c[0] * c[0] * c[0], c[0] * c[0] * c[1], c[0] * c[1] * c[2],
                                       c[2] * c[2] * c[2],
                                       c[0] * c[0] * c[0] * c[1], c[2] * c[2] * c[2] * c[2]])
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_only_first_order_terms(self):
        """Test a case with only first order terms."""

        multiset_list = [[0], [1]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = MatrixPolynomial(coeffs, multiset_list)

        c = np.array([3., 2.])
        self.assertAllClose(mp.compute_monomials(c), c)

    def test_compute_monomials_incomplete_list(self):
        """Test case where the multiset_list is unordered and incomplete."""

        multiset_list = [[2, 2], [0], [1, 2]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = MatrixPolynomial(coeffs, multiset_list)

        c = np.array([3., 2., 4.])
        self.assertAllClose(mp.compute_monomials(c), np.array([16., 3., 8.]))


class TestMatrixPolynomialJax(TestMatrixPolynomial, TestJaxBase):
    """JAX version of TestMatrixPolynomial."""

    def test_jit_compute_monomials(self):
        """Test jitting works."""

        multiset_list = [[0], [1], [0, 0], [0, 1], [1, 1], [0, 0, 0]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = MatrixPolynomial(coeffs, multiset_list)

        monomial_function_jit = jit(mp.compute_monomials)

        rng = np.random.default_rng(4122)
        c = rng.uniform(size=(2,))

        self.assertAllClose(mp.compute_monomials(c), monomial_function_jit(c))

    def test_compute_monomials_grad(self):
        """Test grad works."""

        multiset_list = [[0], [1], [0, 0], [0, 1], [1, 1]]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = MatrixPolynomial(coeffs, multiset_list)

        monomial_function_jit_grad = jit(grad(lambda c: mp.compute_monomials(c).sum()))

        c = np.array([2., 3.])
        expected = np.array([1. + 0. + 4. + 3. + 0., 0. + 1. + 0. + 2. + 6.])

        self.assertAllClose(expected, monomial_function_jit_grad(c))


class TestMultisetIndexConstruction(QiskitDynamicsTestCase):
    """Test cases for construction of index multisets.
    """

    def test_clean_index_multisets(self):
        """Test clean_index_multisets."""
        output = clean_index_multisets([[0], [1, 0], [0, 1, 0], [0, 0, 1]])
        expected = [[0], [0, 1], [0, 0, 1]]
        self.assertTrue(output == expected)

        output = clean_index_multisets([[1, 2, 3], [0, 1], [3, 2, 1], [0]])
        expected = [[1, 2, 3], [0, 1], [0]]
        self.assertTrue(output == expected)

    def test_get_complete_index_multisets_terms_case1(self):
        """Test get_complete_index_multisets case 1."""

        dyson_terms = [[2, 2, 0, 1], [1, 2]]
        expected = [
            [0],
            [1],
            [2],
            [0, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [0, 1, 2],
            [0, 2, 2],
            [1, 2, 2],
            [0, 1, 2, 2],
        ]

        self._test_get_complete_index_multisets(dyson_terms, expected)

    def test_get_complete_index_multisets_case2(self):
        """Test get_complete_index_multisets case 2."""

        dyson_terms = [[2, 2, 0, 3], [1, 2], [0], [0, 2, 2, 3]]
        expected = [
            [0],
            [1],
            [2],
            [3],
            [0, 2],
            [0, 3],
            [1, 2],
            [2, 2],
            [2, 3],
            [0, 2, 2],
            [0, 2, 3],
            [2, 2, 3],
            [0, 2, 2, 3],
        ]

        self._test_get_complete_index_multisets(dyson_terms, expected)

    def test_get_complete_index_multisets_case3(self):
        """Test get_complete_index_multisets case 3."""

        dyson_terms = [[0, 2, 1, 1, 3]]
        expected = [
            [0],
            [1],
            [2],
            [3],
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 1],
            [1, 2],
            [1, 3],
            [2, 3],
            [0, 1, 1],
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 1, 2],
            [1, 1, 3],
            [1, 2, 3],
            [0, 1, 1, 2],
            [0, 1, 1, 3],
            [0, 1, 2, 3],
            [1, 1, 2, 3],
            [0, 1, 1, 2, 3],
        ]

        self._test_get_complete_index_multisets(dyson_terms, expected)

    def _test_get_complete_index_multisets(self, symmetric_indices, expected):
        """Run test case for _full_ordered_dyson_term_list."""
        output = get_complete_index_multisets(symmetric_indices)

        self.assertListEquality(expected, output)
        self.assertIncreasingLen(output)

    def assertIncreasingLen(self, term_list):
        """Assert [len(x) for x in term_list] is increasing."""
        self.assertTrue(np.all(np.diff([len(x) for x in term_list]) >= 0))

    def assertListEquality(self, list_a, list_b):
        """Assert two lists have the same elements."""
        self.assertTrue(len(list_a) == len(list_b))

        for item in list_a:
            self.assertTrue(item in list_b)


class TestMultisetFunctions(QiskitDynamicsTestCase):
    """Test cases for multiset helper fucntions."""

    def test_submultisets_and_complements_full_cases(self):
        """Test a few simple cases."""

        multiset = [0, 0, 0]
        subsets, complements = submultisets_and_complements(multiset)
        expected_subsets, expected_complements = [[0], [0, 0]], [[0, 0], [0]]
        self.assertStrictListEquality(subsets, expected_subsets)
        self.assertStrictListEquality(complements, expected_complements)

        multiset = [0, 0, 1]
        subsets, complements = submultisets_and_complements(multiset)
        expected_subsets = [[0], [1], [0, 0], [0, 1]]
        expected_complements = [[0, 1], [0, 0], [1], [0]]
        self.assertStrictListEquality(subsets, expected_subsets)
        self.assertStrictListEquality(complements, expected_complements)

        multiset = [0, 1, 2]
        subsets, complements = submultisets_and_complements(multiset)
        expected_subsets = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
        expected_complements = [[1, 2], [0, 2], [0, 1], [2], [1], [0]]
        self.assertStrictListEquality(subsets, expected_subsets)
        self.assertStrictListEquality(complements, expected_complements)

    def test_submultisets_and_complements_bound_cases1(self):
        """Test cases with size bound."""

        multiset = [0, 0, 1, 1, 2]
        subsets, complements = submultisets_and_complements(multiset, 3)
        expected_subsets = [[0], [1], [2],
                            [0, 0], [0, 1], [0, 2], [1, 1], [1, 2]]
        expected_complements = [[0, 1, 1, 2], [0, 0, 1, 2], [0, 0, 1, 1],
                                [1, 1, 2], [0, 1, 2], [0, 1, 1], [0, 0, 2], [0, 0, 1]]
        self.assertStrictListEquality(subsets, expected_subsets)
        self.assertStrictListEquality(complements, expected_complements)

    def test_submultisets_and_complements_bound_cases2(self):
        """Test cases where subsets are of size 1."""

        l_terms, r_terms = submultisets_and_complements([0, 0, 2, 2], 2)
        expected_l_terms = [[0], [2]]
        expected_r_terms = [[0, 2, 2], [0, 0, 2]]
        self.assertStrictListEquality(l_terms, expected_l_terms)
        self.assertStrictListEquality(r_terms, expected_r_terms)

        l_terms, r_terms = submultisets_and_complements([0, 0, 1, 2], 2)
        expected_l_terms = [[0], [1], [2]]
        expected_r_terms = [[0, 1, 2], [0, 0, 2], [0, 0, 1]]
        self.assertStrictListEquality(l_terms, expected_l_terms)
        self.assertStrictListEquality(r_terms, expected_r_terms)

    def test_is_submultiset(self):
        """Test is_submultiset utility function."""
        B = [0, 0, 1]
        self.assertFalse(is_submultiset([2], B))
        self.assertFalse(is_submultiset([0, 0, 0], B))
        self.assertTrue(is_submultiset([0], B))
        self.assertTrue(is_submultiset([0, 1], B))
        self.assertTrue(is_submultiset([0, 0], B))
        self.assertTrue(is_submultiset(B, B))

    def test_multiset_complement(self):
        """Test multiset_complement utility function."""
        B = [0, 0, 1, 2, 2]
        self.assertTrue(multiset_complement([0], B) == [0, 1, 2, 2])
        self.assertTrue(multiset_complement([0, 0, 0], B) == [1, 2, 2])
        self.assertTrue(multiset_complement([0, 1, 2], B) == [0, 2])
        self.assertTrue(multiset_complement([0, 2, 1], B) == [0, 2])
        self.assertTrue(multiset_complement([3], B) == B)

    def test_submultiset_filter(self):
        """Test submultiset_filter utility function."""

        multiset_list = [[0, 0, 1], [0, 0, 2]]
        self.assertTrue(submultiset_filter([[0], [1], [0, 2]], multiset_list) == [[0], [1], [0, 2]])
        self.assertTrue(submultiset_filter([[0, 0], [1], [0, 2]], multiset_list) == [[0, 0], [1], [0, 2]])
        self.assertTrue(submultiset_filter([[0, 0, 0], [1], [0, 2]], multiset_list) == [[1], [0, 2]])


    def assertStrictListEquality(self, list_a, list_b):
        """Assert two lists are exactly the same."""

        self.assertTrue(len(list_a) == len(list_b))
        for item_a, item_b in zip(list_a, list_b):
            self.assertTrue(item_a == item_b)
