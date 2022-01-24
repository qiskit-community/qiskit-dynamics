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

"""Tests Multiset."""

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics.perturbation import Multiset
from qiskit_dynamics.perturbation.multiset import submultiset_filter

from ..common import QiskitDynamicsTestCase


class TestMultiset(QiskitDynamicsTestCase):
    """Tests for Multiset class."""

    def test_validation(self):
        """Test that non-integer types, and negative counts, raise errors."""

        with self.assertRaises(QiskitError) as qe:
            Multiset({'a': 1})
        self.assertTrue('must be integers' in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            Multiset({0: 1, 1: -1})
        self.assertTrue('non-negative' in str(qe.exception))

    def test_eq(self):
        """Test __eq__."""

        self.assertTrue(Multiset({0: 2, 1: 1}) == Multiset({1: 1, 0: 2}))
        self.assertFalse(Multiset({0: 2, 1: 1}) == Multiset({1: 2}))

    def test_issubmultiset(self):
        """Test issubmultiset method."""
        B = Multiset({0: 2, 1: 1})
        self.assertFalse(Multiset({2: 1}) <= B)
        self.assertFalse(Multiset({0: 3}) <= B)
        self.assertTrue(B >= Multiset({0: 1}))
        self.assertTrue(Multiset({0: 1, 1: 1}) <= B)
        self.assertTrue(Multiset({0: 2}) <= B)
        self.assertTrue(B <= B)
        self.assertTrue(B >= B)

    def test_difference(self):
        """Test difference method."""
        B = Multiset({0: 2, 1: 1, 2: 2})

        self.assertTrue((B - Multiset({0: 1})) == Multiset({0: 1, 1: 1, 2: 2}))
        self.assertTrue((B - Multiset({0: 3})) == Multiset({1:1, 2: 2}))
        self.assertTrue((B - Multiset({0: 1, 1: 1, 2: 1})) == Multiset({0:1, 2: 1}))
        self.assertTrue((B - Multiset({0: 1, 2: 1, 1: 1})) == Multiset({0:1, 2: 1}))
        self.assertTrue((B - Multiset({3: 1})) == B)

    def test_submultisets_and_complements_full_cases(self):
        """Test a few simple cases."""

        multiset = Multiset({0: 3})
        subsets, complements = multiset.submultisets_and_complements()
        expected_subsets, expected_complements = [Multiset({0: 1}), Multiset({0: 2})], [Multiset({0: 2}), Multiset({0: 1})]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

        multiset = Multiset({0: 2, 1: 1})
        subsets, complements = multiset.submultisets_and_complements()
        expected_subsets = [Multiset({0: 1}), Multiset({1: 1}), Multiset({0: 2}), Multiset({0: 1, 1: 1})]
        expected_complements = [Multiset({0: 1, 1: 1}), Multiset({0: 2}), Multiset({1: 1}), Multiset({0: 1})]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

        multiset = Multiset({0: 1, 1: 1, 2: 1})
        subsets, complements = multiset.submultisets_and_complements()
        expected_subsets = [Multiset({0: 1}), Multiset({1: 1}), Multiset({2: 1}), Multiset({0: 1, 1: 1}), Multiset({0: 1, 2: 1}), Multiset({1: 1, 2: 1})]
        expected_complements = [Multiset({1: 1, 2: 1}), Multiset({0: 1, 2: 1}), Multiset({0: 1, 1: 1}), Multiset({2: 1}), Multiset({1: 1}), Multiset({0: 1})]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

    def test_submultisets_and_complements_bound_cases1(self):
        """Test cases with size bound."""

        multiset = Multiset({0: 2, 1: 2, 2: 1})
        subsets, complements = multiset.submultisets_and_complements(3)
        expected_subsets = [Multiset({0: 1}), Multiset({1: 1}), Multiset({2: 1}),
                            Multiset({0: 2}), Multiset({0: 1, 1: 1}), Multiset({0: 1, 2: 1}),
                            Multiset({1: 2}), Multiset({1: 1, 2: 1})]
        expected_complements = [Multiset({0: 1, 1: 2, 2: 1}),
                                Multiset({0: 2, 1: 1, 2: 1}),
                                Multiset({0: 2, 1: 2}),
                                Multiset({1: 2, 2: 1}),
                                Multiset({0: 1, 1: 1, 2: 1}),
                                Multiset({0: 1, 1: 2}),
                                Multiset({0: 2, 2: 1}),
                                Multiset({0: 2, 1: 1})]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

    def test_submultisets_and_complements_bound_cases2(self):
        """Test cases where subsets are of size 1."""

        multiset = Multiset({0: 2, 2: 2})
        l_terms, r_terms = multiset.submultisets_and_complements(2)
        expected_l_terms = [Multiset({0: 1}), Multiset({2: 1})]
        expected_r_terms = [Multiset({0: 1, 2: 2}), Multiset({0: 2, 2: 1})]
        self.assertTrue(l_terms == expected_l_terms)
        self.assertTrue(r_terms == expected_r_terms)

        multiset = Multiset({0: 2, 1: 1, 2: 1})
        l_terms, r_terms = multiset.submultisets_and_complements(2)
        expected_l_terms = [Multiset({0: 1}), Multiset({1: 1}), Multiset({2: 1})]
        expected_r_terms = [Multiset({0: 1, 1: 1, 2: 1}), Multiset({0: 2, 2: 1}), Multiset({0: 2, 1: 1})]
        self.assertTrue(l_terms == expected_l_terms)
        self.assertTrue(r_terms == expected_r_terms)


class TestMultisetFunctions(QiskitDynamicsTestCase):
    """Test cases for multiset helper functions."""

    def test_submultiset_filter(self):
        """Test submultiset_filter utility function."""

        multiset_list = [Multiset({0: 2, 1:1}), Multiset({0: 2, 2:1})]

        multiset_candidates = [Multiset({0: 1}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates)

        multiset_candidates = [Multiset({0: 2}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates)

        multiset_candidates = [Multiset({0: 3}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(
            submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates[1:]
        )


'''
class TestMultisetIndexConstruction(QiskitDynamicsTestCase):
    """Test cases for construction of index multisets."""

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






    def assertStrictListEquality(self, list_a, list_b):
        """Assert two lists are exactly the same."""

        self.assertTrue(len(list_a) == len(list_b))
        for item_a, item_b in zip(list_a, list_b):
            self.assertTrue(item_a == item_b)
'''
