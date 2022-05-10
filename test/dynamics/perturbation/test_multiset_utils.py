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

"""Test multiset_utils functions."""

from multiset import Multiset

from qiskit import QiskitError

from qiskit_dynamics.perturbation.multiset_utils import (
    validate_non_negative_ints,
    multiset_to_sorted_list,
    sorted_multisets,
    clean_multisets,
    submultiset_filter,
    submultisets_and_complements,
    get_all_submultisets
)

from ..common import QiskitDynamicsTestCase


class TestValidation(QiskitDynamicsTestCase):
    """Test validation of multisets."""

    def test_non_int(self):
        """Test that a non-integer entry raises an error."""

        with self.assertRaises(QiskitError) as qe:
            validate_non_negative_ints(Multiset("abc"))
        self.assertTrue("non-negative integers" in str(qe.exception))

    def test_negative_int(self):
        """Test that a negative integer raises an error."""

        with self.assertRaises(QiskitError) as qe:
            validate_non_negative_ints(Multiset([1, 2, -1]))
        self.assertTrue("non-negative integers" in str(qe.exception))


class TestSortedMultisets(QiskitDynamicsTestCase):
    """Test sorted_multisets function."""

    def test_case1(self):
        """Simple test case."""

        multisets = [Multiset([0, 0, 1]), Multiset([1, 1, 0]), Multiset([0, 2]), Multiset([1])]
        output = sorted_multisets(multisets)
        expected = [Multiset([1]), Multiset([0, 2]), Multiset([0, 0, 1]), Multiset([0, 1, 1])]
        self.assertTrue(output == expected)


class TestToSortedList(QiskitDynamicsTestCase):
    """Test conversion to sorted list."""

    def test_to_sorted_list(self):
        """Test correct conversion to sorted list."""

        ms = Multiset([0, 2, 4, 1, 2, 5, 0, 2])
        self.assertTrue([0, 0, 1, 2, 2, 2, 4, 5] == multiset_to_sorted_list(ms))

        ms = Multiset({1: 3, 4: 2, 0: 1})
        self.assertTrue([0, 1, 1, 1, 4, 4] == multiset_to_sorted_list(ms))


class TestCleanMultisets(QiskitDynamicsTestCase):
    """Test clean_multisets function for removing duplicates and sorting."""

    def test_clean_multisets(self):
        """Test clean_multisets."""
        output = clean_multisets(
            [
                Multiset({0: 1}),
                Multiset({0: 1, 1: 1}),
                Multiset({0: 2, 1: 1}),
                Multiset({0: 2, 1: 1}),
            ]
        )
        expected = [Multiset({0: 1}), Multiset({0: 1, 1: 1}), Multiset({0: 2, 1: 1})]
        self.assertTrue(output == expected)

        output = clean_multisets(
            [
                Multiset({1: 1, 2: 1, 3: 1}),
                Multiset({0: 1, 1: 1}),
                Multiset({1: 1, 2: 1, 3: 1}),
                Multiset({0: 1}),
            ]
        )
        expected = [Multiset({0: 1}), Multiset({0: 1, 1: 1}), Multiset({1: 1, 2: 1, 3: 1})]
        self.assertTrue(output == expected)


class TestSubmultisetFilter(QiskitDynamicsTestCase):
    """Test submultiset_filter function."""

    def test_submultiset_filter(self):
        """Test submultiset_filter utility function."""

        multiset_list = [Multiset({0: 2, 1: 1}), Multiset({0: 2, 2: 1})]

        multiset_candidates = [Multiset({0: 1}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(
            submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates
        )

        multiset_candidates = [Multiset({0: 2}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(
            submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates
        )

        multiset_candidates = [Multiset({0: 3}), Multiset({1: 1}), Multiset({0: 1, 2: 1})]
        self.assertTrue(
            submultiset_filter(multiset_candidates, multiset_list) == multiset_candidates[1:]
        )

class TestSubmultisetsAndComplements(QiskitDynamicsTestCase):
    """Test submultisets_and_complements function."""

    def test_submultisets_and_complements_full_cases(self):
        """Test a few simple cases."""

        multiset = Multiset({0: 3})
        subsets, complements = submultisets_and_complements(multiset)
        expected_subsets, expected_complements = [Multiset({0: 1}), Multiset({0: 2})], [
            Multiset({0: 2}),
            Multiset({0: 1}),
        ]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

        multiset = Multiset({0: 2, 1: 1})
        subsets, complements = submultisets_and_complements(multiset)
        expected_subsets = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
        ]
        expected_complements = [
            Multiset({0: 1, 1: 1}),
            Multiset({0: 2}),
            Multiset({1: 1}),
            Multiset({0: 1}),
        ]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

        multiset = Multiset({0: 1, 1: 1, 2: 1})
        subsets, complements = submultisets_and_complements(multiset)
        expected_subsets = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 1, 2: 1}),
        ]
        expected_complements = [
            Multiset({1: 1, 2: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({0: 1, 1: 1}),
            Multiset({2: 1}),
            Multiset({1: 1}),
            Multiset({0: 1}),
        ]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

    def test_submultisets_and_complements_bound_cases1(self):
        """Test cases with size bound."""

        multiset = Multiset({0: 2, 1: 2, 2: 1})
        subsets, complements = submultisets_and_complements(multiset, 3)
        expected_subsets = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
        ]
        expected_complements = [
            Multiset({0: 1, 1: 2, 2: 1}),
            Multiset({0: 2, 1: 1, 2: 1}),
            Multiset({0: 2, 1: 2}),
            Multiset({1: 2, 2: 1}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({0: 1, 1: 2}),
            Multiset({0: 2, 2: 1}),
            Multiset({0: 2, 1: 1}),
        ]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

    def test_submultisets_and_complements_bound_cases2(self):
        """Test cases where subsets are of size 1."""

        multiset = Multiset({0: 2, 2: 2})
        l_terms, r_terms = submultisets_and_complements(multiset, 2)
        expected_l_terms = [Multiset({0: 1}), Multiset({2: 1})]
        expected_r_terms = [Multiset({0: 1, 2: 2}), Multiset({0: 2, 2: 1})]
        self.assertTrue(l_terms == expected_l_terms)
        self.assertTrue(r_terms == expected_r_terms)

        multiset = Multiset({0: 2, 1: 1, 2: 1})
        l_terms, r_terms = submultisets_and_complements(multiset, 2)
        expected_l_terms = [Multiset({0: 1}), Multiset({1: 1}), Multiset({2: 1})]
        expected_r_terms = [
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({0: 2, 2: 1}),
            Multiset({0: 2, 1: 1}),
        ]
        self.assertTrue(l_terms == expected_l_terms)
        self.assertTrue(r_terms == expected_r_terms)


class TestGetAllSubmultisets(QiskitDynamicsTestCase):
    """Test get_all_submultisets function."""

    def test_get_all_submultisets_case1(self):
        """Test get_all_submultisets case 1."""

        multisets = [Multiset({2: 2, 0: 1, 1: 1}), Multiset({1: 1, 2: 1})]
        expected = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({0: 1, 2: 2}),
            Multiset({1: 1, 2: 2}),
            Multiset({0: 1, 1: 1, 2: 2}),
        ]
        output = get_all_submultisets(multisets)
        self.assertTrue(expected == output)

    def test_get_all_submultisets_case2(self):
        """Test get_all_submultisets case 2."""

        multisets = [
            Multiset({2: 2, 0: 1, 3: 1}),
            Multiset({1: 1, 2: 1}),
            Multiset({0: 1}),
            Multiset({0: 1, 2: 2, 3: 1}),
        ]
        expected = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({3: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({0: 1, 3: 1}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({2: 1, 3: 1}),
            Multiset({0: 1, 2: 2}),
            Multiset({0: 1, 2: 1, 3: 1}),
            Multiset({2: 2, 3: 1}),
            Multiset({0: 1, 2: 2, 3: 1}),
        ]
        output = get_all_submultisets(multisets)
        self.assertTrue(expected == output)

    def test_get_all_submultisets_case3(self):
        """Test get_all_submultisets case 3."""

        multisets = [Multiset({0: 1, 1: 2, 2: 1, 3: 1})]
        expected = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({3: 1}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({0: 1, 3: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
            Multiset({1: 1, 3: 1}),
            Multiset({2: 1, 3: 1}),
            Multiset({0: 1, 1: 2}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({0: 1, 1: 1, 3: 1}),
            Multiset({0: 1, 2: 1, 3: 1}),
            Multiset({1: 2, 2: 1}),
            Multiset({1: 2, 3: 1}),
            Multiset({1: 1, 2: 1, 3: 1}),
            Multiset({0: 1, 1: 2, 2: 1}),
            Multiset({0: 1, 1: 2, 3: 1}),
            Multiset({0: 1, 1: 1, 2: 1, 3: 1}),
            Multiset({1: 2, 2: 1, 3: 1}),
            Multiset({0: 1, 1: 2, 2: 1, 3: 1}),
        ]
        output = get_all_submultisets(multisets)
        self.assertTrue(expected == output)
