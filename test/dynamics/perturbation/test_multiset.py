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

"""Tests Multiset."""

from qiskit import QiskitError

from qiskit_dynamics.perturbation import Multiset
from qiskit_dynamics.perturbation.multiset import (
    submultiset_filter,
    clean_multisets,
    get_all_submultisets,
)

from ..common import QiskitDynamicsTestCase


class TestMultiset(QiskitDynamicsTestCase):
    """Tests for Multiset class."""

    def test_empty_multiset(self):
        """Test empty Multiset."""
        empty = Multiset({})
        self.assertTrue(empty.count(2) == 0)

    def test_validation(self):
        """Test that non-integer types, and negative counts, raise errors."""

        with self.assertRaises(QiskitError) as qe:
            Multiset({"a": 1})
        self.assertTrue("must be integers" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            Multiset({0: 1, 1: -1})
        self.assertTrue("non-negative" in str(qe.exception))

    def test_eq(self):
        """Test __eq__."""

        self.assertTrue(Multiset({0: 2, 1: 1}) == Multiset({1: 1, 0: 2}))
        self.assertFalse(Multiset({0: 2, 1: 1}) == Multiset({1: 2}))

    def test_issubmultiset(self):
        """Test issubmultiset method."""
        B = Multiset({0: 2, 1: 1})
        self.assertFalse(Multiset({2: 1}).issubmultiset(B))
        self.assertFalse(Multiset({0: 3}).issubmultiset(B))
        self.assertTrue(Multiset({0: 1}).issubmultiset(B))
        self.assertTrue(Multiset({0: 1, 1: 1}).issubmultiset(B))
        self.assertTrue(Multiset({0: 2}).issubmultiset(B))
        self.assertTrue(B.issubmultiset(B))

    def test_union(self):
        """Test union."""

        ms1 = Multiset({0: 2, 1: 1})
        ms2 = Multiset({0: 2, 2: 1})

        self.assertTrue(ms1.union(ms2) == Multiset({0: 4, 1: 1, 2: 1}))

    def test_difference(self):
        """Test difference method."""
        B = Multiset({0: 2, 1: 1, 2: 2})

        self.assertTrue((B - Multiset({0: 1})) == Multiset({0: 1, 1: 1, 2: 2}))
        self.assertTrue((B - Multiset({0: 3})) == Multiset({1: 1, 2: 2}))
        self.assertTrue((B - Multiset({0: 1, 1: 1, 2: 1})) == Multiset({0: 1, 2: 1}))
        self.assertTrue((B - Multiset({0: 1, 2: 1, 1: 1})) == Multiset({0: 1, 2: 1}))
        self.assertTrue((B - Multiset({3: 1})) == B)

    def test_relabel(self):
        """Test relabel."""
        base_multiset = Multiset({0: 2, 1: 1})

        # relabeling one element to one not in the multiset
        self.assertTrue(Multiset({2: 2, 1: 1}) == base_multiset.relabel({0: 2}))

        # relabeling an element not in the set to another not in the set
        self.assertTrue(Multiset({0: 2, 1: 1}) == base_multiset.relabel({2: 3}))

        # relabeling all elements
        self.assertTrue(Multiset({1: 2, 0: 1}) == base_multiset.relabel({0: 1, 1: 0}))

        # empty relabeling
        self.assertTrue(Multiset({0: 2, 1: 1}) == base_multiset.relabel())

    def test_relabel_validation_errors(self):
        """Test relabeling validation errors."""
        base_multiset = Multiset({0: 2, 1: 1})

        with self.assertRaisesRegex(QiskitError, "must imply"):
            base_multiset.relabel({0: 1})

        with self.assertRaisesRegex(QiskitError, "must imply"):
            base_multiset.relabel({0: 0, 2: 0})

    def test_submultisets_and_complements_full_cases(self):
        """Test a few simple cases."""

        multiset = Multiset({0: 3})
        subsets, complements = multiset.submultisets_and_complements()
        expected_subsets, expected_complements = [Multiset({0: 1}), Multiset({0: 2})], [
            Multiset({0: 2}),
            Multiset({0: 1}),
        ]
        self.assertTrue(subsets == expected_subsets)
        self.assertTrue(complements == expected_complements)

        multiset = Multiset({0: 2, 1: 1})
        subsets, complements = multiset.submultisets_and_complements()
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
        subsets, complements = multiset.submultisets_and_complements()
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
        subsets, complements = multiset.submultisets_and_complements(3)
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
        l_terms, r_terms = multiset.submultisets_and_complements(2)
        expected_l_terms = [Multiset({0: 1}), Multiset({2: 1})]
        expected_r_terms = [Multiset({0: 1, 2: 2}), Multiset({0: 2, 2: 1})]
        self.assertTrue(l_terms == expected_l_terms)
        self.assertTrue(r_terms == expected_r_terms)

        multiset = Multiset({0: 2, 1: 1, 2: 1})
        l_terms, r_terms = multiset.submultisets_and_complements(2)
        expected_l_terms = [Multiset({0: 1}), Multiset({1: 1}), Multiset({2: 1})]
        expected_r_terms = [
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({0: 2, 2: 1}),
            Multiset({0: 2, 1: 1}),
        ]
        self.assertTrue(l_terms == expected_l_terms)
        self.assertTrue(r_terms == expected_r_terms)

    def test_lt(self):
        """Test less than."""
        self.assertTrue(Multiset({0: 2}) < Multiset({1: 2}))
        self.assertTrue(Multiset({0: 2}) < Multiset({0: 2, 1: 2}))
        self.assertFalse(Multiset({0: 2}) < Multiset({0: 2}))
        self.assertFalse(Multiset({0: 2, 1: 2}) < Multiset({0: 2}))
        self.assertFalse(Multiset({1: 2}) < Multiset({0: 2}))


class TestMultisetFunctions(QiskitDynamicsTestCase):
    """Test cases for multiset helper functions."""

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
