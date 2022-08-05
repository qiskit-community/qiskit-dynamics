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

"""Tests for perturbation_utils.py."""

from qiskit import QiskitError

from multiset import Multiset

from qiskit_dynamics.perturbation.multiset_utils import _sorted_multisets
from qiskit_dynamics.perturbation.perturbation_utils import (
    _ordered_partitions,
    _merge_multiset_expansion_order_labels,
    _merge_list_expansion_order_labels,
)

from ..common import QiskitDynamicsTestCase


class Test_merge_multiset_expansion_order_labels(QiskitDynamicsTestCase):
    """Test helper function _merge_multiset_expansion_order_labels."""

    def test_arg_validation(self):
        """Test error raised when not enough args."""

        with self.assertRaisesRegex(QiskitError, "At least one"):
            _merge_multiset_expansion_order_labels(
                perturbation_labels=[[0]], expansion_order=None, expansion_labels=None
            )

    def test_non_negative_validation(self):
        """Test error is raised when anything other than non-negative ints used."""
        with self.assertRaisesRegex(QiskitError, "non-negative"):
            _merge_multiset_expansion_order_labels(
                perturbation_labels=["a"], expansion_order=2, expansion_labels=None
            )

        with self.assertRaisesRegex(QiskitError, "non-negative"):
            _merge_multiset_expansion_order_labels(
                perturbation_labels=[[0]], expansion_order=2, expansion_labels=["a"]
            )

    def test_order(self):
        """Test specifying terms up to a given order."""
        perturbation_labels = [[0], [1], [2]]
        perturbation_labels = [Multiset(label) for label in perturbation_labels]

        output = _merge_multiset_expansion_order_labels(
            perturbation_labels=perturbation_labels,
            expansion_order=3,
            expansion_labels=None,
        )
        expected = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 2],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ]
        expected = [Multiset(label) for label in expected]

        self.assertTrue(output == expected)

    def test_order_skipped_terms(self):
        """Test handling of 'missing' indices."""

        perturbation_labels = [[0], [2], [3]]
        perturbation_labels = [Multiset(label) for label in perturbation_labels]

        output = _merge_multiset_expansion_order_labels(
            perturbation_labels=perturbation_labels,
            expansion_order=3,
            expansion_labels=None,
        )
        expected = [
            [0, 0, 0],
            [0, 0, 2],
            [0, 0, 3],
            [0, 2, 2],
            [0, 2, 3],
            [0, 3, 3],
            [2, 2, 2],
            [2, 2, 3],
            [2, 3, 3],
            [3, 3, 3],
        ]
        expected = [Multiset(label) for label in expected]

        self.assertTrue(output == expected)

    def test_terms_with_no_order(self):
        """Test that nothing happens if expansion_labels is specified
        while expansion_order is not."""

        input_terms = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 1],
            [0, 1, 2],
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
            [1, 0],
            [0, 2],
        ]
        input_terms = [Multiset(label) for label in input_terms]

        perturbation_labels = [[0], [1], [2]]
        perturbation_labels = [Multiset(label) for label in perturbation_labels]
        output = _merge_multiset_expansion_order_labels(
            perturbation_labels=perturbation_labels,
            expansion_order=None,
            expansion_labels=input_terms,
        )
        expected = [
            [0, 1],
            [0, 2],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 1],
            [0, 1, 2],
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 2],
        ]
        expected = [Multiset(label) for label in expected]
        self.assertTrue(output == expected)

    def test_merge(self):
        """Test for when both expansion_labels and expansion_order
        are specified."""

        extra_terms = [Multiset({0: 3}), Multiset({0: 1, 1: 1, 2: 1})]
        perturbation_labels = [
            Multiset([0]),
            Multiset([1]),
            Multiset([2]),
        ]
        output = _merge_multiset_expansion_order_labels(
            perturbation_labels=perturbation_labels,
            expansion_order=2,
            expansion_labels=extra_terms,
        )

        expected = [[0, 0, 0], [0, 1, 2], [0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]
        expected = _sorted_multisets([Multiset(label) for label in expected])
        self.assertTrue(output == expected)

    def test_merge_skipped_terms(self):
        """Test for when both expansion_labels and expansion_order
        are specified, with terms skipped"""

        extra_terms = [Multiset({0: 3}), Multiset({0: 1, 1: 1, 2: 1})]
        perturbation_labels = [
            Multiset([0]),
            Multiset([2]),
            Multiset([3]),
        ]
        output = _merge_multiset_expansion_order_labels(
            perturbation_labels=perturbation_labels,
            expansion_order=2,
            expansion_labels=extra_terms,
        )

        expected = [[0, 0, 0], [0, 1, 2], [0, 0], [0, 2], [0, 3], [2, 2], [2, 3], [3, 3]]
        expected = _sorted_multisets([Multiset(label) for label in expected])
        self.assertTrue(output == expected)


class Test_merge_list_expansion_order_labels(QiskitDynamicsTestCase):
    """Test helper function _merge_list_expansion_order_labels."""

    def test_validation(self):
        """Test error raised when not enough args."""

        with self.assertRaisesRegex(QiskitError, "At least one"):
            _merge_list_expansion_order_labels(
                perturbation_num=3, expansion_order=None, expansion_labels=None
            )

    def test_order_only(self):
        """Test specifying terms up to a given order."""

        output = _merge_list_expansion_order_labels(
            perturbation_num=3, expansion_order=3, expansion_labels=None
        )
        expected = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
        ]

        self.assertTrue(output == expected)

    def test_labels_only(self):
        """Test case for when only expansion_labels is specified."""

        output = _merge_list_expansion_order_labels(
            perturbation_num=3, expansion_order=None, expansion_labels=[[0, 1]]
        )
        self.assertTrue([[0, 1]] == output)

    def test_merge(self):
        """Test case for when both order and labels specified."""

        output = _merge_list_expansion_order_labels(
            perturbation_num=3,
            expansion_order=3,
            expansion_labels=[[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 0]],
        )
        expected = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 0],
            [1, 0, 1],
            [1, 0, 2],
            [1, 1, 0],
            [1, 1, 1],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
        ]

        self.assertTrue(output == expected)


class Test_ordered_partitions(QiskitDynamicsTestCase):
    """Tests for _ordered_partitions function."""

    def test_case_1(self):
        """Test case 1."""
        output = _ordered_partitions(3, 2)
        expected = [[0, 3], [1, 2], [2, 1], [3, 0]]
        self.assertTrue(output == expected)

    def test_case_2(self):
        """Test case 2."""
        output = _ordered_partitions(1, 3)
        expected = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
        self.assertTrue(output == expected)

    def test_case_3(self):
        """Test case 3."""
        output = _ordered_partitions(4, 3)
        expected = [
            [0, 0, 4],
            [0, 1, 3],
            [0, 2, 2],
            [0, 3, 1],
            [0, 4, 0],
            [1, 0, 3],
            [1, 1, 2],
            [1, 2, 1],
            [1, 3, 0],
            [2, 0, 2],
            [2, 1, 1],
            [2, 2, 0],
            [3, 0, 1],
            [3, 1, 0],
            [4, 0, 0],
        ]
        self.assertTrue(output == expected)
