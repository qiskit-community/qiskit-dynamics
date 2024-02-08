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
# pylint: disable=invalid-name,no-member

"""Tests for functions in dyson_magnus.py."""

from functools import partial

import numpy as np

from multiset import Multiset

from qiskit_dynamics.perturbation.dyson_magnus import (
    _get_dyson_lmult_rule,
    _get_complete_dyson_like_indices,
    _required_dyson_generator_indices,
    _get_dyson_like_lmult_rule,
    _get_q_term_list,
    _q_product_rule,
    _magnus_from_dyson,
    _magnus_from_dyson_jax,
)

from ..common import QiskitDynamicsTestCase, test_array_backends

try:
    from jax import jit
except ImportError:
    pass


@partial(test_array_backends, array_libraries=["numpy", "jax"])
class TestMagnusFromDyson:
    """Test _magnus_from_dyson and _magnus_from_dyson_jax functions."""

    def setUp(self):
        """Set dyson to magnus conversion function based on array library."""
        if self.array_library() == "jax":
            self._magnus_from_dyson = _magnus_from_dyson_jax
        else:
            self._magnus_from_dyson = _magnus_from_dyson

    def test__magnus_from_dyson_case1(self):
        """Case 1: a single base index to high order."""

        oc_symmetric_indices = [Multiset({0: k}) for k in range(1, 6)]

        # random dyson terms
        rng = np.random.default_rng(9381)
        D = rng.uniform(size=(5, 10, 10))
        output = self._magnus_from_dyson(oc_symmetric_indices, D)

        # compute expected output via manual application of Magnus recursion relations
        D1 = D[0]
        D2 = D[1]
        D3 = D[2]
        D4 = D[3]
        D5 = D[4]

        M1 = D1
        M2 = D2 - 0.5 * (M1 @ M1)
        M3 = D3 - 0.5 * (M1 @ M2 + M2 @ M1) - (1.0 / 6) * (M1 @ M1 @ M1)
        M4 = (
            D4
            - 0.5 * (M1 @ M3 + M3 @ M1 + M2 @ M2)
            - (1.0 / 6) * (M1 @ M1 @ M2 + M1 @ M2 @ M1 + M2 @ M1 @ M1)
            - (1.0 / 24) * (M1 @ M1 @ M1 @ M1)
        )
        M5 = (
            D5
            - 0.5 * (M1 @ M4 + M4 @ M1 + M2 @ M3 + M3 @ M2)
            - (1.0 / 6)
            * (
                M1 @ M1 @ M3
                + M1 @ M3 @ M1
                + M3 @ M1 @ M1
                + M1 @ M2 @ M2
                + M2 @ M1 @ M2
                + M2 @ M2 @ M1
            )
            - (1.0 / 24)
            * (M1 @ M1 @ M1 @ M2 + M1 @ M1 @ M2 @ M1 + M1 @ M2 @ M1 @ M1 + M2 @ M1 @ M1 @ M1)
            - (1.0 / (24 * 5)) * (M1 @ M1 @ M1 @ M1 @ M1)
        )

        expected = np.array([M1, M2, M3, M4, M5])

        self.assertAllClose(expected, output)

    def test__magnus_from_dyson_case2(self):
        """Case 2: two base indices."""

        oc_symmetric_indices = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
            Multiset({0: 2, 1: 1}),
        ]

        # random dyson terms
        rng = np.random.default_rng(12412)
        D = rng.uniform(size=(6, 15, 15))
        output = self._magnus_from_dyson(oc_symmetric_indices, D)

        # compute expected output via manual application of Magnus recursion relations
        D0 = D[0]
        D1 = D[1]
        D00 = D[2]
        D01 = D[3]
        D11 = D[4]
        D001 = D[5]

        M0 = D0
        M1 = D1
        M00 = D00 - 0.5 * (M0 @ M0)
        M01 = D01 - 0.5 * (M0 @ M1 + M1 @ M0)
        M11 = D11 - 0.5 * (M1 @ M1)
        M001 = (
            D001
            - 0.5 * (M0 @ M01 + M1 @ M00 + M00 @ M1 + M01 @ M0)
            - (1.0 / 6) * (M0 @ M0 @ M1 + M0 @ M1 @ M0 + M1 @ M0 @ M0)
        )

        expected = np.array([M0, M1, M00, M01, M11, M001])

        self.assertAllClose(expected, output)

    def test__magnus_from_dyson_case3(self):
        """Case 3: missing intermediate indices."""

        oc_symmetric_indices = [
            Multiset({0: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 2: 1}),
            Multiset({0: 2, 2: 1}),
        ]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(5, 8, 8))
        output = self._magnus_from_dyson(oc_symmetric_indices, D)

        # compute expected output via manual application of Magnus recursion relations
        D0 = D[0]
        D2 = D[1]
        D00 = D[2]
        D02 = D[3]
        D002 = D[4]

        M0 = D0
        M2 = D2
        M00 = D00 - 0.5 * (M0 @ M0)
        M02 = D02 - 0.5 * (M0 @ M2 + M2 @ M0)
        M002 = (
            D002
            - 0.5 * (M0 @ M02 + M2 @ M00 + M00 @ M2 + M02 @ M0)
            - (1.0 / 6) * (M0 @ M0 @ M2 + M0 @ M2 @ M0 + M2 @ M0 @ M0)
        )

        expected = np.array([M0, M2, M00, M02, M002])

        self.assertAllClose(expected, output)

    def test__magnus_from_dyson_1st_order(self):
        """Test special handling when only first order terms are present."""

        oc_symmetric_indices = [Multiset({0: 1}), Multiset({2: 1})]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(2, 8, 8))
        output = self._magnus_from_dyson(oc_symmetric_indices, D)

        # should do nothing to the input
        self.assertAllClose(D, output)

    def test__magnus_from_dyson_vectorized(self):
        """Test that the function is 'vectorized', i.e. works if
        the dyson terms and magnus terms are defined as a 3d array.
        """

        oc_symmetric_indices = [
            Multiset({0: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 2: 1}),
            Multiset({0: 2, 2: 1}),
        ]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(5, 3, 8, 8))
        output = self._magnus_from_dyson(oc_symmetric_indices, D)

        # compute expected output via manual application of Magnus recursion relations
        D0 = D[0]
        D2 = D[1]
        D00 = D[2]
        D02 = D[3]
        D002 = D[4]

        M0 = D0
        M2 = D2
        M00 = D00 - 0.5 * (M0 @ M0)
        M02 = D02 - 0.5 * (M0 @ M2 + M2 @ M0)
        M002 = (
            D002
            - 0.5 * (M0 @ M02 + M2 @ M00 + M00 @ M2 + M02 @ M0)
            - (1.0 / 6) * (M0 @ M0 @ M2 + M0 @ M2 @ M0 + M2 @ M0 @ M0)
        )

        expected = np.array([M0, M2, M00, M02, M002])
        self.assertAllClose(expected, output)


@partial(test_array_backends, array_libraries=["jax"])
class TestMagnusFromDysonJAXTransformations:
    """Test JAX transformations on _magnus_from_dyson_jax"""

    def test__magnus_from_dyson_jit(self):
        """Test that the function works with jitting."""

        oc_symmetric_indices = [
            Multiset({0: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 2: 1}),
            Multiset({0: 2, 2: 1}),
        ]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(5, 8, 8))

        # jit and compute
        jitted_func = jit(lambda x: _magnus_from_dyson_jax(oc_symmetric_indices, x))
        output = jitted_func(D)

        # compute expected output via manual application of Magnus recursion relations
        D0 = D[0]
        D2 = D[1]
        D00 = D[2]
        D02 = D[3]
        D002 = D[4]

        M0 = D0
        M2 = D2
        M00 = D00 - 0.5 * (M0 @ M0)
        M02 = D02 - 0.5 * (M0 @ M2 + M2 @ M0)
        M002 = (
            D002
            - 0.5 * (M0 @ M02 + M2 @ M00 + M00 @ M2 + M02 @ M0)
            - (1.0 / 6) * (M0 @ M0 @ M2 + M0 @ M2 @ M0 + M2 @ M0 @ M0)
        )

        expected = np.array([M0, M2, M00, M02, M002])

        self.assertAllClose(expected, output)


class TestMagnusQTerms(QiskitDynamicsTestCase):
    """Test functions for constructing symmetric Magnus Q
    matrix descriptions.
    """

    def test__get_q_term_list_case1(self):
        """Test q_term list construction case 1."""

        oc_symmetric_indices = [[0], [1], [0, 1]]
        oc_symmetric_indices = [Multiset({0: 1}), Multiset({1: 1}), Multiset({0: 1, 1: 1})]
        output = _get_q_term_list(oc_symmetric_indices)
        expected = [
            (Multiset({0: 1}), 1),
            (Multiset({1: 1}), 1),
            (Multiset({0: 1, 1: 1}), 2),
            (Multiset({0: 1, 1: 1}), 1),
        ]

        self.assertTrue(output == expected)

    def test__get_q_term_list_case2(self):
        """Test q_term list construction case 2."""

        oc_symmetric_indices = [[0], [1], [2], [0, 1], [0, 2], [1, 1], [1, 2], [1, 1, 2]]
        oc_symmetric_indices = [Multiset(multiset) for multiset in oc_symmetric_indices]
        output = _get_q_term_list(oc_symmetric_indices)
        expected = [
            (Multiset([0]), 1),
            (Multiset([1]), 1),
            (Multiset([2]), 1),
            (Multiset([0, 1]), 2),
            (Multiset([0, 1]), 1),
            (Multiset([0, 2]), 2),
            (Multiset([0, 2]), 1),
            (Multiset([1, 1]), 2),
            (Multiset([1, 1]), 1),
            (Multiset([1, 2]), 2),
            (Multiset([1, 2]), 1),
            (Multiset([1, 1, 2]), 3),
            (Multiset([1, 1, 2]), 2),
            (Multiset([1, 1, 2]), 1),
        ]

        self.assertTrue(output == expected)

    def test__q_product_rule_case1(self):
        """Test construction of the _q_product_rule."""
        oc_q_terms = [([0], 1), ([1], 1), ([0, 1], 2), ([0, 1], 1)]
        oc_q_terms = [(Multiset(x), y) for (x, y) in oc_q_terms]

        q_term = (Multiset([0, 1]), 2)
        output = _q_product_rule(q_term, oc_q_terms)
        expected = [(np.array([1.0, 1.0]), np.array([[0, 1], [1, 0]]))]
        self.assertMultRulesEqual(output, expected)

        q_term = (Multiset([0, 1]), 1)
        output = _q_product_rule(q_term, oc_q_terms)
        expected = [(np.array([1.0, -0.5]), np.array([[4, 3], [4, 2]]))]
        self.assertMultRulesEqual(output, expected)

    def test__q_product_rule_case2(self):
        """Test construction of the _q_product_rule case 2."""
        oc_q_terms = [
            ([0], 1),
            ([1], 1),
            ([2], 1),
            ([0, 1], 2),
            ([0, 1], 1),
            ([0, 2], 2),
            ([0, 2], 1),
            ([1, 2], 2),
            ([1, 2], 1),
            ([0, 1, 2], 3),
            ([0, 1, 2], 2),
            ([0, 1, 2], 1),
        ]
        oc_q_terms = [(Multiset(x), y) for (x, y) in oc_q_terms]

        q_term = (Multiset([0, 1, 2]), 3)
        output = _q_product_rule(q_term, oc_q_terms)
        expected = [(np.array([1.0, 1.0, 1.0]), np.array([[0, 7], [1, 5], [2, 3]]))]
        self.assertMultRulesEqual(output, expected)

        q_term = (Multiset([0, 1, 2]), 2)
        output = _q_product_rule(q_term, oc_q_terms)
        expected = [
            (np.ones(6, dtype=float), np.array([[0, 8], [1, 6], [2, 4], [4, 2], [6, 1], [8, 0]]))
        ]
        self.assertMultRulesEqual(output, expected)

        q_term = (Multiset([0, 1, 2]), 1)
        output = _q_product_rule(q_term, oc_q_terms)
        expected = [
            (np.array([1.0, -(1.0 / 2), -(1.0 / 6)]), np.array([[12, 11], [12, 10], [12, 9]]))
        ]
        self.assertMultRulesEqual(output, expected)

    def assertMultRulesEqual(self, rule1, rule2):
        """Assert two multiplication rules are equal."""
        for sub_rule1, sub_rule2 in zip(rule1, rule2):
            self.assertAllClose(sub_rule1[0], sub_rule2[0])
            self.assertAllClose(sub_rule1[1], sub_rule2[1])


class TestDysonLikeIndicesAndProduct(QiskitDynamicsTestCase):
    """Test cases for Dyson-like index and custom product handling."""

    def test__required_dyson_generator_indices(self):
        """Test that required generator indices are correctly identified."""
        complete_indices = [[0], [1], [0, 1], [0, 0, 1]]
        expected = [0, 1]
        self.assertEqual(expected, _required_dyson_generator_indices(complete_indices))

        complete_indices = [[0], [1, 0]]
        expected = [0, 1]
        self.assertEqual(expected, _required_dyson_generator_indices(complete_indices))

        complete_indices = [[0], [0, 0]]
        expected = [0]
        self.assertEqual(expected, _required_dyson_generator_indices(complete_indices))

    def test__get_dyson_like_lmult_rule_case1(self):
        """Test get_dyson_mult_rules case 1."""
        complete_dyson_term_list = [[0], [1], [1, 1]]
        expected_lmult_rule = [
            (np.array([1.0]), np.array([-1, -1])),
            (np.array([1.0, 1.0]), np.array([[-1, 0], [0, -1]])),
            (np.array([1.0, 1.0]), np.array([[-1, 1], [1, -1]])),
            (np.array([1.0, 1.0]), np.array([[-1, 2], [1, 1]])),
        ]
        output = _get_dyson_like_lmult_rule(complete_dyson_term_list, [0, 1])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def test__get_dyson_like_lmult_rule_case2(self):
        """Test get_dyson_mult_rules case 2.

        Note: This test was written before canonical ordering of
        complete lists was enforced.
        """
        complete_dyson_term_list = [
            [0],
            [1],
            [0, 0],
            [1, 0],
            [0, 1],
            [1, 0, 0],
            [0, 0, 1],
            [1, 0, 0, 1],
        ]
        expected_lmult_rule = [
            (np.array([1.0]), np.array([[-1, -1]])),
            (np.ones(2), np.array([[-1, 0], [0, -1]])),
            (np.ones(2), np.array([[-1, 1], [1, -1]])),
            (np.ones(2), np.array([[-1, 2], [0, 0]])),
            (np.ones(2), np.array([[-1, 3], [1, 0]])),
            (np.ones(2), np.array([[-1, 4], [0, 1]])),
            (np.ones(2), np.array([[-1, 5], [1, 2]])),
            (np.ones(2), np.array([[-1, 6], [0, 4]])),
            (np.ones(2), np.array([[-1, 7], [1, 6]])),
        ]
        output = _get_dyson_like_lmult_rule(complete_dyson_term_list, [0, 1])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def test__get_dyson_like_lmult_rule_case3(self):
        """Test get_dyson_mult_rules case 3.

        Note: This test was written before canonical ordering of
        complete lists was enforced.
        """
        complete_dyson_term_list = [
            [0],
            [1],
            [2],
            [0, 0],
            [1, 0],
            [1, 2],
            [0, 2],
            [2, 1],
            [2, 1, 0],
            [2, 1, 2],
        ]
        expected_lmult_rule = [
            (np.array([1.0]), np.array([[-1, -1]])),
            (np.ones(2), np.array([[-1, 0], [0, -1]])),
            (np.ones(2), np.array([[-1, 1], [1, -1]])),
            (np.ones(2), np.array([[-1, 2], [2, -1]])),
            (np.ones(2), np.array([[-1, 3], [0, 0]])),
            (np.ones(2), np.array([[-1, 4], [1, 0]])),
            (np.ones(2), np.array([[-1, 5], [1, 2]])),
            (np.ones(2), np.array([[-1, 6], [0, 2]])),
            (np.ones(2), np.array([[-1, 7], [2, 1]])),
            (np.ones(2), np.array([[-1, 8], [2, 4]])),
            (np.ones(2), np.array([[-1, 9], [2, 5]])),
        ]
        output = _get_dyson_like_lmult_rule(complete_dyson_term_list, [0, 1, 2])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def test__get_dyson_like_lmult_rule_case4(self):
        """Test get_dyson_mult_rules case 4: non-complete list.

        Note: This test was written before canonical ordering of
        complete lists was enforced.
        """
        complete_dyson_term_list = [
            [1],
            [0, 1],
            [0, 0, 1],
            [0, 0, 0, 1],
        ]
        expected_lmult_rule = [
            (np.array([1.0]), np.array([[-1, -1]])),
            (np.ones(2), np.array([[-1, 0], [1, -1]])),
            (np.ones(2), np.array([[-1, 1], [0, 0]])),
            (np.ones(2), np.array([[-1, 2], [0, 1]])),
            (np.ones(2), np.array([[-1, 3], [0, 2]])),
        ]
        output = _get_dyson_like_lmult_rule(complete_dyson_term_list, [0, 1])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def assertMultRulesEqual(self, rule1, rule2):
        """Assert two multiplication rules are equal."""
        for sub_rule1, sub_rule2 in zip(rule1, rule2):
            self.assertAllClose(sub_rule1[0], sub_rule2[0])
            self.assertAllClose(sub_rule1[1], sub_rule2[1])

    def test__get_complete_dyson_like_indices(self):
        """Test _get_complete_dyson_terms case 1."""
        dyson_terms = [[2], [0], [2, 1], [2, 2, 0, 1]]
        expected = [
            [0],
            [1],
            [2],
            [0, 1],
            [2, 1],
            [2, 0, 1],
            [2, 2, 0, 1],
        ]

        self._test__get_complete_dyson_like_indices(dyson_terms, expected)

    def test__get_complete_dyson_like_indices_case2(self):
        """Test _get_complete_dyson_terms case 2."""
        dyson_terms = [[2, 2, 0, 1]]
        expected = [[1], [0, 1], [2, 0, 1], [2, 2, 0, 1]]

        self._test__get_complete_dyson_like_indices(dyson_terms, expected)

    def _test__get_complete_dyson_like_indices(self, dyson_terms, expected):
        """Run test case for _get_complete_dyson_terms."""
        output = _get_complete_dyson_like_indices(dyson_terms)

        self.assertListEquality(expected, output)
        self.assertIncreasingLen(output)

    def assertIncreasingLen(self, term_list):
        """Assert [len(x) for x in term_list] is increasing."""
        self.assertTrue(np.all(np.diff([len(x) for x in term_list]) >= 0))

    def assertListEquality(self, list_a, list_b):
        """Assert two lists have the same elements."""
        self.assertTrue(list_a == list_b)


class TestDysonProduct(QiskitDynamicsTestCase):
    """Test cases for Dyson RHS product setup."""

    def test__get_dyson_lmult_rule_power_series_case1(self):
        """Test _get_dyson_lmult_rule for higher order terms in the generator
        decomposition case 1.
        """

        expansion_labels = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        expansion_labels = [Multiset(label) for label in expansion_labels]
        perturbation_labels = [[0], [1], [0, 1], [1, 1]]
        perturbation_labels = [Multiset(label) for label in perturbation_labels]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [1, -1]])),
            (np.ones(4, dtype=float), np.array([[-1, 2], [0, 1], [1, 0], [2, -1]])),
            (np.ones(3, dtype=float), np.array([[-1, 3], [1, 1], [3, -1]])),
            (np.ones(5, dtype=float), np.array([[-1, 4], [0, 3], [1, 2], [2, 1], [3, 0]])),
        ]

        self._test__get_dyson_lmult_rule(
            expansion_labels,
            expected_lmult_rule,
            perturbation_labels=perturbation_labels,
        )

    def test__get_dyson_lmult_rule_power_series_case1_missing(self):
        """Test _get_dyson_lmult_rule for higher order terms in the generator
        decomposition case 1 where there is no corresponding perturbation_label term for
        a desired expansion term.
        """

        expansion_labels = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        expansion_labels = [Multiset(label) for label in expansion_labels]
        perturbation_labels = [[0], [0, 1], [1, 1]]
        perturbation_labels = [Multiset(label) for label in perturbation_labels]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(1, dtype=float), np.array([[-1, 1]])),
            (np.ones(3, dtype=float), np.array([[-1, 2], [0, 1], [1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 3], [2, -1]])),
            (np.ones(4, dtype=float), np.array([[-1, 4], [0, 3], [1, 1], [2, 0]])),
        ]

        self._test__get_dyson_lmult_rule(
            expansion_labels,
            expected_lmult_rule,
            perturbation_labels=perturbation_labels,
        )

    def test__get_dyson_lmult_rule_power_series_case2(self):
        """Test _get_dyson_lmult_rule for higher order terms in the generator
        decomposition case 2.
        """

        expansion_labels = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        expansion_labels = [Multiset(label) for label in expansion_labels]
        perturbation_labels = [[0], [1], [2], [0, 1]]
        perturbation_labels = [Multiset(label) for label in perturbation_labels]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [1, -1]])),
            (np.ones(4, dtype=float), np.array([[-1, 2], [0, 1], [1, 0], [3, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 3], [1, 1]])),
            (np.ones(4, dtype=float), np.array([[-1, 4], [0, 3], [1, 2], [3, 1]])),
        ]

        self._test__get_dyson_lmult_rule(
            expansion_labels,
            expected_lmult_rule,
            perturbation_labels=perturbation_labels,
        )

    def test__get_dyson_lmult_rule_power_series_case2_unordered(self):
        """Test _get_dyson_lmult_rule for higher order terms in the generator
        decomposition case 2, with the perturbation_labels being non-canonically ordered.

        Note that the conversion of case2 to case2_unordered requires relabelling 1 <-> 3
        in expected_lmult_rule, but also changing the ordering of the pairs in the matrix-product
        description of expected_lmult_rule, as the rule is constructed by iterating through the
        entries of perturbation_labels. The matrix-product description must be ordered by the first
        index.
        """

        expansion_labels = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        expansion_labels = [Multiset(label) for label in expansion_labels]
        perturbation_labels = [[0], [0, 1], [2], [1]]
        perturbation_labels = [Multiset(label) for label in perturbation_labels]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [3, -1]])),
            (np.ones(4, dtype=float), np.array([[-1, 2], [0, 1], [1, -1], [3, 0]])),
            (np.ones(2, dtype=float), np.array([[-1, 3], [3, 1]])),
            (np.ones(4, dtype=float), np.array([[-1, 4], [0, 3], [1, 1], [3, 2]])),
        ]

        self._test__get_dyson_lmult_rule(
            expansion_labels,
            expected_lmult_rule,
            perturbation_labels=perturbation_labels,
        )

    def test__get_dyson_lmult_rule_case1(self):
        """Test __get_dyson_lmult_rule case 1."""
        expansion_labels = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        expansion_labels = [Multiset(label) for label in expansion_labels]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [1, -1]])),
            (np.ones(3, dtype=float), np.array([[-1, 2], [0, 1], [1, 0]])),
            (np.ones(2, dtype=float), np.array([[-1, 3], [1, 1]])),
            (np.ones(3, dtype=float), np.array([[-1, 4], [0, 3], [1, 2]])),
        ]

        self._test__get_dyson_lmult_rule(expansion_labels, expected_lmult_rule)

    def test__get_dyson_lmult_rule_case2(self):
        """Test __get_dyson_lmult_rule case 2."""
        expansion_labels = [
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
        expansion_labels = [Multiset(label) for label in expansion_labels]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 2], [2, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 3], [3, -1]])),
            (np.ones(3, dtype=float), np.array([[-1, 4], [0, 1], [1, 0]])),
            (np.ones(3, dtype=float), np.array([[-1, 5], [0, 2], [2, 0]])),
            (np.ones(3, dtype=float), np.array([[-1, 6], [0, 3], [3, 0]])),
            (np.ones(2, dtype=float), np.array([[-1, 7], [1, 1]])),
            (np.ones(3, dtype=float), np.array([[-1, 8], [1, 2], [2, 1]])),
            (np.ones(3, dtype=float), np.array([[-1, 9], [1, 3], [3, 1]])),
            (np.ones(3, dtype=float), np.array([[-1, 10], [2, 3], [3, 2]])),
            (np.ones(3, dtype=float), np.array([[-1, 11], [0, 7], [1, 4]])),
            (np.ones(4, dtype=float), np.array([[-1, 12], [0, 8], [1, 5], [2, 4]])),
            (np.ones(4, dtype=float), np.array([[-1, 13], [0, 9], [1, 6], [3, 4]])),
            (np.ones(4, dtype=float), np.array([[-1, 14], [0, 10], [2, 6], [3, 5]])),
            (np.ones(3, dtype=float), np.array([[-1, 15], [1, 8], [2, 7]])),
            (np.ones(3, dtype=float), np.array([[-1, 16], [1, 9], [3, 7]])),
            (np.ones(4, dtype=float), np.array([[-1, 17], [1, 10], [2, 9], [3, 8]])),
            (np.ones(4, dtype=float), np.array([[-1, 18], [0, 15], [1, 12], [2, 11]])),
            (np.ones(4, dtype=float), np.array([[-1, 19], [0, 16], [1, 13], [3, 11]])),
            (np.ones(5, dtype=float), np.array([[-1, 20], [0, 17], [1, 14], [2, 13], [3, 12]])),
            (np.ones(4, dtype=float), np.array([[-1, 21], [1, 17], [2, 16], [3, 15]])),
            (np.ones(5, dtype=float), np.array([[-1, 22], [0, 21], [1, 20], [2, 19], [3, 18]])),
        ]

        self._test__get_dyson_lmult_rule(expansion_labels, expected_lmult_rule)

    def _test__get_dyson_lmult_rule(
        self, complete_index_multisets, expected, perturbation_labels=None
    ):
        """Run a test case for _get_dyson_lmult_rule."""
        lmult_rule = _get_dyson_lmult_rule(complete_index_multisets, perturbation_labels)
        self.assertMultRulesEqual(lmult_rule, expected)

    def assertMultRulesEqual(self, rule1, rule2):
        """Assert two multiplication rules are equal."""
        for sub_rule1, sub_rule2 in zip(rule1, rule2):
            self.assertAllClose(sub_rule1[0], sub_rule2[0])
            self.assertAllClose(sub_rule1[1], sub_rule2[1])


class TestWorkaround(QiskitDynamicsTestCase):
    """Test whether workaround in dyson_magnus._setup_dyson_rhs_jax is no longer required.

    The workaround was introduced in the same commit as this test class to avoid an error being
    raised by a non-trivial combination of JAX transformations. The test in this class has been
    set up to expect the original minimal reproduction of the issue to fail. Once it no longer
    fails, the changes made to _setup_dyson_rhs_jax in this commit should be reverted.

    See https://github.com/google/jax/discussions/9951#discussioncomment-2385157 for discussion of
    issue.
    """

    def test_minimal_example(self):
        """Test minimal reproduction of issue."""

        with self.assertRaises(Exception):
            import jax.numpy as jnp
            from jax import grad, vmap
            from jax.lax import switch
            from jax.experimental.ode import odeint

            # pylint: disable=unused-argument
            def A0(t):
                return 2.0

            # pylint: disable=unused-argument
            def A1(a, t):
                return a**2

            y0 = np.random.rand(2)
            T = np.pi * 1.232

            def test_func(a):
                eval_list = [A0, lambda t: A1(a, t)]

                def single_eval(idx, t):
                    return switch(idx, eval_list, t)

                multiple_eval = vmap(single_eval, in_axes=(0, None))
                idx_list = jnp.array([0, 1])

                def rhs(y, t):
                    return multiple_eval(idx_list, t) * y

                out = odeint(rhs, y0=y0, t=jnp.array([0, T], dtype=float), atol=1e-13, rtol=1e-13)
                return out

            jit(grad(lambda a: test_func(a)[-1][1].real))(1.0)
