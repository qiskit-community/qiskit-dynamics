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

"""Tests for functions in dyson_magnus.py."""

import numpy as np

from qiskit_dynamics.perturbation.dyson_magnus import (
    get_symmetric_dyson_lmult_rule,
    get_complete_dyson_indices,
    required_dyson_generator_indices,
    get_dyson_lmult_rule,
    get_q_term_list,
    q_product_rule,
    symmetric_magnus_from_dyson,
    symmetric_magnus_from_dyson_jax,
)

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit
except ImportError:
    pass


class TestSymmetricMagnusFromDyson(QiskitDynamicsTestCase):
    """Test symmetric_magnus_from_dyson function."""

    def setUp(self):
        self.symmetric_magnus_from_dyson = symmetric_magnus_from_dyson

    def test_symmetric_magnus_from_dyson_case1(self):
        """Case 1: a single base index to high order."""

        oc_symmetric_indices = [[0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]

        # random dyson terms
        rng = np.random.default_rng(9381)
        D = rng.uniform(size=(5, 10, 10))
        output = self.symmetric_magnus_from_dyson(oc_symmetric_indices, D)

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

    def test_symmetric_magnus_from_dyson_case2(self):
        """Case 2: two base indices."""

        oc_symmetric_indices = [[0], [1], [0, 0], [0, 1], [1, 1], [0, 0, 1]]

        # random dyson terms
        rng = np.random.default_rng(12412)
        D = rng.uniform(size=(6, 15, 15))
        output = self.symmetric_magnus_from_dyson(oc_symmetric_indices, D)

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

    def test_symmetric_magnus_from_dyson_case3(self):
        """Case 3: missing intermediate indices."""

        oc_symmetric_indices = [[0], [2], [0, 0], [0, 2], [0, 0, 2]]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(5, 8, 8))
        output = self.symmetric_magnus_from_dyson(oc_symmetric_indices, D)

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

    def test_symmetric_magnus_from_dyson_1st_order(self):
        """Test special handling when only first order terms are present."""

        oc_symmetric_indices = [[0], [2]]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(2, 8, 8))
        output = self.symmetric_magnus_from_dyson(oc_symmetric_indices, D)

        # should do nothing to the input
        self.assertAllClose(D, output)

    def test_symmetric_magnus_from_dyson_vectorized(self):
        """Test that the function is 'vectorized', i.e. works if
        the dyson terms and magnus terms are defined as a 3d array.
        """

        oc_symmetric_indices = [[0], [2], [0, 0], [0, 2], [0, 0, 2]]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(5, 3, 8, 8))
        output = self.symmetric_magnus_from_dyson(oc_symmetric_indices, D)

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


class TestSymmetricMagnusFromDysonJax(TestSymmetricMagnusFromDyson, TestJaxBase):
    """Jax version of TestSymmetricMagnusFromDyson."""

    def setUp(self):
        self.symmetric_magnus_from_dyson = symmetric_magnus_from_dyson_jax

    def test_magnus_from_dyson_jit(self):
        """Test that the function works with jitting."""

        oc_symmetric_indices = [[0], [2], [0, 0], [0, 2], [0, 0, 2]]

        # random dyson terms
        rng = np.random.default_rng(12398)
        D = rng.uniform(size=(5, 8, 8))

        # jit and compute
        jitted_func = jit(lambda x: self.symmetric_magnus_from_dyson(oc_symmetric_indices, x))
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


class TestSymmetricMagnusQTerms(QiskitDynamicsTestCase):
    """Test functions for constructing symmetric Magnus Q
    matrix descriptions.
    """

    def test_get_q_term_list_case1(self):
        """Test q_term list construction case 1."""

        oc_symmetric_indices = [[0], [1], [0, 1]]
        output = get_q_term_list(oc_symmetric_indices)
        expected = [([0], 1), ([1], 1), ([0, 1], 2), ([0, 1], 1)]

        self.assertTrue(output == expected)

    def test_get_q_term_list_case2(self):
        """Test q_term list construction case 2."""

        oc_symmetric_indices = [[0], [1], [2], [0, 1], [0, 2], [1, 1], [1, 2], [1, 1, 2]]
        output = get_q_term_list(oc_symmetric_indices)
        expected = [
            ([0], 1),
            ([1], 1),
            ([2], 1),
            ([0, 1], 2),
            ([0, 1], 1),
            ([0, 2], 2),
            ([0, 2], 1),
            ([1, 1], 2),
            ([1, 1], 1),
            ([1, 2], 2),
            ([1, 2], 1),
            ([1, 1, 2], 3),
            ([1, 1, 2], 2),
            ([1, 1, 2], 1),
        ]

        self.assertTrue(output == expected)

    def test_q_product_rule_case1(self):
        """Test construction of the q_product_rule."""
        oc_q_terms = [([0], 1), ([1], 1), ([0, 1], 2), ([0, 1], 1)]

        q_term = ([0, 1], 2)
        output = q_product_rule(q_term, oc_q_terms)
        expected = [(np.array([1.0, 1.0]), np.array([[0, 1], [1, 0]]))]
        self.assertMultRulesEqual(output, expected)

        q_term = ([0, 1], 1)
        output = q_product_rule(q_term, oc_q_terms)
        expected = [(np.array([1.0, -0.5]), np.array([[-1, 3], [-1, 2]]))]
        self.assertMultRulesEqual(output, expected)

    def test_q_product_rule_case2(self):
        """Test construction of the q_product_rule case 2."""
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

        q_term = ([0, 1, 2], 3)
        output = q_product_rule(q_term, oc_q_terms)
        expected = [(np.array([1.0, 1.0, 1.0]), np.array([[0, 7], [1, 5], [2, 3]]))]
        self.assertMultRulesEqual(output, expected)

        q_term = ([0, 1, 2], 2)
        output = q_product_rule(q_term, oc_q_terms)
        expected = [
            (np.ones(6, dtype=float), np.array([[0, 8], [1, 6], [2, 4], [4, 2], [6, 1], [8, 0]]))
        ]
        self.assertMultRulesEqual(output, expected)

        q_term = ([0, 1, 2], 1)
        output = q_product_rule(q_term, oc_q_terms)
        expected = [
            (np.array([1.0, -(1.0 / 2), -(1.0 / 6)]), np.array([[-1, 11], [-1, 10], [-1, 9]]))
        ]
        self.assertMultRulesEqual(output, expected)

    def assertMultRulesEqual(self, rule1, rule2):
        """Assert two multiplication rules are equal."""
        for sub_rule1, sub_rule2 in zip(rule1, rule2):
            self.assertAllClose(sub_rule1[0], sub_rule2[0])
            self.assertAllClose(sub_rule1[1], sub_rule2[1])


class TestDysonIndicesAndProduct(QiskitDynamicsTestCase):
    """Test cases for non-symmetric Dyson index and custom product handling."""

    def test_required_dyson_generator_indices(self):
        """Test that required generator indices are correctly identified."""
        complete_indices = [[0], [1], [0, 1], [0, 0, 1]]
        expected = [0, 1]
        self.assertEqual(expected, required_dyson_generator_indices(complete_indices))

        complete_indices = [[0], [1, 0]]
        expected = [0, 1]
        self.assertEqual(expected, required_dyson_generator_indices(complete_indices))

        complete_indices = [[0], [0, 0]]
        expected = [0]
        self.assertEqual(expected, required_dyson_generator_indices(complete_indices))

    def test_get_dyson_lmult_rule_case1(self):
        """Test get_dyson_mult_rules case 1."""
        complete_dyson_term_list = [[0], [1], [1, 1]]
        expected_lmult_rule = [
            (np.array([1.0]), np.array([-1, -1])),
            (np.array([1.0, 1.0]), np.array([[-1, 0], [0, -1]])),
            (np.array([1.0, 1.0]), np.array([[-1, 1], [1, -1]])),
            (np.array([1.0, 1.0]), np.array([[-1, 2], [1, 1]])),
        ]
        output = get_dyson_lmult_rule(complete_dyson_term_list, [0, 1])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def test_get_dyson_lmult_rule_case2(self):
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
        output = get_dyson_lmult_rule(complete_dyson_term_list, [0, 1])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def test_get_dyson_lmult_rule_case3(self):
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
        expected = expected_lmult_rule
        output = get_dyson_lmult_rule(complete_dyson_term_list, [0, 1, 2])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def test_get_dyson_lmult_rule_case4(self):
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
        expected = expected_lmult_rule
        output = get_dyson_lmult_rule(complete_dyson_term_list, [0, 1])
        self.assertMultRulesEqual(output, expected_lmult_rule)

    def assertMultRulesEqual(self, rule1, rule2):
        """Assert two multiplication rules are equal."""
        for sub_rule1, sub_rule2 in zip(rule1, rule2):
            self.assertAllClose(sub_rule1[0], sub_rule2[0])
            self.assertAllClose(sub_rule1[1], sub_rule2[1])

    def test_get_complete_dyson_indices(self):
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

        self._test_get_complete_dyson_indices(dyson_terms, expected)

    def test_get_complete_dyson_indices_case2(self):
        """Test _get_complete_dyson_terms case 2."""
        dyson_terms = [[2, 2, 0, 1]]
        expected = [[1], [0, 1], [2, 0, 1], [2, 2, 0, 1]]

        self._test_get_complete_dyson_indices(dyson_terms, expected)

    def _test_get_complete_dyson_indices(self, dyson_terms, expected):
        """Run test case for _get_complete_dyson_terms."""
        output = get_complete_dyson_indices(dyson_terms)

        self.assertListEquality(expected, output)
        self.assertIncreasingLen(output)

    def assertIncreasingLen(self, term_list):
        """Assert [len(x) for x in term_list] is increasing."""
        self.assertTrue(np.all(np.diff([len(x) for x in term_list]) >= 0))

    def assertListEquality(self, list_a, list_b):
        """Assert two lists have the same elements."""
        self.assertTrue(list_a == list_b)


class TestSymmetricDysonProduct(QiskitDynamicsTestCase):
    """Test cases for symmetric index handling and
    symmetric Dyson custom product setup.
    """

    def test_get_symmetric_dyson_lmult_rule_power_series_case1(self):
        """Test get_symmetric_dyson_lmult_rule for higher order terms in the generator
        decomposition case 1.
        """

        complete_symmetric_dyson_terms = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        A_list_indices = [[0], [1], [0, 1], [1, 1]]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [1, -1]])),
            (np.ones(4, dtype=float), np.array([[-1, 2], [0, 1], [1, 0], [2, -1]])),
            (np.ones(3, dtype=float), np.array([[-1, 3], [1, 1], [3, -1]])),
            (np.ones(5, dtype=float), np.array([[-1, 4], [0, 3], [1, 2], [2, 1], [3, 0]])),
        ]

        self._test_get_symmetric_dyson_lmult_rule(
            complete_symmetric_dyson_terms, expected_lmult_rule, A_list_indices=A_list_indices
        )

    def test_get_symmetric_dyson_lmult_rule_power_series_case2(self):
        """Test get_symmetric_dyson_lmult_rule for higher order terms in the generator
        decomposition case 2.
        """

        complete_symmetric_dyson_terms = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        A_list_indices = [[0], [1], [2], [0, 1]]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [1, -1]])),
            (np.ones(4, dtype=float), np.array([[-1, 2], [0, 1], [1, 0], [3, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 3], [1, 1]])),
            (np.ones(4, dtype=float), np.array([[-1, 4], [0, 3], [1, 2], [3, 1]])),
        ]

        self._test_get_symmetric_dyson_lmult_rule(
            complete_symmetric_dyson_terms, expected_lmult_rule, A_list_indices=A_list_indices
        )

    def test_get_symmetric_dyson_lmult_rule_case1(self):
        """Test _get_symmetric_dyson_lmult_rule case 1."""
        complete_symmetric_dyson_terms = [[0], [1], [0, 1], [1, 1], [0, 1, 1]]
        expected_lmult_rule = [
            (np.ones(1, dtype=float), np.array([[-1, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 0], [0, -1]])),
            (np.ones(2, dtype=float), np.array([[-1, 1], [1, -1]])),
            (np.ones(3, dtype=float), np.array([[-1, 2], [0, 1], [1, 0]])),
            (np.ones(2, dtype=float), np.array([[-1, 3], [1, 1]])),
            (np.ones(3, dtype=float), np.array([[-1, 4], [0, 3], [1, 2]])),
        ]

        self._test_get_symmetric_dyson_lmult_rule(
            complete_symmetric_dyson_terms, expected_lmult_rule
        )

    def test_get_symmetric_dyson_lmult_rule_case2(self):
        """Test _get_symmetric_dyson_lmult_rule case 2."""
        term_list = [
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

        self._test_get_symmetric_dyson_lmult_rule(term_list, expected_lmult_rule)

    def _test_get_symmetric_dyson_lmult_rule(
        self, complete_symmetric_dyson_term_list, expected, A_list_indices=None
    ):
        """Run a test case for _get_symmetric_dyson_mult_rules."""
        lmult_rule = get_symmetric_dyson_lmult_rule(
            complete_symmetric_dyson_term_list, A_list_indices
        )

        self.assertMultRulesEqual(lmult_rule, expected)

    def assertMultRulesEqual(self, rule1, rule2):

        """Assert two multiplication rules are equal."""
        for sub_rule1, sub_rule2 in zip(rule1, rule2):
            self.assertAllClose(sub_rule1[0], sub_rule2[0])
            self.assertAllClose(sub_rule1[1], sub_rule2[1])
