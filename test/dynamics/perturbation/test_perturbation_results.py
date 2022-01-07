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

"""Tests for perturbation_results.py"""

from qiskit import QiskitError

from qiskit_dynamics.array import Array
from qiskit_dynamics.perturbation.perturbation_results import PerturbationResults

from ..common import QiskitDynamicsTestCase


class TestPerturbationResults(QiskitDynamicsTestCase):
    """Test PerturbationResults."""

    def test_subscripting(self):
        """Test that subscripting via __getitem__ works."""

        results = PerturbationResults(
            expansion_method="dyson",
            expansion_labels=[[0], [1], [0, 1]],
            expansion_terms=Array([5, 6, 7]),
            sort_requested_labels=False,
        )

        self.assertTrue(results[[1]] == Array(6))

    def test_sorted_subscripting(self):
        """Test that subscripting via __getitem__ works."""

        results = PerturbationResults(
            expansion_method="dyson",
            expansion_labels=[[0], [1], [0, 1]],
            expansion_terms=Array([5, 6, 7]),
            sort_requested_labels=True,
        )

        self.assertTrue(results[[1, 0]] == Array(7))

    def test_subscripting_error(self):
        """Test an error gets raised when a requested term doesn't exist."""

        results = PerturbationResults(
            expansion_method="dyson",
            expansion_labels=[[0], [1], [0, 1]],
            expansion_terms=Array([5, 6, 7]),
            sort_requested_labels=False,
        )

        with self.assertRaises(QiskitError):
            # pylint: disable=pointless-statement
            results[[2]]
