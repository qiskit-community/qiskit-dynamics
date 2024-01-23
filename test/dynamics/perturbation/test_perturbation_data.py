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

import numpy as np

from qiskit import QiskitError

from multiset import Multiset

from qiskit_dynamics.perturbation.perturbation_data import PowerSeriesData, DysonLikeData

from ..common import QiskitDynamicsTestCase


class TestDysonLikeData(QiskitDynamicsTestCase):
    """Test DysonLikeData."""

    def test_get_item(self):
        """Test that get_term works."""
        results = DysonLikeData(data=np.array([5, 6, 7]), labels=[[0], [1], [0, 1]])
        self.assertTrue(results.get_item([1]) == 6)

    def test_get_item_error(self):
        """Test an error gets raised when a requested term doesn't exist."""
        results = DysonLikeData(data=np.array([5, 6, 7]), labels=[[0], [1], [0, 1]])
        with self.assertRaises(QiskitError):
            # pylint: disable=pointless-statement
            results.get_item([2])


class TestPowerSeriesData(QiskitDynamicsTestCase):
    """Test PowerSeriesData."""

    def test_get_item(self):
        """Test that get_term works."""
        results = PowerSeriesData(
            data=np.array([5, 6, 7]), labels=[Multiset([0]), Multiset([1]), Multiset([0, 1])]
        )
        self.assertTrue(results.get_item(Multiset([1])) == np.array(6))

    def test_automatic_casting(self):
        """Test that get_item works with automatic casting to Multiset."""
        results = PowerSeriesData(
            data=np.array([5, 6, 7]), labels=[Multiset([0]), Multiset([1]), Multiset([0, 1])]
        )
        self.assertTrue(results.get_item([1, 0]) == np.array(7))

    def test_get_item_error(self):
        """Test an error gets raised when a requested term doesn't exist."""
        results = PowerSeriesData(
            data=np.array([5, 6, 7]), labels=[Multiset([0]), Multiset([1]), Multiset([0, 1])]
        )
        with self.assertRaises(QiskitError):
            # pylint: disable=pointless-statement
            results.get_item([2])
