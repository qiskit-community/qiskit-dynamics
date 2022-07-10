# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint can't determine member or type for return from trim_t_results
# pylint: disable=no-member

"""
Tests to convert from pulse schedules to signals.
"""

import numpy as np

from qiskit_dynamics.solvers.solver_utils import merge_t_args, trim_t_results

from ..common import QiskitDynamicsTestCase


class TestTimeArgsHandling(QiskitDynamicsTestCase):
    """Tests for merge_t_args and trim_t_results functions."""

    def test_merge_t_args_dim_error(self):
        """Test raising of ValueError for non-1d t_eval."""
        with self.assertRaisesRegex(ValueError, "t_eval must be 1 dimensional."):
            merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([[0.0]]))

    def test_merge_t_args_interval_error(self):
        """Test raising ValueError if t_eval not in t_span."""
        with self.assertRaisesRegex(ValueError, "t_eval entries must lie in t_span."):
            merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([1.5]))

    def test_merge_t_args_interval_error_backwards(self):
        """Test raising ValueError if t_eval not in t_span for backwards integration."""
        with self.assertRaisesRegex(ValueError, "t_eval entries must lie in t_span."):
            merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=np.array([-1.5]))

    def test_merge_t_args_sort_error(self):
        """Test raising ValueError if t_eval is not correctly sorted."""
        with self.assertRaisesRegex(ValueError, "t_eval must be ordered"):
            merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.75, 0.25]))

    def test_merge_t_args_sort_error_backwards(self):
        """Test raising ValueError if t_eval is not correctly sorted for
        backwards integration.
        """
        with self.assertRaisesRegex(ValueError, "t_eval must be ordered"):
            merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=np.array([-0.75, -0.25]))

    def test_merge_t_args_no_overlap(self):
        """Test merging with no overlaps."""
        times = merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.25, 0.75]))
        self.assertAllClose(times, np.array([0.0, 0.25, 0.75, 1.0]))

    def test_merge_t_args_no_overlap_backwards(self):
        """Test merging with no overlaps for backwards integration."""
        times = merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=-np.array([0.25, 0.75]))
        self.assertAllClose(times, -np.array([0.0, 0.25, 0.75, 1.0]))

    def test_merge_t_args_with_overlap(self):
        """Test merging with with overlaps."""
        times = merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.0, 0.25, 0.75]))
        self.assertAllClose(times, np.array([0.0, 0.25, 0.75, 1.0]))

        times = merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.25, 0.75, 1.0]))
        self.assertAllClose(times, np.array([0.0, 0.25, 0.75, 1.0]))

        times = merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.0, 0.25, 0.75, 1.0]))
        self.assertAllClose(times, np.array([0.0, 0.25, 0.75, 1.0]))

    def test_merge_t_args_with_overlap_backwards(self):
        """Test merging with with overlaps for backwards integration."""
        times = merge_t_args(t_span=np.array([1.0, -1.0]), t_eval=np.array([1.0, -0.25, -0.75]))
        self.assertAllClose(times, np.array([1.0, -0.25, -0.75, -1.0]))

        times = merge_t_args(t_span=np.array([1.0, -1.0]), t_eval=np.array([-0.25, -0.75, -1.0]))
        self.assertAllClose(times, np.array([1.0, -0.25, -0.75, -1.0]))

        times = merge_t_args(
            t_span=np.array([1.0, -1.0]), t_eval=np.array([1.0, -0.25, -0.75, -1.0])
        )
        self.assertAllClose(times, np.array([1.0, -0.25, -0.75, -1.0]))

    def test_trim_t_results_overlap(self):
        """Test trim_t_results does nothing if endpoints are in t_eval."""

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([0.0, 1.0, 2.0])
        empty_obj.y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        t_span = np.array([0.0, 2.0])
        t_eval = np.array([0.0, 1.0, 2.0])
        trimmed_obj = trim_t_results(empty_obj, t_span, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([0.0, 1.0, 2.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]))

    def test_trim_t_results_overlap_backwards(self):
        """Test trim_t_results does nothing if endpoints are in t_eval for backwards integration."""

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([1.0, -1.0, -2.0])
        empty_obj.y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        t_span = np.array([1.0, -2.0])
        t_eval = np.array([1.0, -1.0, -2.0])
        trimmed_obj = trim_t_results(empty_obj, t_span, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([1.0, -1.0, -2.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]))

    def test_trim_t_results_no_overlap(self):
        """Test trim_t_results removes endpoints if not in t_eval."""

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([0.0, 1.0, 2.0])
        empty_obj.y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        t_span = np.array([0.0, 2.0])
        t_eval = np.array([1.0])
        trimmed_obj = trim_t_results(empty_obj, t_span, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([1.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.5, 0.5]]))

    def test_trim_t_results_no_overlap_backwards(self):
        """Test trim_t_results removes endpoints if not in t_eval for backwards integration."""

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([0.0, -1.0, -2.0])
        empty_obj.y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        t_span = np.array([0.0, -2.0])
        t_eval = np.array([-1.0])
        trimmed_obj = trim_t_results(empty_obj, t_span, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([-1.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.5, 0.5]]))
