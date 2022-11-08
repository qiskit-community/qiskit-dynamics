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

from functools import partial
import numpy as np

from qiskit_dynamics.solvers.solver_utils import (
    jit_with_static_mutables,
    merge_t_args,
    trim_t_results,
    merge_t_args_jax,
    trim_t_results_jax,
)

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    import jax.numpy as jnp
except ImportError:
    pass


class TestTimeArgsHandling(QiskitDynamicsTestCase):
    """Tests for merge_t_args and trim_t_results functions."""

    def merge_t_args(self, t_span, t_eval=None):
        """Wrapper function for controlling which function is used via inheritance."""
        return merge_t_args(t_span, t_eval)

    def trim_t_results(self, results, t_eval=None):
        """Wrapper function for controlling which function is used via inheritance."""
        return trim_t_results(results, t_eval)

    def test_merge_t_args_dim_error(self):
        """Test raising of ValueError for non-1d t_eval."""
        with self.assertRaisesRegex(ValueError, "t_eval must be 1 dimensional."):
            self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([[0.0]]))

    def test_merge_t_args_interval_error(self):
        """Test raising ValueError if t_eval not in t_span."""
        with self.assertRaisesRegex(ValueError, "t_eval entries must lie in t_span."):
            self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([1.5]))

    def test_merge_t_args_interval_error_backwards(self):
        """Test raising ValueError if t_eval not in t_span for backwards integration."""
        with self.assertRaisesRegex(ValueError, "t_eval entries must lie in t_span."):
            self.merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=np.array([-1.5]))

    def test_merge_t_args_sort_error(self):
        """Test raising ValueError if t_eval is not correctly sorted."""
        with self.assertRaisesRegex(ValueError, "t_eval must be ordered"):
            self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.75, 0.25]))

    def test_merge_t_args_sort_error_backwards(self):
        """Test raising ValueError if t_eval is not correctly sorted for
        backwards integration.
        """
        with self.assertRaisesRegex(ValueError, "t_eval must be ordered"):
            self.merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=np.array([-0.75, -0.25]))

    def test_merge_t_args_no_overlap(self):
        """Test merging with no overlaps."""
        times = self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.25, 0.75]))
        self.assertAllClose(times, np.array([0.0, 0.25, 0.75, 1.0]))

    def test_merge_t_args_no_overlap_backwards(self):
        """Test merging with no overlaps for backwards integration."""
        times = self.merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=-np.array([0.25, 0.75]))
        self.assertAllClose(times, -np.array([0.0, 0.25, 0.75, 1.0]))

    def test_merge_t_args_with_overlap(self):
        """Test merging with with overlaps."""
        times = self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.0, 0.25, 0.75]))
        self.assertAllClose(times, np.array([0.0, 0.0, 0.25, 0.75, 1.0]))

        times = self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.25, 0.75, 1.0]))
        self.assertAllClose(times, np.array([0.0, 0.25, 0.75, 1.0, 1.0]))

        times = self.merge_t_args(
            t_span=np.array([0.0, 1.0]), t_eval=np.array([0.0, 0.25, 0.75, 1.0])
        )
        self.assertAllClose(times, np.array([0.0, 0.0, 0.25, 0.75, 1.0, 1.0]))

    def test_merge_t_args_with_overlap_backwards(self):
        """Test merging with with overlaps for backwards integration."""
        times = self.merge_t_args(
            t_span=np.array([1.0, -1.0]), t_eval=np.array([1.0, -0.25, -0.75])
        )
        self.assertAllClose(times, np.array([1.0, 1.0, -0.25, -0.75, -1.0]))

        times = self.merge_t_args(
            t_span=np.array([1.0, -1.0]), t_eval=np.array([-0.25, -0.75, -1.0])
        )
        self.assertAllClose(times, np.array([1.0, -0.25, -0.75, -1.0, -1.0]))

        times = self.merge_t_args(
            t_span=np.array([1.0, -1.0]), t_eval=np.array([1.0, -0.25, -0.75, -1.0])
        )
        self.assertAllClose(times, np.array([1.0, 1.0, -0.25, -0.75, -1.0, -1.0]))

    def test_trim_t_results(self):
        """Test trim_t_results works if t_eval is not None."""

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([0.0, 1.0, 2.0])
        empty_obj.y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        t_eval = np.array([1.0])
        trimmed_obj = self.trim_t_results(empty_obj, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([1.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.5, 0.5]]))

    def test_trim_t_results_backwards(self):
        """Test trim_t_results works if t_eval is not None and backwards integration."""

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([1.0, -1.0, -2.0])
        empty_obj.y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        t_eval = np.array([-1.0])
        trimmed_obj = self.trim_t_results(empty_obj, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([-1.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.5, 0.5]]))

    def test_trim_t_results_t_eval_is_None(self):
        """Test trim_t_results when t_eval is None."""

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([0.0, 1.0, 2.0])
        empty_obj.y = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])

        t_eval = None
        trimmed_obj = self.trim_t_results(empty_obj, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([0.0, 1.0, 2.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]]))


class TestTimeArgsHandlingJAX(TestTimeArgsHandling, TestJaxBase):
    """Tests for merge_t_args_jax and trim_t_results_jax functions."""

    def merge_t_args(self, t_span, t_eval=None):
        return merge_t_args_jax(t_span, t_eval)

    def trim_t_results(self, results, t_eval=None):
        return trim_t_results_jax(results, t_eval)

    def test_merge_t_args_interval_error(self):
        """Test output nan if t_eval not in t_span."""
        out = self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([1.5]))
        self.assertTrue(jnp.isnan(out.data).all())
        self.assertTrue(out.shape == (3,))

    def test_merge_t_args_interval_error_backwards(self):
        """Test output nan if t_eval not in t_span for backwards integration."""
        out = self.merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=np.array([-1.5]))
        self.assertTrue(jnp.isnan(out.data).all())
        self.assertTrue(out.shape == (3,))

    def test_merge_t_args_sort_error(self):
        """Test output nan if t_eval is not correctly sorted."""
        out = self.merge_t_args(t_span=np.array([0.0, 1.0]), t_eval=np.array([0.75, 0.25]))
        self.assertTrue(jnp.isnan(out.data).all())
        self.assertTrue(out.shape == (4,))

    def test_merge_t_args_sort_error_backwards(self):
        """Test raising ValueError if t_eval is not correctly sorted for
        backwards integration.
        """
        out = self.merge_t_args(t_span=np.array([0.0, -1.0]), t_eval=np.array([-0.75, -0.25]))
        self.assertTrue(jnp.isnan(out.data).all())
        self.assertTrue(out.shape == (4,))

    def test_trim_t_results_t0_duplicate(self):
        """Test trim_t_results_jax with duplicate in first time entry. Verifies that the first state
        for the corresponding time is kept. Due to a peculiarity in jax_odeint, the second will
        be nan (but only for the first time entry).
        """

        # empty object to assign attributes to
        empty_obj = type("", (), {})()

        empty_obj.t = np.array([0.0, 0.0, 1.0, 2.0])
        empty_obj.y = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [1.0, 0.0]])

        t_eval = np.array([0.0, 1.0])
        trimmed_obj = self.trim_t_results(empty_obj, t_eval)

        self.assertAllClose(trimmed_obj.t, np.array([0.0, 1.0]))
        self.assertAllClose(trimmed_obj.y, np.array([[0.0, 1.0], [0.5, 0.5]]))


class TestJitWithStaticMutables(QiskitDynamicsTestCase):
    """
    Tests the ``jit_with_static_mutables`` decorator function.
    """

    def test_cache(self):
        """Test that lru_cache mechanics are working."""
        # with jax.jit alone, the following would not be possible because dicts are mutable
        @partial(jit_with_static_mutables, static_argnames="cls", static_mutable_argnames="kwargs")
        def my_jit(arr, cls, kwargs):
            mult = 1 if cls is int else 2
            mult *= 2 if kwargs["a"] else 1
            return mult * arr

        self.assertAllClose(my_jit(np.array([1, 2, 3]), int, dict(a=True)), [2, 4, 6])
        self.assertEqual(tuple(my_jit.cache_info()), (0, 1, 128, 1))
        self.assertAllClose(my_jit(np.array([1, 2, 3]), float, dict(a=True)), [4, 8, 12])
        self.assertEqual(tuple(my_jit.cache_info()), (1, 1, 128, 1))
        self.assertAllClose(my_jit(np.array([1, 2, 3]), int, dict(a=True)), [2, 4, 6])
        self.assertEqual(tuple(my_jit.cache_info()), (2, 1, 128, 1))

        self.assertAllClose(my_jit(np.array([1, 2, 3]), float, dict(a=False)), [2, 4, 6])
        self.assertEqual(tuple(my_jit.cache_info()), (2, 2, 128, 2))
        self.assertAllClose(my_jit(np.array([1, 2, 3]), int, dict(a=False)), [1, 2, 3])
        self.assertEqual(tuple(my_jit.cache_info()), (3, 2, 128, 2))
        self.assertAllClose(my_jit(np.array([1, 2, 3]), int, dict(a=False)), [1, 2, 3])
        self.assertEqual(tuple(my_jit.cache_info()), (4, 2, 128, 2))

        my_jit.cache_clear()
        self.assertEqual(tuple(my_jit.cache_info()), (0, 0, 128, 0))

    def test_static_mutable_argnames(self):
        """Test that static_mutable_argnames accepts values as documented."""

        def my_jit(arr, a, b, c):  # pylint: disable=unused-argument,invalid-name
            return arr

        jit_with_static_mutables(my_jit)(*np.array([1, 2, 3, 4]))
        jit_with_static_mutables(my_jit, static_mutable_argnames="a")(*np.array([1, 2, 3, 4]))
        jit_with_static_mutables(my_jit, static_mutable_argnames=["a", "c"])(
            *np.array([1, 2, 3, 4])
        )

        with self.assertRaisesRegex(ValueError, "is not present"):
            jit_with_static_mutables(my_jit, static_mutable_argnames="d")

        with self.assertRaisesRegex(ValueError, "is not present"):
            jit_with_static_mutables(my_jit, static_mutable_argnames=["b", "d"])
