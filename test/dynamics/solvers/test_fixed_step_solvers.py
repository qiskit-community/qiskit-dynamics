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
# pylint: disable=invalid-name

"""
Tests for correctness of fixed-step solvers.
"""

from abc import ABC, abstractmethod

import numpy as np

from scipy.linalg import expm

from qiskit_dynamics.array import Array
from qiskit_dynamics.solvers.fixed_step_solvers import (
    RK4_solver,
    scipy_expm_solver,
    lanczos_diag_solver,
    jax_lanczos_diag_solver,
    jax_RK4_solver,
    jax_RK4_parallel_solver,
    jax_expm_solver,
    jax_expm_parallel_solver,
)
from qiskit_dynamics.solvers.lanczos import (
    lanczos_expm,
    jax_lanczos_expm,
)

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax.scipy.linalg import expm as jexpm
# pylint: disable=broad-except
except Exception:
    pass


class TestFixedStepBase(ABC, QiskitDynamicsTestCase):
    """Base class for defining common test patterns for fixed step solvers.

    Assumes the solvers follow the signature of those in fixed_step_solvers.py.
    """

    def setUp(self):
        """Setup RHS functions for testing of fixed step solvers. Constructed as LMDEs
        so that the tests can be used for both LMDE and ODE methods.
        """
        self.constant_generator = lambda t: -1j * Array([[0.0, 1.0], [1.0, 0.0]]).data

        def constant_rhs(t, y=None):
            if y is None:
                return self.constant_generator(t)
            else:
                return self.constant_generator(t) @ y

        self.constant_rhs = constant_rhs

        self.linear_generator = (
            lambda t: -1j * Array([[0.0, 1.0 - 1j * t], [1.0 + 1j * t, 0.0]]).data
        )

        def linear_rhs(t, y=None):
            if y is None:
                return self.linear_generator(t)
            else:
                return self.linear_generator(t) @ y

        self.linear_rhs = linear_rhs

        self.id2 = np.eye(2, dtype=complex)

        # build additional random generator and initial state
        rng = np.random.default_rng(5213)
        dim = 5
        self.dim = dim
        b = 1.0  # bound on size of random terms
        rand_ops = rng.uniform(low=-b, high=b, size=(3, dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(3, dim, dim)
        )
        # make anti-hermitian for numerical stability
        rand_ops = rand_ops - rand_ops.conj().transpose((0, 2, 1))

        self.random_y0 = rng.uniform(low=-b, high=b, size=(dim, dim)) + 1j * rng.uniform(
            low=-b, high=b, size=(dim, dim)
        )

        def random_generator(t):
            t = Array(t)
            output = np.sin(t) * rand_ops[0] + (t**5) * rand_ops[1] + np.exp(t) * rand_ops[2]
            return Array(output).data

        self.random_generator = random_generator

        def random_rhs(t, y=None):
            if y is None:
                return self.random_generator(t)
            else:
                return self.random_generator(t) @ y

        self.random_rhs = random_rhs
        self.rand_id = np.eye(dim, dtype=complex)

    @abstractmethod
    def take_step(self, rhs, t, y, h):
        """Definition of a single step in the fixed step scheme."""

    def take_n_steps(self, rhs, t, y, h, n_steps):
        """Take n steps of the solver of a given step size h."""

        for _ in range(n_steps):
            y = self.take_step(rhs, t, y, h)
            t = t + h

        return y

    @abstractmethod
    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        """Call the solver."""

    def test_t_eval_arg_no_overlap(self):
        """Test handling of t_eval when no overlap with t_span."""

        t_span = np.array([0.0, 1.0])
        t_eval = np.array([0.3, 0.5, 0.78])

        results = self.solve(self.constant_rhs, t_span, self.id2, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y0 = self.take_n_steps(self.constant_rhs, t=0.0, y=self.id2, h=0.1, n_steps=3)
        expected_y1 = self.take_n_steps(self.constant_rhs, t=0.3, y=expected_y0, h=0.1, n_steps=2)
        h = (0.78 - 0.5) / 3
        expected_y2 = self.take_n_steps(self.constant_rhs, t=0.5, y=expected_y1, h=h, n_steps=3)

        expected_y = [expected_y0, expected_y1, expected_y2]
        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_no_overlap_backwards(self):
        """Test handling of t_eval when no overlap with t_span for backwards integration."""

        t_span = np.array([1.0, 0.0])
        t_eval = np.array([0.78, 0.5, 0.3])

        results = self.solve(self.constant_rhs, t_span, self.id2, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        h = (0.78 - 1.0) / 3
        expected_y0 = self.take_n_steps(self.constant_rhs, t=1.0, y=self.id2, h=h, n_steps=3)
        h = (0.5 - 0.78) / 3
        expected_y1 = self.take_n_steps(self.constant_rhs, t=0.78, y=expected_y0, h=h, n_steps=3)
        expected_y2 = self.take_n_steps(self.constant_rhs, t=0.5, y=expected_y1, h=-0.1, n_steps=2)

        expected_y = [expected_y0, expected_y1, expected_y2]

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap(self):
        """Test handling of t_eval with overlap with t_span."""

        t_span = np.array([0.0, 1.0])
        t_eval = np.array([0.0, 0.5, 0.78])

        results = self.solve(self.constant_rhs, t_span, self.id2, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y0 = self.id2
        expected_y1 = self.take_n_steps(self.constant_rhs, t=0.0, y=expected_y0, h=0.1, n_steps=5)
        h = (0.78 - 0.5) / 3
        expected_y2 = self.take_n_steps(self.constant_rhs, t=0.5, y=expected_y1, h=h, n_steps=3)

        expected_y = [expected_y0, expected_y1, expected_y2]

        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap2(self):
        """Test handling of t_eval with overlap with t_span case 2."""

        t_span = np.array([0.0, 1.5])
        t_eval = np.array([0.0, 0.25, 1.37, 1.5])

        results = self.solve(self.linear_rhs, t_span, self.id2, max_dt=0.5, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y0 = self.id2
        expected_y1 = self.take_n_steps(self.linear_rhs, t=0.0, y=expected_y0, h=0.25, n_steps=1)
        h = (1.37 - 0.25) / 3
        expected_y2 = self.take_n_steps(self.linear_rhs, t=0.25, y=expected_y1, h=h, n_steps=3)
        h = 1.5 - 1.37
        expected_y3 = self.take_n_steps(self.linear_rhs, t=1.37, y=expected_y2, h=h, n_steps=1)

        expected_y = np.array([expected_y0, expected_y1, expected_y2, expected_y3])
        self.assertAllClose(expected_y, results.y)

    def test_t_eval_arg_overlap_backwards(self):
        """Test handling of t_eval with overlap with t_span for backwards integration."""

        t_span = np.array([1.0, 0.0])
        t_eval = np.array([0.78, 0.5, 0.0])

        results = self.solve(self.constant_rhs, t_span, self.id2, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        h = (0.78 - 1.0) / 3
        expected_y0 = self.take_n_steps(self.constant_rhs, t=1.0, y=self.id2, h=h, n_steps=3)
        h = (0.5 - 0.78) / 3
        expected_y1 = self.take_n_steps(self.constant_rhs, t=0.78, y=expected_y0, h=h, n_steps=3)
        expected_y2 = self.take_n_steps(self.constant_rhs, t=0.5, y=expected_y1, h=-0.1, n_steps=5)

        expected_y = [expected_y0, expected_y1, expected_y2]

        self.assertAllClose(expected_y, results.y)

    def test_solve_linear_generator(self):
        """Test solving a problem with non-trivial behaviour."""

        t_span = np.array([0.0, 1.0])
        t_eval = np.array([0.3, 0.5, 0.78])

        results = self.solve(self.linear_rhs, t_span, self.id2, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y0 = self.take_n_steps(self.linear_rhs, t=0.0, y=self.id2, h=0.1, n_steps=3)
        expected_y1 = self.take_n_steps(self.linear_rhs, t=0.3, y=expected_y0, h=0.1, n_steps=2)
        h = (0.78 - 0.5) / 3
        expected_y2 = self.take_n_steps(self.linear_rhs, t=0.5, y=expected_y1, h=h, n_steps=3)

        expected_y = [expected_y0, expected_y1, expected_y2]

        self.assertAllClose(expected_y, results.y)

    def test_solve_linear_generator_backwards(self):
        """Test solving a problem with non-trivial behaviour backwards."""

        t_span = np.array([1.0, 0.0])
        t_eval = np.array([0.78, 0.5, 0.3])

        results = self.solve(self.linear_rhs, t_span, self.id2, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        h = (0.78 - 1.0) / 3
        expected_y0 = self.take_n_steps(self.linear_rhs, t=1.0, y=self.id2, h=h, n_steps=3)
        h = (0.5 - 0.78) / 3
        expected_y1 = self.take_n_steps(self.linear_rhs, t=0.78, y=expected_y0, h=h, n_steps=3)
        expected_y2 = self.take_n_steps(self.linear_rhs, t=0.5, y=expected_y1, h=-0.1, n_steps=2)

        expected_y = [expected_y0, expected_y1, expected_y2]

        self.assertAllClose(expected_y, results.y)

    def test_random_generator(self):
        """Test generator with pseudo random structure."""

        t_span = np.array([2.1, 3.2])
        t_eval = np.array([2.3, 2.5, 2.78])
        y0 = self.random_y0

        results = self.solve(self.random_rhs, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y0 = self.take_n_steps(self.random_rhs, t=2.1, y=y0, h=0.1, n_steps=2)
        expected_y1 = self.take_n_steps(self.random_rhs, t=2.3, y=expected_y0, h=0.1, n_steps=2)
        h = (2.78 - 2.5) / 3
        expected_y2 = self.take_n_steps(self.random_rhs, t=2.5, y=expected_y1, h=h, n_steps=3)

        expected_y = np.array([expected_y0, expected_y1, expected_y2])
        self.assertAllClose(expected_y, results.y)

    def test_random_generator_nonsquare_y0(self):
        """Test generator with pseudo random structure for non-square y0."""

        t_span = np.array([2.1, 3.2])
        t_eval = np.array([2.3, 2.5, 2.78])
        y0 = self.random_y0[0]

        results = self.solve(self.random_rhs, t_span, y0, max_dt=0.1, t_eval=t_eval)

        self.assertAllClose(t_eval, results.t)

        expected_y0 = self.take_n_steps(self.random_rhs, t=2.1, y=y0, h=0.1, n_steps=2)
        expected_y1 = self.take_n_steps(self.random_rhs, t=2.3, y=expected_y0, h=0.1, n_steps=2)
        h = (2.78 - 2.5) / 3
        expected_y2 = self.take_n_steps(self.random_rhs, t=2.5, y=expected_y1, h=h, n_steps=3)

        expected_y = np.array([expected_y0, expected_y1, expected_y2])
        self.assertAllClose(expected_y, results.y)


class TestRK4Solver(TestFixedStepBase):
    """Tests for RK4_solver."""

    def take_step(self, rhs, t, y, h):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * h, y + (h * k1 / 2))
        k3 = rhs(t + 0.5 * h, y + (h * k2 / 2))
        k4 = rhs(t + h, y + h * k3)

        return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return RK4_solver(rhs, t_span, y0, max_dt, t_eval)


class TestScipyExpmSolver(TestFixedStepBase):
    """Tests for scipy_expm_solver."""

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        return expm(rhs(t + 0.5 * h) * h) @ y

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return scipy_expm_solver(rhs, t_span, y0, max_dt, t_eval)


class TestScipyExpmSolver_magnus2(TestFixedStepBase):
    """Tests for scipy_expm_solver with magnus order 2."""

    @classmethod
    def setUpClass(cls):
        """Setup constants."""

        cls.c1 = 0.5 - np.sqrt(3) / 6
        cls.c2 = 0.5 + np.sqrt(3) / 6
        cls.c3 = np.sqrt(3) / 12

        super().setUpClass()

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        A1 = rhs(t + self.c1 * h)
        A2 = rhs(t + self.c2 * h)

        return expm(0.5 * h * (A1 + A2) - (h**2) * self.c3 * (A1 @ A2 - A2 @ A1)) @ y

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return scipy_expm_solver(rhs, t_span, y0, max_dt, t_eval, magnus_order=2)


class TestScipyExpmSolver_magnus3(TestFixedStepBase):
    """Tests for scipy_expm_solver with magnus order 3."""

    @classmethod
    def setUpClass(cls):
        """Setup constants."""

        cls.c1 = 0.5 - np.sqrt(15) / 10
        cls.c2 = 0.5
        cls.c3 = 0.5 + np.sqrt(15) / 10
        cls.c4 = np.sqrt(15) / 3
        cls.c5 = 10. / 3

        super().setUpClass()

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        A1 = rhs(t + self.c1 * h)
        A2 = rhs(t + self.c2 * h)
        A3 = rhs(t + self.c3 * h)

        a1 = h * A2
        a2 = self.c4 * h * (A3 - A1)
        a3 = self.c5 * h * (A3 - 2 * A2 + A1)

        C1 = a1 @ a2 - a2 @ a1
        x0 = 2 * a3 + C1
        C2 = (x0 @ a1 - a1 @ x0) / 60

        x1 = (-20 * a1) - a3 + C1
        x2 = a2 + C2
        terms = a1 + (a3 / 12) + ((x1 @ x2 - x2 @ x1) / 240)

        return expm(terms) @ y

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return scipy_expm_solver(rhs, t_span, y0, max_dt, t_eval, magnus_order=3)


class TestLanczosDiagSolver(TestFixedStepBase):
    """Test cases for lanczos_diag."""

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        return lanczos_expm(rhs(t + 0.5 * h) * h, y, rhs(0).shape[0])

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return lanczos_diag_solver(rhs, t_span, y0, max_dt, self.dim, t_eval)

    def test_1d_2d_consistency(self):
        """Test that checks consistency of y0 being 1d v.s. 2d."""

        t_span = [0.0, 1.0]
        gen = self.random_rhs
        results = np.array(
            [
                self.solve(gen, t_span=t_span, y0=self.random_y0[:, idx], max_dt=0.1).y
                for idx in range(5)
            ]
        ).transpose(1, 2, 0)

        results2d = self.solve(gen, t_span=t_span, y0=self.random_y0, max_dt=0.1).y

        self.assertAllClose(results, results2d)

    def test_case_ix(self):
        """Standalone test case 1."""
        gen = lambda t: -1j * np.array([[0.0, 1.0], [1.0, 0.0]])
        y0 = np.array([0.0, 1.0])
        t_span = [0.0, np.pi / 4]
        result = self.solve(
            rhs=gen,
            t_span=t_span,
            y0=y0,
            max_dt=0.1,
        ).y

        self.assertAllClose(result[-1], expm(gen(0) * t_span[-1]) @ y0)

    def test_case_iz(self):
        """Standalone test case 2."""
        gen = lambda t: -1j * np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, -1]])
        y01 = np.array([0.0, 0.0, 1.0])
        y02 = np.array([0.0, 1.0, 1.0])
        t_span = [0.0, np.pi / 4]
        result1 = self.solve(
            rhs=gen,
            t_span=t_span,
            y0=y01,
            max_dt=0.1,
        ).y
        result2 = self.solve(
            rhs=gen,
            t_span=t_span,
            y0=y02,
            max_dt=0.1,
        ).y

        self.assertAllClose(result1[-1], expm(gen(0) * t_span[-1]) @ y01)
        self.assertAllClose(result2[-1], expm(gen(0) * t_span[-1]) @ y02)


class TestJaxFixedStepBase(TestFixedStepBase, TestJaxBase):
    """JAX version of TestFixedStepBase, adding JAX setup class TestJaxBase,
    and adding jit/grad test.
    """

    def test_t_span_with_jax_transformations(self):
        """Test handling of t_span as a list with jax transformations."""
        t_span = [0.0, 1.0]

        def func(amp):
            results = self.solve(
                lambda *args: amp * self.constant_rhs(*args), t_span, self.id2, max_dt=0.1
            )
            return Array(results.y[-1]).data

        jit_func = self.jit_wrap(func)
        output = jit_func(1.0)
        expected_y = self.take_n_steps(self.constant_rhs, t=0.0, y=self.id2, h=0.1, n_steps=10)
        self.assertAllClose(expected_y, output)

        grad_func = self.jit_grad_wrap(func)
        grad_func(1.0)


class TestJaxRK4Solver(TestJaxFixedStepBase):
    """Test cases for jax_RK4_solver."""

    def take_step(self, rhs, t, y, h):
        k1 = rhs(t, y)
        k2 = rhs(t + 0.5 * h, y + (h * k1 / 2))
        k3 = rhs(t + 0.5 * h, y + (h * k2 / 2))
        k4 = rhs(t + h, y + h * k3)

        return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return jax_RK4_solver(rhs, t_span, y0, max_dt, t_eval)


class TestJaxRK4ParallelSolver(TestJaxRK4Solver):
    """Test cases for jax_RK4_parallel_solver."""

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        # ensure that warning is raised as tests are run on CPU
        with self.assertWarns(Warning) as w:
            results = jax_RK4_parallel_solver(rhs, t_span, y0, max_dt, t_eval)

        self.assertTrue("run slower on CPUs" in str(w.warning))
        return results


class TestJaxExpmSolver(TestJaxFixedStepBase):
    """Test cases for jax_expm_solver."""

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        return jexpm(rhs(t + 0.5 * h) * h) @ y

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return jax_expm_solver(rhs, t_span, y0, max_dt, t_eval)


class TestJaxExpmSolver_magnus2(TestJaxFixedStepBase):
    """Test cases for jax_expm_solver with magnus_order 2."""
    
    @classmethod
    def setUpClass(cls):
        """Setup constants."""

        cls.c1 = 0.5 - np.sqrt(3) / 6
        cls.c2 = 0.5 + np.sqrt(3) / 6
        cls.c3 = np.sqrt(3) / 12

        super().setUpClass()

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        A1 = rhs(t + self.c1 * h)
        A2 = rhs(t + self.c2 * h)

        return jexpm(0.5 * h * (A1 + A2) - (h**2) * self.c3 * (A1 @ A2 - A2 @ A1)) @ y

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return jax_expm_solver(rhs, t_span, y0, max_dt, t_eval, magnus_order=2)


class TestJaxExpmSolver_magnus3(TestJaxFixedStepBase):
    """Test cases for jax_expm_solver with magnus_order 3."""
    
    @classmethod
    def setUpClass(cls):
        """Setup constants."""

        cls.c1 = 0.5 - np.sqrt(15) / 10
        cls.c2 = 0.5
        cls.c3 = 0.5 + np.sqrt(15) / 10
        cls.c4 = np.sqrt(15) / 3
        cls.c5 = 10. / 3

        super().setUpClass()

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        A1 = rhs(t + self.c1 * h)
        A2 = rhs(t + self.c2 * h)
        A3 = rhs(t + self.c3 * h)

        a1 = h * A2
        a2 = self.c4 * h * (A3 - A1)
        a3 = self.c5 * h * (A3 - 2 * A2 + A1)

        C1 = a1 @ a2 - a2 @ a1
        x0 = 2 * a3 + C1
        C2 = (x0 @ a1 - a1 @ x0) / 60

        x1 = (-20 * a1) - a3 + C1
        x2 = a2 + C2
        return jexpm(a1 + (a3 / 12) + ((x1 @ x2 - x2 @ x1) / 240)) @ y

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return jax_expm_solver(rhs, t_span, y0, max_dt, t_eval, magnus_order=3)



class TestJaxExpmParallelSolver(TestJaxExpmSolver):
    """Test cases for jax_expm_parallel_solver. Runs the same tests as
    TestJaxExpmSolver but for parallel version.
    """

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        # ensure that warning is raised as tests are run on CPU
        with self.assertWarns(Warning) as w:
            results = jax_expm_parallel_solver(rhs, t_span, y0, max_dt, t_eval)

        self.assertTrue("run slower on CPUs" in str(w.warning))
        return results

class TestJaxExpmParallelSolver_magnus2(TestJaxExpmSolver_magnus2):
    """Test cases for jax_expm_parallel_solver with magnus_order==2. Runs the same tests as
    TestJaxExpmSolver_magnus2 but for parallel version.
    """

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        # ensure that warning is raised as tests are run on CPU
        with self.assertWarns(Warning) as w:
            results = jax_expm_parallel_solver(rhs, t_span, y0, max_dt, t_eval, magnus_order=2)

        self.assertTrue("run slower on CPUs" in str(w.warning))
        return results

class TestJaxExpmParallelSolver_magnus3(TestJaxExpmSolver_magnus3):
    """Test cases for jax_expm_parallel_solver with magnus_order==3. Runs the same tests as
    TestJaxExpmSolver_magnus2 but for parallel version.
    """

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        # ensure that warning is raised as tests are run on CPU
        with self.assertWarns(Warning) as w:
            results = jax_expm_parallel_solver(rhs, t_span, y0, max_dt, t_eval, magnus_order=3)

        self.assertTrue("run slower on CPUs" in str(w.warning))
        return results


class TestJaxLanczosDiagSolver(TestLanczosDiagSolver, TestJaxFixedStepBase):
    """Test cases for jax_lanczos_diag."""

    def take_step(self, rhs, t, y, h):
        """In this case treat rhs like a generator."""
        return jax_lanczos_expm(rhs(t + 0.5 * h) * h, y, rhs(0).shape[0])

    def solve(self, rhs, t_span, y0, max_dt, t_eval=None):
        return jax_lanczos_diag_solver(rhs, t_span, y0, max_dt, self.dim, t_eval)


# to ensure unittest doesn't try to run the abstract classes
del TestFixedStepBase, TestJaxFixedStepBase
