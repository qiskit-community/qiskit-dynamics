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
# pylint: disable=invalid-name, no-member

"""Tests for perturbative_solver.py"""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from qiskit_dynamics import Signal, Solver
from qiskit_dynamics.array import Array

from qiskit_dynamics.perturbation.perturbative_solver import (
    PerturbativeSolver,
    construct_DCT,
    multi_interval_DCT,
    signal_envelope_DCT,
    signal_list_envelope_DCT,
    evaluate_cheb_series,
    evaluate_cheb_series_jax,
)

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit, grad
except ImportError:
    pass


class TestPerturbativeSolver(QiskitDynamicsTestCase):
    """Tests for perturbative solver."""

    @classmethod
    def setUpClass(cls):
        """Set up class."""
        cls.build_testing_objects(cls)

    @staticmethod
    def build_testing_objects(obj, integration_method="DOP853"):
        """Set up simple model parameters and solution to be used in multiple tests."""

        r = 0.2

        def gaussian(amp, sig, t0, t):
            return amp * np.exp(-((t - t0) ** 2) / (2 * sig ** 2))

        # specifications for generating envelope
        amp = 1.0  # amplitude
        sig = 0.399128 / r  # sigma
        t0 = 3.5 * sig  # center of Gaussian
        T = 7 * sig  # end of signal

        # Function to define gaussian envelope, using gaussian wave function
        gaussian_envelope = lambda t: gaussian(Array(amp), Array(sig), Array(t0), Array(t))

        obj.gauss_signal = Signal(gaussian_envelope, carrier_freq=5.0)

        obj.dt = 0.025
        obj.n_steps = int(T // obj.dt) // 3

        obj.hamiltonian_operators = 2 * np.pi * r * np.array([[[0.0, 1.0], [1.0, 0.0]]]) / 2
        obj.static_hamiltonian = 2 * np.pi * 5.0 * np.array([[1.0, 0.0], [0.0, -1.0]]) / 2

        reg_solver = Solver(
            static_hamiltonian=obj.static_hamiltonian,
            hamiltonian_operators=obj.hamiltonian_operators,
            rotating_frame=obj.static_hamiltonian,
            hamiltonian_signals=[obj.gauss_signal],
        )

        obj.simple_yf = reg_solver.solve(
            t_span=[0.0, obj.dt * obj.n_steps],
            y0=np.eye(2, dtype=complex),
            method=integration_method,
            atol=1e-12,
            rtol=1e-12,
        ).y[-1]

        obj.simple_dyson_solver = PerturbativeSolver(
            operators=-1j * obj.hamiltonian_operators,
            frame_operator=-1j * obj.static_hamiltonian,
            dt=obj.dt,
            carrier_freqs=[5.0],
            chebyshev_orders=[1],
            expansion_method="dyson",
            expansion_order=6,
            integration_method=integration_method,
            atol=1e-10,
            rtol=1e-10,
        )
        obj.simple_magnus_solver = PerturbativeSolver(
            operators=-1j * obj.hamiltonian_operators,
            frame_operator=-1j * obj.static_hamiltonian,
            dt=obj.dt,
            carrier_freqs=[5.0],
            chebyshev_orders=[1],
            expansion_method="magnus",
            expansion_order=3,
            integration_method=integration_method,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_dyson_solver(self):
        """Test dyson solver on a simple qubit model."""

        dyson_yf = self.simple_dyson_solver.solve(
            signals=[self.gauss_signal], y0=np.eye(2, dtype=complex), t0=0.0, n_steps=self.n_steps
        )

        self.assertAllClose(dyson_yf, self.simple_yf, rtol=1e-6, atol=1e-6)

    def test_magnus_solver(self):
        """Test magnus solver on a simple qubit model."""

        magnus_yf = self.simple_magnus_solver.solve(
            signals=[self.gauss_signal], y0=np.eye(2, dtype=complex), t0=0.0, n_steps=self.n_steps
        )

        self.assertAllClose(magnus_yf, self.simple_yf, rtol=1e-6, atol=1e-6)


class TestPerturbativeSolverJAX(TestJaxBase, TestPerturbativeSolver):
    """Tests for perturbative solver operating in JAX mode."""

    @classmethod
    def setUpClass(cls):
        # calls TestJaxBase setUpClass
        super().setUpClass()
        # builds common objects
        TestPerturbativeSolver.build_testing_objects(cls, integration_method="jax_odeint")

    def test_simple_dyson_solve_grad_jit(self):
        """Test jitting and gradding of dyson solve."""

        def func(c):
            dyson_yf = self.simple_dyson_solver.solve(
                signals=[Signal(Array(c), carrier_freq=5.0)],
                y0=np.eye(2, dtype=complex),
                t0=0.0,
                n_steps=self.n_steps,
            )
            return dyson_yf

        jitted_func = self.jit_wrap(func)
        self.assertAllClose(func(1.0), jitted_func(1.0))

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)

    def test_simple_magnus_solve_jit_grad(self):
        """Test jitting of and gradding of magnus solve."""

        def func(c):
            dyson_yf = self.simple_magnus_solver.solve(
                signals=[Signal(Array(c), carrier_freq=5.0)],
                y0=np.eye(2, dtype=complex),
                t0=0.0,
                n_steps=self.n_steps,
            )
            return dyson_yf

        jitted_func = self.jit_wrap(func)
        self.assertAllClose(func(1.0), jitted_func(1.0))

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)


class TestChebyshevFunctions(QiskitDynamicsTestCase):
    """Test cases for Chebyshev polynomial-related functions."""

    def setUp(self):
        """Set Chebyshev evaluation function."""

        self.cheb_eval_func = evaluate_cheb_series

    def test_construct_DCT(self):
        """Verify consistency with numpy functions."""

        f = lambda t: 1.0 + t ** 2 + t ** 3

        expected = Chebyshev.interpolate(f, deg=2)
        M, x_vals = construct_DCT(degree=2)

        self.assertAllClose(expected.coef, M @ f(x_vals))

    def test_multinterval_DCT(self):
        """Verify consistency with numpy functions."""

        dt = 0.1212
        t0 = 2.13214
        int0 = [t0, t0 + dt]
        int1 = [t0 + dt, t0 + 2 * dt]
        int2 = [t0 + 2 * dt, t0 + 3 * dt]

        f = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(Array(3.123) * t)
        multi_int_coeffs = multi_interval_DCT(f, degree=4, t0=t0, dt=dt, n_intervals=3)

        # force to resolve to numpy arrays for comparison to numpy functions
        f = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(3.123 * t)

        # check correctness over each interval
        expected = Chebyshev.interpolate(f, deg=4, domain=int0)
        self.assertAllClose(expected.coef, multi_int_coeffs[:, 0])

        expected = Chebyshev.interpolate(f, deg=4, domain=int1)
        self.assertAllClose(expected.coef, multi_int_coeffs[:, 1])

        expected = Chebyshev.interpolate(f, deg=4, domain=int2)
        self.assertAllClose(expected.coef, multi_int_coeffs[:, 2])

    def test_signal_envelope_DCT(self):
        """Test correct approximation of a signal envelope relative to a reference
        frequency.
        """
        dt = 0.1212
        t0 = 2.13214
        t1 = t0 + dt
        t2 = t1 + dt
        t3 = t2 + dt
        int0 = [t0, t1]
        int1 = [t1, t2]
        int2 = [t2, t3]

        f = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(Array(3.123) * t)
        carrier_freq = 1.0
        reference_freq = 0.23
        signal = Signal(f, carrier_freq)
        env_dct = signal_envelope_DCT(
            signal, reference_freq=reference_freq, degree=5, t0=t0, dt=dt, n_intervals=3
        )

        # construct pure numpy comparison function
        f = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(3.123 * t)
        carrier_phase_arg = 1j * 2 * np.pi * carrier_freq
        ref_phase_arg = -1j * 2 * np.pi * reference_freq
        final_phase_shift = np.exp(-ref_phase_arg * np.array([t0, t1, t2]))
        shifted_env = lambda t: f(t) * np.exp((carrier_phase_arg + ref_phase_arg) * t)

        # check correctness over each interval
        expected = Chebyshev.interpolate(shifted_env, deg=5, domain=int0)
        self.assertAllClose(expected.coef * final_phase_shift[0], env_dct[:, 0])

        expected = Chebyshev.interpolate(shifted_env, deg=5, domain=int1)
        self.assertAllClose(expected.coef * final_phase_shift[1], env_dct[:, 1])

        expected = Chebyshev.interpolate(shifted_env, deg=5, domain=int2)
        self.assertAllClose(expected.coef * final_phase_shift[2], env_dct[:, 2])

    def test_signal_list_envelope_DCT(self):
        """Test correct approximation of the envelopes of a list of signals
        with reference frequencies.
        """

        dt = 0.1212
        t0 = 2.13214
        t1 = t0 + dt
        t2 = t1 + dt
        t3 = t2 + dt
        int0 = [t0, t1]
        int1 = [t1, t2]
        int2 = [t2, t3]

        f1 = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(Array(3.123) * t)
        carrier_freq1 = 1.0
        reference_freq1 = 0.23
        signal1 = Signal(f1, carrier_freq1)

        f2 = lambda t: 2.1 + t ** 2 + t ** 4 + np.cos(Array(3.123) * t)
        carrier_freq2 = 2.0
        reference_freq2 = 1.1
        signal2 = Signal(f2, carrier_freq2)

        list_dct = signal_list_envelope_DCT(
            [signal1, signal2],
            [reference_freq1, reference_freq2],
            degrees=[3, 4],
            t0=t0,
            dt=dt,
            n_intervals=3,
        )

        f1 = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(3.123 * t)
        carrier_phase_arg1 = 1j * 2 * np.pi * carrier_freq1
        ref_phase_arg1 = -1j * 2 * np.pi * reference_freq1
        final_phase_shift1 = np.exp(-ref_phase_arg1 * np.array([t0, t1, t2]))
        shifted_env1 = lambda t: f1(t) * np.exp((carrier_phase_arg1 + ref_phase_arg1) * t)

        f2 = lambda t: 2.1 + t ** 2 + t ** 4 + np.cos(3.123 * t)
        carrier_phase_arg2 = 1j * 2 * np.pi * carrier_freq2
        ref_phase_arg2 = -1j * 2 * np.pi * reference_freq2
        final_phase_shift2 = np.exp(-ref_phase_arg2 * np.array([t0, t1, t2]))
        shifted_env2 = lambda t: f2(t) * np.exp((carrier_phase_arg2 + ref_phase_arg2) * t)

        # check correctness for first signal
        # each one is broken into the real and imaginary part
        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int0)
        coeffs = expected.coef * final_phase_shift1[0]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[:8, 0])

        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int1)
        coeffs = expected.coef * final_phase_shift1[1]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[:8, 1])

        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int2)
        coeffs = expected.coef * final_phase_shift1[2]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[:8, 2])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int0)
        coeffs = expected.coef * final_phase_shift2[0]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[8:, 0])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int1)
        coeffs = expected.coef * final_phase_shift2[1]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[8:, 1])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int2)
        coeffs = expected.coef * final_phase_shift2[2]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[8:, 2])

    def test_signal_list_envelope_DCT_include_imag_case1(self):
        """Test correct approximation of the envelopes of a list of signals
        with reference frequencies, including an instruction to not include
        the imaginary component for a signal.
        """

        dt = 0.1212
        t0 = 2.13214
        t1 = t0 + dt
        t2 = t1 + dt
        t3 = t2 + dt
        int0 = [t0, t1]
        int1 = [t1, t2]
        int2 = [t2, t3]

        f1 = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(Array(3.123) * t)
        carrier_freq1 = 1.0
        reference_freq1 = 0.23
        signal1 = Signal(f1, carrier_freq1)

        f2 = lambda t: 2.1 + t ** 2 + t ** 4 + np.cos(Array(3.123) * t)
        carrier_freq2 = 2.0
        reference_freq2 = 1.1
        signal2 = Signal(f2, carrier_freq2)

        list_dct = signal_list_envelope_DCT(
            [signal1, signal2],
            [reference_freq1, reference_freq2],
            degrees=[3, 4],
            t0=t0,
            dt=dt,
            n_intervals=3,
            include_imag=[False, True],
        )

        f1 = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(3.123 * t)
        carrier_phase_arg1 = 1j * 2 * np.pi * carrier_freq1
        ref_phase_arg1 = -1j * 2 * np.pi * reference_freq1
        final_phase_shift1 = np.exp(-ref_phase_arg1 * np.array([t0, t1, t2]))
        shifted_env1 = lambda t: f1(t) * np.exp((carrier_phase_arg1 + ref_phase_arg1) * t)

        f2 = lambda t: 2.1 + t ** 2 + t ** 4 + np.cos(3.123 * t)
        carrier_phase_arg2 = 1j * 2 * np.pi * carrier_freq2
        ref_phase_arg2 = -1j * 2 * np.pi * reference_freq2
        final_phase_shift2 = np.exp(-ref_phase_arg2 * np.array([t0, t1, t2]))
        shifted_env2 = lambda t: f2(t) * np.exp((carrier_phase_arg2 + ref_phase_arg2) * t)

        # check correctness for first signal
        # each one is broken into the real and imaginary part
        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int0)
        coeffs = expected.coef * final_phase_shift1[0]
        coeffs = coeffs.real
        self.assertAllClose(coeffs, list_dct[:4, 0])

        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int1)
        coeffs = expected.coef * final_phase_shift1[1]
        coeffs = coeffs.real
        self.assertAllClose(coeffs, list_dct[:4, 1])

        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int2)
        coeffs = expected.coef * final_phase_shift1[2]
        coeffs = coeffs.real
        self.assertAllClose(coeffs, list_dct[:4, 2])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int0)
        coeffs = expected.coef * final_phase_shift2[0]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[4:, 0])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int1)
        coeffs = expected.coef * final_phase_shift2[1]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[4:, 1])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int2)
        coeffs = expected.coef * final_phase_shift2[2]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[4:, 2])

    def test_signal_list_envelope_DCT_include_imag_case2(self):
        """Test correct approximation of the envelopes of a list of signals
        with reference frequencies, including an instruction to not include
        the imaginary component for a signal.
        """

        dt = 0.1212
        t0 = 2.13214
        t1 = t0 + dt
        t2 = t1 + dt
        t3 = t2 + dt
        int0 = [t0, t1]
        int1 = [t1, t2]
        int2 = [t2, t3]

        f1 = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(Array(3.123) * t)
        carrier_freq1 = 1.0
        reference_freq1 = 0.23
        signal1 = Signal(f1, carrier_freq1)

        f2 = lambda t: 2.1 + t ** 2 + t ** 4 + np.cos(Array(3.123) * t)
        carrier_freq2 = 2.0
        reference_freq2 = 1.1
        signal2 = Signal(f2, carrier_freq2)

        list_dct = signal_list_envelope_DCT(
            [signal1, signal2],
            [reference_freq1, reference_freq2],
            degrees=[3, 4],
            t0=t0,
            dt=dt,
            n_intervals=3,
            include_imag=[True, False],
        )

        f1 = lambda t: 1.0 + t ** 2 + t ** 3 + np.sin(3.123 * t)
        carrier_phase_arg1 = 1j * 2 * np.pi * carrier_freq1
        ref_phase_arg1 = -1j * 2 * np.pi * reference_freq1
        final_phase_shift1 = np.exp(-ref_phase_arg1 * np.array([t0, t1, t2]))
        shifted_env1 = lambda t: f1(t) * np.exp((carrier_phase_arg1 + ref_phase_arg1) * t)

        f2 = lambda t: 2.1 + t ** 2 + t ** 4 + np.cos(3.123 * t)
        carrier_phase_arg2 = 1j * 2 * np.pi * carrier_freq2
        ref_phase_arg2 = -1j * 2 * np.pi * reference_freq2
        final_phase_shift2 = np.exp(-ref_phase_arg2 * np.array([t0, t1, t2]))
        shifted_env2 = lambda t: f2(t) * np.exp((carrier_phase_arg2 + ref_phase_arg2) * t)

        # check correctness for first signal
        # each one is broken into the real and imaginary part
        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int0)
        coeffs = expected.coef * final_phase_shift1[0]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[:8, 0])

        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int1)
        coeffs = expected.coef * final_phase_shift1[1]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[:8, 1])

        expected = Chebyshev.interpolate(shifted_env1, deg=3, domain=int2)
        coeffs = expected.coef * final_phase_shift1[2]
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
        self.assertAllClose(coeffs, list_dct[:8, 2])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int0)
        coeffs = expected.coef * final_phase_shift2[0]
        coeffs = coeffs.real
        self.assertAllClose(coeffs, list_dct[8:, 0])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int1)
        coeffs = expected.coef * final_phase_shift2[1]
        coeffs = coeffs.real
        self.assertAllClose(coeffs, list_dct[8:, 1])

        expected = Chebyshev.interpolate(shifted_env2, deg=4, domain=int2)
        coeffs = expected.coef * final_phase_shift2[2]
        coeffs = coeffs.real
        self.assertAllClose(coeffs, list_dct[8:, 2])

    def test_evaluate_cheb_series_case1(self):
        """Test Chebyshev evaluation function, linear case."""

        coeffs = np.array([0.231, 1.1])
        x = np.array([0.2, 0.4, 1.5])
        domain = [0.0, 4.0]

        expected = Chebyshev(coef=coeffs, domain=domain)(x)
        output = self.cheb_eval_func(x, coeffs, domain)

        self.assertAllClose(expected, output)

    def test_evaluate_cheb_series_case2(self):
        """Test Chebyshev evaluation function, higher order case."""

        coeffs = np.array([0.231, 1.1, 2.1, 3.0])
        x = np.array([0.2, 0.4, 1.5])
        domain = [0.0, 4.0]

        expected = Chebyshev(coef=coeffs, domain=domain)(x)
        output = self.cheb_eval_func(x, coeffs, domain)

        self.assertAllClose(expected, output)

    def test_evaluate_cheb_series_case3(self):
        """Test Chebyshev evaluation function, non-vectorized."""

        coeffs = np.array([0.231, 1.1, 2.1, 3.0, 5.1, 2.2])
        x = 0.4
        domain = [0.0, 4.0]

        expected = Chebyshev(coef=coeffs, domain=domain)(x)
        output = self.cheb_eval_func(x, coeffs, domain)

        self.assertAllClose(expected, output)


class TestChebyshevFunctionsJax(TestChebyshevFunctions, TestJaxBase):
    """JAX version of TestChebyshevFunctions."""

    def setUp(self):
        """Set Chebyshev evaluation function."""

        self.cheb_eval_func = evaluate_cheb_series_jax

    def test_evaluate_cheb_series_jit(self):
        """Test jitting of evaluate_cheb_series_jax."""

        coeffs = np.array([0.231, 1.1, 2.1, 3.0])
        domain = [0.0, 4.0]

        jit_func = jit(lambda x: self.cheb_eval_func(x, coeffs, domain))

        x = np.array([0.2, 0.4, 1.5])

        expected = Chebyshev(coef=coeffs, domain=domain)(x)
        output = jit_func(x)
        self.assertAllClose(expected, output)

    def test_evaluate_cheb_series_jit_grad(self):
        """Test jitting of grad of evaluate_cheb_series_jax."""

        coeffs = np.array([0.231, 1.1, 2.1, 3.0])
        domain = [0.0, 4.0]

        jit_grad_func = jit(grad(lambda x: self.cheb_eval_func(x, coeffs, domain).sum()))

        x = np.array([0.2, 0.4, 1.5])

        jit_grad_func(x)

    def test_jit_grad_through_DCT(self):
        """Test jitting and grad through a DCT."""

        def func(a):
            sig = Signal(lambda t: a * t, carrier_freq=1.0)
            return signal_list_envelope_DCT(
                [sig], reference_freqs=[1.0], degrees=[2], t0=0.0, dt=0.5, n_intervals=3
            ).data

        jit_func = jit(func)
        self.assertAllClose(jit_func(1.0), func(1.0))

        def func2(a):
            return func(a).sum()

        jit_grad_func = jit(grad(func2))
        jit_grad_func(1.0)