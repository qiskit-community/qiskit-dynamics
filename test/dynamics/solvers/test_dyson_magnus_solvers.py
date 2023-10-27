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

"""Tests for perturbative solvers."""

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from ddt import ddt, data, unpack

from qiskit import QiskitError

from qiskit_dynamics import Signal, Solver, DysonSolver, MagnusSolver
from qiskit_dynamics.array import Array
from qiskit_dynamics import DYNAMICS_NUMPY as unp

from qiskit_dynamics.solvers.perturbative_solvers.expansion_model import (
    _construct_DCT,
    _multi_interval_DCT,
    _signal_envelope_DCT,
    _signal_list_envelope_DCT,
    _evaluate_cheb_series,
    _evaluate_cheb_series_jax,
)

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit, grad
except ImportError:
    pass


@ddt
class Test_DysonMagnusSolver_Validation(QiskitDynamicsTestCase):
    """Validation tests for DysonSolver and MagnusSolver."""

    @unpack
    @data((DysonSolver,), (MagnusSolver,))
    def test_invalid_carrier_freqs(self, SolverClass):
        """Test error is raised if carrier_freqs length doesn't match operators length."""
        with self.assertRaisesRegex(QiskitError, "carrier_freqs must have the same length"):
            SolverClass(
                operators=np.array([[[1.0]], [[2.0]]]),
                rotating_frame=np.array([[1.0]]),
                dt=1.0,
                carrier_freqs=np.array([1.0]),
                chebyshev_orders=np.array([1, 1]),
            )

    @unpack
    @data((DysonSolver,), (MagnusSolver,))
    def test_invalid_chebyshev_orders(self, SolverClass):
        """Test error is raised if chebyshev_orders length doesn't match operators length."""
        with self.assertRaisesRegex(QiskitError, "chebyshev_orders must have the same length"):
            SolverClass(
                operators=np.array([[[1.0]], [[2.0]]]),
                rotating_frame=np.array([[1.0]]),
                dt=1.0,
                carrier_freqs=np.array([1.0, 1.0]),
                chebyshev_orders=np.array([1, 1, 1]),
            )


class Test_PerturbativeSolver(QiskitDynamicsTestCase):
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
            return amp * unp.exp(-((t - t0) ** 2) / (2 * sig**2))

        # specifications for generating envelope
        amp = 1.0  # amplitude
        sig = 0.399128 / r  # sigma
        t0 = 3.5 * sig  # center of Gaussian
        T = 7 * sig  # end of signal

        # Function to define gaussian envelope, using gaussian wave function
        gaussian_envelope = lambda t: gaussian(amp, sig, t0, t)

        obj.gauss_signal = Signal(gaussian_envelope, carrier_freq=5.0)

        dt = 0.0125
        obj.n_steps = int(T // dt) // 3
        hamiltonian_operators = 2 * np.pi * r * np.array([[[0.0, 1.0], [1.0, 0.0]]]) / 2
        static_hamiltonian = 2 * np.pi * 5.0 * np.array([[1.0, 0.0], [0.0, -1.0]]) / 2

        reg_solver = Solver(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            rotating_frame=static_hamiltonian,
        )

        obj.simple_yf = reg_solver.solve(
            t_span=[0.0, dt * obj.n_steps],
            y0=np.eye(2, dtype=complex),
            signals=[obj.gauss_signal],
            method=integration_method,
            atol=1e-12,
            rtol=1e-12,
        ).y[-1]

        obj.simple_dyson_solver = DysonSolver(
            operators=-1j * hamiltonian_operators,
            rotating_frame=-1j * static_hamiltonian,
            dt=dt,
            carrier_freqs=[5.0],
            chebyshev_orders=[1],
            expansion_order=6,
            integration_method=integration_method,
            atol=1e-10,
            rtol=1e-10,
        )
        obj.simple_magnus_solver = MagnusSolver(
            operators=-1j * hamiltonian_operators,
            rotating_frame=-1j * static_hamiltonian,
            dt=dt,
            carrier_freqs=[5.0],
            chebyshev_orders=[1],
            expansion_order=3,
            integration_method=integration_method,
            atol=1e-10,
            rtol=1e-10,
        )

        # set up more complicated two transmon example
        w_c = 2 * np.pi * 5.033
        w_t = 2 * np.pi * 4.067
        alpha_c = 2 * np.pi * (-0.33534)
        alpha_t = 2 * np.pi * (-0.33834)
        J = 2 * np.pi * 0.002

        dim = 5
        obj.dim_2q = dim

        a = np.diag(np.sqrt(np.arange(1, dim)), 1)
        adag = a.transpose()
        N = np.diag(np.arange(dim))
        ident = np.eye(dim)
        ident2 = np.eye(dim**2)

        # operators on the control qubit (first tensor factor)
        a0 = np.kron(a, ident)
        adag0 = np.kron(adag, ident)
        N0 = np.kron(N, ident)

        # operators on the target qubit (first tensor factor)
        a1 = np.kron(ident, a)
        adag1 = np.kron(ident, adag)
        N1 = np.kron(ident, N)

        H0 = (
            w_c * N0
            + 0.5 * alpha_c * N0 @ (N0 - ident2)
            + w_t * N1
            + 0.5 * alpha_t * N1 @ (N1 - ident2)
            + J * (a0 @ adag1 + adag0 @ a1)
        )
        Hdc = 2 * np.pi * (a0 + adag0)
        Hdt = 2 * np.pi * (a1 + adag1)

        dense_solver = Solver(
            static_hamiltonian=H0,
            hamiltonian_operators=[Hdc, Hdt],
            rotating_frame=H0,
        )

        T = 10.0
        dt = 0.01
        obj.n_steps_2q = int(T // dt)

        obj.yf_2q = dense_solver.solve(
            t_span=[0.0, dt * obj.n_steps_2q],
            y0=np.eye(dim**2, dtype=complex),
            signals=[obj.gauss_signal, obj.gauss_signal],
            method=integration_method,
            atol=1e-12,
            rtol=1e-12,
        ).y[-1]

        obj.dyson_solver_2q = DysonSolver(
            operators=[-1j * Hdc, -1j * Hdt],
            rotating_frame=-1j * H0,
            dt=dt,
            carrier_freqs=[5.0, 5.0],
            chebyshev_orders=[1, 1],
            expansion_order=6,
            integration_method=integration_method,
            atol=1e-10,
            rtol=1e-10,
        )
        obj.dyson_solver_2q_0_carrier = DysonSolver(
            operators=[-1j * Hdc, -1j * Hdt],
            rotating_frame=-1j * H0,
            dt=dt,
            carrier_freqs=[0.0, 0.0],
            chebyshev_orders=[3, 3],
            expansion_order=6,
            integration_method=integration_method,
            include_imag=[False, False],
            atol=1e-10,
            rtol=1e-10,
        )
        obj.magnus_solver_2q = MagnusSolver(
            operators=[-1j * Hdc, -1j * Hdt],
            rotating_frame=-1j * H0,
            dt=dt,
            carrier_freqs=[5.0, 5.0],
            chebyshev_orders=[1, 1],
            expansion_order=3,
            integration_method=integration_method,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_signal_length_validation(self):
        """Test correct validation of signal length."""
        with self.assertRaisesRegex(QiskitError, "must be the same length"):
            # pylint: disable=expression-not-assigned
            self.simple_dyson_solver.solve(
                t0=0.0,
                n_steps=self.n_steps,
                y0=np.eye(2, dtype=complex),
                signals=[self.gauss_signal, self.gauss_signal],
            )

    def test_simple_dyson_solver(self):
        """Test dyson solver on a simple qubit model."""

        dyson_yf = self.simple_dyson_solver.solve(
            t0=0.0, n_steps=self.n_steps, y0=np.eye(2, dtype=complex), signals=[self.gauss_signal]
        ).y[-1]

        self.assertAllClose(dyson_yf, self.simple_yf, rtol=1e-6, atol=1e-6)

    def test_simple_magnus_solver(self):
        """Test magnus solver on a simple qubit model."""

        magnus_yf = self.simple_magnus_solver.solve(
            t0=0.0, n_steps=self.n_steps, y0=np.eye(2, dtype=complex), signals=[self.gauss_signal]
        ).y[-1]

        self.assertAllClose(magnus_yf, self.simple_yf, rtol=1e-6, atol=1e-6)

    def test_dyson_solver_2q(self):
        """Test dyson solver on a two transmon model."""

        dyson_yf = self.dyson_solver_2q.solve(
            t0=0.0,
            n_steps=self.n_steps_2q,
            y0=np.eye(self.dim_2q**2, dtype=complex),
            signals=[self.gauss_signal, self.gauss_signal],
        ).y[-1]
        # measure similarity with fidelity
        self.assertTrue(
            np.abs(1.0 - np.abs((dyson_yf.conj() * self.yf_2q).sum()) ** 2 / (self.dim_2q**4))
            < 1e-6
        )

    def test_dyson_solver_2q_0_carrier(self):
        """Test dyson solver on a two transmon model."""

        dyson_yf = self.dyson_solver_2q_0_carrier.solve(
            t0=0.0,
            n_steps=self.n_steps_2q,
            y0=np.eye(self.dim_2q**2, dtype=complex),
            signals=[self.gauss_signal, self.gauss_signal],
        ).y[-1]

        # measure similarity with fidelity
        self.assertTrue(
            np.abs(1.0 - np.abs((dyson_yf.conj() * self.yf_2q).sum()) ** 2 / (self.dim_2q**4))
            < 1e-6
        )

    def test_magnus_solver_2q(self):
        """Test magnus solver on a two transmon model."""

        magnus_yf = self.magnus_solver_2q.solve(
            t0=0.0,
            n_steps=self.n_steps_2q,
            y0=np.eye(self.dim_2q**2, dtype=complex),
            signals=[self.gauss_signal, self.gauss_signal],
        ).y[-1]
        # measure similarity with fidelity
        self.assertTrue(
            np.abs(1.0 - np.abs((magnus_yf.conj() * self.yf_2q).sum()) ** 2 / (self.dim_2q**4))
            < 1e-6
        )

    def test_list_simulation(self):
        """Test running lists of simulations."""

        rng = np.random.default_rng(21342)
        y00 = rng.uniform(low=-1, high=1, size=(2, 2)) + 1j * rng.uniform(
            low=-1, high=1, size=(2, 2)
        )
        y01 = rng.uniform(low=-1, high=1, size=(2, 2)) + 1j * rng.uniform(
            low=-1, high=1, size=(2, 2)
        )

        dyson_results = self.simple_dyson_solver.solve(
            t0=0.0, n_steps=self.n_steps, y0=[y00, y01], signals=[self.gauss_signal]
        )

        self.assertAllClose(dyson_results[0].y[-1], self.simple_yf @ y00, rtol=1e-6, atol=1e-6)
        self.assertAllClose(dyson_results[1].y[-1], self.simple_yf @ y01, rtol=1e-6, atol=1e-6)


class Test_PerturbativeSolverJAX(TestJaxBase, Test_PerturbativeSolver):
    """Tests for perturbative solver operating in JAX mode."""

    @classmethod
    def setUpClass(cls):
        # calls TestJaxBase setUpClass
        super().setUpClass()
        # builds common objects
        Test_PerturbativeSolver.build_testing_objects(cls, integration_method="jax_odeint")

    def test_simple_dyson_solve_grad_jit(self):
        """Test jitting and gradding of dyson solve."""

        def func(c):
            dyson_yf = self.simple_dyson_solver.solve(
                t0=0.0,
                n_steps=self.n_steps,
                y0=np.eye(2, dtype=complex),
                signals=[Signal(c, carrier_freq=5.0)],
            ).y[-1]
            return dyson_yf

        jitted_func = self.jit_wrap(func)
        self.assertAllClose(func(1.0), jitted_func(1.0))

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)

    def test_simple_magnus_solve_jit_grad(self):
        """Test jitting of and gradding of magnus solve."""

        def func(c):
            magnus_yf = self.simple_magnus_solver.solve(
                t0=0.0,
                n_steps=self.n_steps,
                y0=np.eye(2, dtype=complex),
                signals=[Signal(c, carrier_freq=5.0)],
            ).y[-1]
            return magnus_yf

        jitted_func = self.jit_wrap(func)
        self.assertAllClose(func(1.0), jitted_func(1.0))

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)


class TestChebyshevFunctions(QiskitDynamicsTestCase):
    """Test cases for Chebyshev polynomial-related functions."""

    def setUp(self):
        """Set Chebyshev evaluation function."""

        self.cheb_eval_func = _evaluate_cheb_series

    def test__construct_DCT(self):
        """Verify consistency with numpy functions."""

        f = lambda t: 1.0 + t**2 + t**3

        expected = Chebyshev.interpolate(f, deg=2)
        M, x_vals = _construct_DCT(degree=2)

        self.assertAllClose(expected.coef, M @ f(x_vals))

    def test_multinterval_DCT(self):
        """Verify consistency with numpy functions."""

        dt = 0.1212
        t0 = 2.13214
        int0 = [t0, t0 + dt]
        int1 = [t0 + dt, t0 + 2 * dt]
        int2 = [t0 + 2 * dt, t0 + 3 * dt]

        f = lambda t: 1.0 + t**2 + t**3 + np.sin(Array(3.123) * t)
        multi_int_coeffs = _multi_interval_DCT(f, degree=4, t0=t0, dt=dt, n_intervals=3)

        # force to resolve to numpy arrays for comparison to numpy functions
        f = lambda t: 1.0 + t**2 + t**3 + np.sin(3.123 * t)

        # check correctness over each interval
        expected = Chebyshev.interpolate(f, deg=4, domain=int0)
        self.assertAllClose(expected.coef, multi_int_coeffs[:, 0])

        expected = Chebyshev.interpolate(f, deg=4, domain=int1)
        self.assertAllClose(expected.coef, multi_int_coeffs[:, 1])

        expected = Chebyshev.interpolate(f, deg=4, domain=int2)
        self.assertAllClose(expected.coef, multi_int_coeffs[:, 2])

    def test__signal_envelope_DCT(self):
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

        f = lambda t: 1.0 + t**2 + t**3 + np.sin(Array(3.123) * t)
        carrier_freq = 1.0
        reference_freq = 0.23
        signal = Signal(f, carrier_freq)
        env_dct = _signal_envelope_DCT(
            signal, reference_freq=reference_freq, degree=5, t0=t0, dt=dt, n_intervals=3
        )

        # construct pure numpy comparison function
        f = lambda t: 1.0 + t**2 + t**3 + np.sin(3.123 * t)
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

    def test__signal_list_envelope_DCT(self):
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

        f1 = lambda t: 1.0 + t**2 + t**3 + np.sin(Array(3.123) * t)
        carrier_freq1 = 1.0
        reference_freq1 = 0.23
        signal1 = Signal(f1, carrier_freq1)

        f2 = lambda t: 2.1 + t**2 + t**4 + np.cos(Array(3.123) * t)
        carrier_freq2 = 2.0
        reference_freq2 = 1.1
        signal2 = Signal(f2, carrier_freq2)

        list_dct = _signal_list_envelope_DCT(
            [signal1, signal2],
            [reference_freq1, reference_freq2],
            degrees=[3, 4],
            t0=t0,
            dt=dt,
            n_intervals=3,
        )

        f1 = lambda t: 1.0 + t**2 + t**3 + np.sin(3.123 * t)
        carrier_phase_arg1 = 1j * 2 * np.pi * carrier_freq1
        ref_phase_arg1 = -1j * 2 * np.pi * reference_freq1
        final_phase_shift1 = np.exp(-ref_phase_arg1 * np.array([t0, t1, t2]))
        shifted_env1 = lambda t: f1(t) * np.exp((carrier_phase_arg1 + ref_phase_arg1) * t)

        f2 = lambda t: 2.1 + t**2 + t**4 + np.cos(3.123 * t)
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

    def test__signal_list_envelope_DCT_include_imag_case1(self):
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

        f1 = lambda t: 1.0 + t**2 + t**3 + np.sin(Array(3.123) * t)
        carrier_freq1 = 1.0
        reference_freq1 = 0.23
        signal1 = Signal(f1, carrier_freq1)

        f2 = lambda t: 2.1 + t**2 + t**4 + np.cos(Array(3.123) * t)
        carrier_freq2 = 2.0
        reference_freq2 = 1.1
        signal2 = Signal(f2, carrier_freq2)

        list_dct = _signal_list_envelope_DCT(
            [signal1, signal2],
            [reference_freq1, reference_freq2],
            degrees=[3, 4],
            t0=t0,
            dt=dt,
            n_intervals=3,
            include_imag=[False, True],
        )

        f1 = lambda t: 1.0 + t**2 + t**3 + np.sin(3.123 * t)
        carrier_phase_arg1 = 1j * 2 * np.pi * carrier_freq1
        ref_phase_arg1 = -1j * 2 * np.pi * reference_freq1
        final_phase_shift1 = np.exp(-ref_phase_arg1 * np.array([t0, t1, t2]))
        shifted_env1 = lambda t: f1(t) * np.exp((carrier_phase_arg1 + ref_phase_arg1) * t)

        f2 = lambda t: 2.1 + t**2 + t**4 + np.cos(3.123 * t)
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

    def test__signal_list_envelope_DCT_include_imag_case2(self):
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

        f1 = lambda t: 1.0 + t**2 + t**3 + np.sin(Array(3.123) * t)
        carrier_freq1 = 1.0
        reference_freq1 = 0.23
        signal1 = Signal(f1, carrier_freq1)

        f2 = lambda t: 2.1 + t**2 + t**4 + np.cos(Array(3.123) * t)
        carrier_freq2 = 2.0
        reference_freq2 = 1.1
        signal2 = Signal(f2, carrier_freq2)

        list_dct = _signal_list_envelope_DCT(
            [signal1, signal2],
            [reference_freq1, reference_freq2],
            degrees=[3, 4],
            t0=t0,
            dt=dt,
            n_intervals=3,
            include_imag=[True, False],
        )

        f1 = lambda t: 1.0 + t**2 + t**3 + np.sin(3.123 * t)
        carrier_phase_arg1 = 1j * 2 * np.pi * carrier_freq1
        ref_phase_arg1 = -1j * 2 * np.pi * reference_freq1
        final_phase_shift1 = np.exp(-ref_phase_arg1 * np.array([t0, t1, t2]))
        shifted_env1 = lambda t: f1(t) * np.exp((carrier_phase_arg1 + ref_phase_arg1) * t)

        f2 = lambda t: 2.1 + t**2 + t**4 + np.cos(3.123 * t)
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

    def test__evaluate_cheb_series_case1(self):
        """Test Chebyshev evaluation function, linear case."""

        coeffs = np.array([0.231, 1.1])
        x = np.array([0.2, 0.4, 1.5])
        domain = [0.0, 4.0]

        expected = Chebyshev(coef=coeffs, domain=domain)(x)
        output = self.cheb_eval_func(x, coeffs, domain)

        self.assertAllClose(expected, output)

    def test__evaluate_cheb_series_case2(self):
        """Test Chebyshev evaluation function, higher order case."""

        coeffs = np.array([0.231, 1.1, 2.1, 3.0])
        x = np.array([0.2, 0.4, 1.5])
        domain = [0.0, 4.0]

        expected = Chebyshev(coef=coeffs, domain=domain)(x)
        output = self.cheb_eval_func(x, coeffs, domain)

        self.assertAllClose(expected, output)

    def test__evaluate_cheb_series_case3(self):
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

        self.cheb_eval_func = _evaluate_cheb_series_jax

    def test__evaluate_cheb_series_jit(self):
        """Test jitting of _evaluate_cheb_series_jax."""

        coeffs = np.array([0.231, 1.1, 2.1, 3.0])
        domain = [0.0, 4.0]

        jit_func = jit(lambda x: self.cheb_eval_func(x, coeffs, domain))

        x = np.array([0.2, 0.4, 1.5])

        expected = Chebyshev(coef=coeffs, domain=domain)(x)
        output = jit_func(x)
        self.assertAllClose(expected, output)

    def test__evaluate_cheb_series_jit_grad(self):
        """Test jitting of grad of _evaluate_cheb_series_jax."""

        coeffs = np.array([0.231, 1.1, 2.1, 3.0])
        domain = [0.0, 4.0]

        jit_grad_func = jit(grad(lambda x: self.cheb_eval_func(x, coeffs, domain).sum()))

        x = np.array([0.2, 0.4, 1.5])

        jit_grad_func(x)

    def test_jit_grad_through_DCT(self):
        """Test jitting and grad through a DCT."""

        def func(a):
            sig = Signal(lambda t: a * t, carrier_freq=1.0)
            return _signal_list_envelope_DCT(
                [sig], reference_freqs=[1.0], degrees=[2], t0=0.0, dt=0.5, n_intervals=3
            ).data

        jit_func = jit(func)
        self.assertAllClose(jit_func(1.0), func(1.0))

        def func2(a):
            return func(a).sum()

        jit_grad_func = jit(grad(func2))
        jit_grad_func(1.0)
