# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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
Tests for solver classes module.
"""

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info import Operator, Statevector, SuperOp, DensityMatrix

from qiskit_dynamics import Solver
from qiskit_dynamics.signals import Signal
from qiskit_dynamics.array import Array
from qiskit_dynamics.type_utils import to_array

from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestSolverValidation(QiskitDynamicsTestCase):
    """Test validation for Hamiltonian terms."""

    def test_hamiltonian_operators_array_not_hermitian(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]

        with self.assertRaises(QiskitError) as qe:
            Solver(hamiltonian_operators=operators)
        self.assertTrue("operators must be Hermitian." in str(qe.exception))

    def test_validation_override_hamiltonian(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]
        solver = Solver(hamiltonian_operators=operators, validate=False)

        self.assertAllClose(solver.model.operators, operators)

    def test_hamiltonian_operators_array_not_hermitian_lindblad(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]

        with self.assertRaises(QiskitError) as qe:
            Solver(hamiltonian_operators=operators, static_dissipators=operators)
        self.assertTrue("operators must be Hermitian." in str(qe.exception))

    def test_validation_override_lindblad(self):
        """Test raising error if operators are not Hermitian."""

        operators = [np.array([[0.0, 1.0], [0.0, 0.0]])]
        solver = Solver(
            hamiltonian_operators=operators, static_dissipators=operators, validate=False
        )

        self.assertAllClose(solver.model.hamiltonian_operators, operators)


class TestSolverExceptions(QiskitDynamicsTestCase):
    """Tests for Solver exception raising based on input types."""

    def setUp(self):
        X = Operator.from_label("X")
        self.ham_solver = Solver(hamiltonian_operators=[X], hamiltonian_signals=[1.0])

        self.lindblad_solver = Solver(
            hamiltonian_operators=[X], hamiltonian_signals=[1.0], static_dissipators=[X]
        )

        self.vec_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[1.0],
            static_dissipators=[X],
            evaluation_mode="dense_vectorized",
        )

    def test_hamiltonian_shape_error(self):
        """Test error raising if invalid shape for Hamiltonian model."""

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve(signals=None, t_span=[0.0, 1.0], y0=np.array([1.0, 0.0, 0.0]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve(signals=None, t_span=[0.0, 1.0], y0=np.array([[[1.0, 0.0, 0.0]]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve(signals=None, t_span=[0.0, 1.0], y0=Statevector(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_lindblad_shape_error(self):
        """Test error raising if invalid shape for Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve(signals=None, t_span=[0.0, 1.0], y0=np.array([1.0, 0.0, 0.0]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve(signals=None, t_span=[0.0, 1.0], y0=np.array([[[1.0, 0.0, 0.0]]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve(signals=None, t_span=[0.0, 1.0], y0=Statevector(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_vectorized_lindblad_shape_error(self):
        """Test error raising if invalid shape for vectorized Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve(signals=None, t_span=[0.0, 1.0], y0=np.array([[1.0, 0.0], [0.0, 1.0]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve(signals=None, t_span=[0.0, 1.0], y0=DensityMatrix(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve(signals=None, t_span=[0.0, 1.0], y0=Statevector(np.array([1.0, 0.0, 0.0])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_non_vectorized_SuperOp_error(self):
        """Test SuperOp simulation attempt for non-vectorized Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve(signals=None, t_span=[0.0, 1.0], y0=SuperOp(np.eye(4)))
        self.assertTrue("Simulating SuperOp" in str(qe.exception))


class TestSolver(QiskitDynamicsTestCase):
    """Tests for Solver class."""

    def setUp(self):
        """Set up some simple models."""
        X = 2 * np.pi * Operator.from_label("X") / 2
        Z = 2 * np.pi * Operator.from_label("Z") / 2
        self.X = X
        self.Z = Z
        self.ham_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.rwa_ham_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            rwa_cutoff_freq=2 * 5.0,
        )

        self.lindblad_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
        )

        self.vec_lindblad_solver = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_dissipators=[0.01 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            evaluation_mode="dense_vectorized",
        )

        # lindblad solver with no dissipation for testing
        self.vec_lindblad_solver_no_diss = Solver(
            hamiltonian_operators=[X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_dissipators=[0.0 * X],
            static_hamiltonian=5 * Z,
            rotating_frame=5 * Z,
            evaluation_mode="dense_vectorized",
        )
        self.method = "DOP853"

    def test_state_dims_preservation(self):
        """Test that state shapes are correctly preserved."""

        # Hamiltonian only model
        solver = Solver(static_hamiltonian=np.zeros((6, 6)))
        y0 = Statevector(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(signals=None, t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, Statevector))
        self.assertTrue(yf.dims() == (2, 3))

        y0 = DensityMatrix(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(signals=None, t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, DensityMatrix))
        self.assertTrue(yf.dims() == (2, 3))

        # model with Lindblad terms
        solver = Solver(static_dissipators=np.zeros((1, 6, 6)))
        y0 = Statevector(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(signals=None, t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, DensityMatrix))
        self.assertTrue(yf.dims() == (2, 3))

        y0 = DensityMatrix(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), dims=(2, 3))
        yf = solver.solve(signals=None, t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, DensityMatrix))
        self.assertTrue(yf.dims() == (2, 3))

        # SuperOp
        solver.model.evaluation_mode = "dense_vectorized"
        y0 = SuperOp(np.eye(36), input_dims=(2, 3), output_dims=(3, 2))
        yf = solver.solve(signals=None, t_span=[0.0, 0.1], y0=y0).y[-1]
        self.assertTrue(isinstance(yf, SuperOp))
        self.assertTrue(yf.input_dims() == (2, 3) and yf.output_dims() == (3, 2))

    def test_lindblad_solve_statevector(self):
        """Test correct conversion of Statevector to DensityMatrix."""

        results = self.lindblad_solver.solve(
            signals=None, t_span=[0.0, 1.0], y0=Statevector([0.0, 1.0]), method=self.method
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(results.y[-1].data[0, 0] > 0.99 and results.y[-1].data[0, 0] < 0.999)

    def test_vec_lindblad_statevector(self):
        """Test correct conversion of Statevector to DensityMatrix and vectorized solving."""

        results = self.vec_lindblad_solver.solve(
            signals=None, t_span=[0.0, 1.0], y0=Statevector([0.0, 1.0]), method=self.method
        )
        results2 = self.lindblad_solver.solve(
            signals=None, t_span=[0.0, 1.0], y0=Statevector([0.0, 1.0]), method=self.method
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertAllClose(results.y[-1].data, results2.y[-1].data)

    def test_array_vectorized_lindblad(self):
        """Test Lindblad solver is array-vectorized."""
        results = self.lindblad_solver.solve(
            signals=None,
            t_span=[0.0, 1.0],
            y0=np.array([[[0.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 0.0]]]),
            method=self.method,
        )
        self.assertTrue(results.y[-1][0, 0, 0] > 0.99 and results.y[-1][0, 0, 0] < 0.999)
        self.assertTrue(results.y[-1][1, 1, 1] > 0.99 and results.y[-1][1, 1, 1] < 0.999)

    def test_rwa_hamiltonian(self):
        """Test perfect inversion for pi pulse with RWA."""
        results = self.rwa_ham_solver.solve(
            signals=None, t_span=[0.0, 1.0], y0=np.array([0.0, 1.0]), atol=1e-10, rtol=1e-10, method=self.method
        )
        self.assertTrue(np.abs(results.y[-1][0]) > (1 - 1e-8))

    def test_hamiltonian_DensityMatrix(self):
        """Test correct conjugation of Hamiltonian-based density matrix simulation."""
        results = self.ham_solver.solve(
            signals=None,
            t_span=[0.0, 1.0],
            y0=DensityMatrix(np.array([0.0, 1.0])),
            atol=1e-10,
            rtol=1e-10,
            method=self.method,
        )
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(np.abs(results.y[-1].data[0, 0]) > 0.999)

    def test_hamiltonian_SuperOp(self):
        """Test Hamiltonian-based SuperOp simulation."""
        results = self.rwa_ham_solver.solve(
            signals=None, t_span=[0.0, 1.0], y0=SuperOp(np.eye(4)), atol=1e-10, rtol=1e-10, method=self.method
        )
        self.assertTrue(isinstance(results.y[-1], SuperOp))
        X = np.array([[0.0, 1.0], [1.0, 0.0]])
        self.assertAllClose(results.y[-1].data, np.kron(X, X))

    def test_hamiltonian_lindblad_SuperOp_consistency(self):
        """Test Hamiltonian-based SuperOp simulation."""
        results = self.ham_solver.solve(
            signals=None, t_span=[0.0, 0.432], y0=SuperOp(np.eye(4)), atol=1e-10, rtol=1e-10, method=self.method
        )
        results2 = self.vec_lindblad_solver_no_diss.solve(
            signals=None, t_span=[0.0, 0.432], y0=SuperOp(np.eye(4)), atol=1e-10, rtol=1e-10
        )
        self.assertAllClose(results.y[-1].data, results2.y[-1].data)

    def test_lindblad_solver_consistency(self):
        """Test consistency of lindblad solver with dissipators specified
        as constant v.s. non-constant.
        """
        lindblad_solver2 = Solver(
            hamiltonian_operators=[self.X],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            dissipator_operators=[self.X],
            dissipator_signals=[0.01**2],
            static_hamiltonian=5 * self.Z,
            rotating_frame=5 * self.Z,
        )

        results = lindblad_solver2.solve(signals=None, t_span=[0.0, 1.0], y0=Statevector([0.0, 1.0]), method=self.method)
        self.assertTrue(isinstance(results.y[-1], DensityMatrix))
        self.assertTrue(results.y[-1].data[0, 0] > 0.99 and results.y[-1].data[0, 0] < 0.999)


class TestSolverJax(TestSolver, TestJaxBase):
    """JAX version of TestSolver."""

    def setUp(self):
        """Set method to 'jax_odeint' to speed up running of jax version of tests."""
        super().setUp()
        self.method = "jax_odeint"

    def test_transform_through_construction_when_validate_false(self):
        """Test that a function building a Solver can be compiled if validate=False."""

        Z = to_array(self.Z)
        X = to_array(self.X)

        def func(a):
            solver = Solver(
                static_hamiltonian=5 * Z,
                hamiltonian_operators=[X],
                hamiltonian_signals=[Signal(Array(a), 5.0)],
                rotating_frame=5 * Z,
                validate=False,
            )
            yf = solver.solve(signals=None, t_span=np.array([0.0, 0.1]), y0=np.array([0.0, 1.0]), method=self.method).y[
                -1
            ]
            return yf

        jit_func = self.jit_wrap(func)
        self.assertAllClose(jit_func(2.0), func(2.0))

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)

    def test_jit_solve(self):
        """Test jitting setting signals and solving."""

        def func(a):
            ham_solver = self.ham_solver.copy()
            ham_solver.signals = [Signal(lambda t: a, 5.0)]
            yf = ham_solver.solve(
                signals=None, t_span=np.array([0.0, 1.0]), y0=np.array([0.0, 1.0]), method=self.method
            ).y[-1]
            return yf

        jit_func = self.jit_wrap(func)
        self.assertAllClose(jit_func(2.0), func(2.0))

    def test_jit_grad_solve(self):
        """Test jitting setting signals and solving."""

        X = Operator.from_label("X")
        solver = Solver(
            hamiltonian_operators=[X], hamiltonian_signals=[1.0], dissipator_operators=[X]
        )

        def func(a):
            solver_copy = solver.copy()
            solver_copy.signals = [[Signal(lambda t: a, 5.0)], [1.0]]
            yf = solver_copy.solve(
                signals=None, t_span=[0.0, 1.0], y0=np.array([[0.0, 1.0], [0.0, 1.0]], dtype=complex), method=self.method
            ).y[-1]
            return yf

        jit_grad_func = self.jit_grad_wrap(func)
        jit_grad_func(1.0)
