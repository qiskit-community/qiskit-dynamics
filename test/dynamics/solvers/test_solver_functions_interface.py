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
# pylint: disable=invalid-name,broad-except

"""Tests error handling of solve_ode and solve_lmde, and helper functions.

Tests for results of solvers are in test_solver_functions.py.
"""

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info import Operator

from qiskit_dynamics.models import GeneratorModel, HamiltonianModel, LindbladModel
from qiskit_dynamics.signals import Signal
from qiskit_dynamics import solve_ode, solve_lmde
from qiskit_dynamics.solvers.solver_functions import (
    setup_generator_model_rhs_y0_in_frame_basis,
    results_y_out_of_frame_basis,
)

from ..common import QiskitDynamicsTestCase


class Testsolve_ode_exceptions(QiskitDynamicsTestCase):
    """Test exceptions of solve_ode."""

    def test_method_does_not_exist(self):
        """Test method does not exist exception."""

        with self.assertRaisesRegex(QiskitError, "not supported"):
            solve_ode(lambda t, y: y, t_span=[0.0, 1.0], y0=np.array([1.0]), method="notamethod")


class Testsolve_lmde_exceptions(QiskitDynamicsTestCase):
    """Test for solve_lmde error raising."""

    def setUp(self):
        self.lindblad_model = LindbladModel(
            hamiltonian_operators=[np.diag([1.0, -1.0])], hamiltonian_signals=[1.0]
        )
        # self.hamiltonian_model = HamiltonianModel(operators=[])

    def test_lmde_method_non_vectorized_lindblad(self):
        """Test error raising if LMDE method is specified for non-vectorized Lindblad."""

        with self.assertRaisesRegex(QiskitError, "vectorized evaluation"):
            solve_lmde(
                self.lindblad_model, t_span=[0.0, 1.0], y0=np.diag([1.0, 0.0]), method="scipy_expm"
            )

    def test_lanczos_antihermitian(self):
        """Test error raising when lanczos is used for non anti-Hermitian generators."""

        with self.assertRaisesRegex(QiskitError, "anti-Hermitian generators"):
            solve_lmde(
                lambda t: np.array([[0, 1], [1, 0]]),
                t_span=[0.0, 1.0],
                y0=np.array([1.0, 0.0]),
                method="lanczos_diag",
                max_dt=0.1,
                k_dim=2,
            )

    def test_lanczos_k_dim(self):
        """Test error raising when number of lanczos vectors is greater than dimension of
        generator."""

        with self.assertRaisesRegex(QiskitError, "greater than dimension of generator"):
            solve_lmde(
                lambda t: np.array([[0, 1], [-1, 0]]),
                t_span=[0.0, 1.0],
                y0=np.array([1.0, 0.0]),
                method="lanczos_diag",
                max_dt=0.1,
                k_dim=4,
            )

    def test_lanczos_y0_dim(self):
        """Test error raising when y0 is not 1d or 2d in lanczos."""

        with self.assertRaisesRegex(ValueError, "y0 must be 1d or 2d"):
            solve_lmde(
                lambda t: np.array([[0, 1], [-1, 0]]),
                t_span=[0.0, 1.0],
                y0=np.random.rand(2, 2, 2),
                method="lanczos_diag",
                max_dt=0.1,
                k_dim=2,
            )

    def test_method_does_not_exist(self):
        """Test method does not exist exception."""

        with self.assertRaisesRegex(QiskitError, "not supported"):
            solve_lmde(lambda t: t, t_span=[0.0, 1.0], y0=np.array([1.0]), method="notamethod")

    def test_jax_expm_sparse_mode(self):
        """Verify an error gets raised if the jax expm solver is attempted to be used
        in sparse mode."""

        model = GeneratorModel(
            static_operator=np.array([[0.0, 1.0], [1.0, 0.0]]), evaluation_mode="sparse"
        )

        with self.assertRaisesRegex(QiskitError, "jax_expm cannot be used"):
            solve_lmde(
                model, t_span=[0.0, 1.0], y0=np.array([1.0, 1.0]), method="jax_expm", max_dt=0.1
            )


class TestLMDEGeneratorModelSetup(QiskitDynamicsTestCase):
    """Tests for LMDE GeneratorModel setup methods."""

    def setUp(self):
        """Define some models with different properties."""

        self.ham_model = HamiltonianModel(
            operators=[Operator.from_label("X")],
            signals=[Signal(1.0, 5.0)],
            static_operator=Operator.from_label("Z"),
        )

        self.lindblad_model = LindbladModel(
            hamiltonian_operators=[Operator.from_label("X")],
            hamiltonian_signals=[Signal(1.0, 5.0)],
            static_hamiltonian=Operator.from_label("Z"),
            static_dissipators=[Operator.from_label("Y")],
        )

        self.vec_lindblad_model = self.lindblad_model.copy()
        self.vec_lindblad_model.evaluation_mode = "dense_vectorized"

        self.frame_op = 1.2 * Operator.from_label("X") - 3.132 * Operator.from_label("Y")
        _, U = np.linalg.eigh(self.frame_op)
        self.U = U
        self.Uadj = self.U.conj().transpose()

    def test_hamiltonian_setup_no_frame(self):
        """Test functions for Hamiltonian with no frame."""

        y0 = np.array([3.43, 1.31])
        gen, rhs, new_y0, model_in_frame_basis = setup_generator_model_rhs_y0_in_frame_basis(
            self.ham_model, y0
        )

        # expect nothing to happen
        self.assertAllClose(y0, new_y0)
        t = 231.232
        self.assertAllClose(gen(t), self.ham_model(t))
        self.assertAllClose(rhs(t, y0), self.ham_model(t, y0))

        # check frame parameters
        self.assertFalse(model_in_frame_basis)
        # check that model has been converted to being in frame basis
        self.assertTrue(self.ham_model.in_frame_basis)

    def test_hamiltonian_setup(self):
        """Test functions for Hamiltonian with frame."""

        ham_model = self.ham_model.copy()
        ham_model.rotating_frame = self.frame_op
        y0 = np.array([3.43, 1.31])
        gen, rhs, new_y0, model_in_frame_basis = setup_generator_model_rhs_y0_in_frame_basis(
            ham_model, y0
        )

        # check frame parameters
        self.assertFalse(model_in_frame_basis)
        # check that model has been converted to being in frame basis
        self.assertTrue(ham_model.in_frame_basis)
        self.assertFalse(self.ham_model.in_frame_basis)

        self.assertAllClose(self.Uadj @ y0, new_y0)
        t = 231.232
        self.ham_model.in_frame_basis = True
        self.assertAllClose(gen(t), ham_model(t))
        self.assertAllClose(rhs(t, y0), ham_model(t, y0))

    def test_lindblad_setup_no_frame(self):
        """Test functions for LindbladModel with no frame."""

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]])
        _, rhs, new_y0, model_in_frame_basis = setup_generator_model_rhs_y0_in_frame_basis(
            self.lindblad_model, y0
        )

        # expect nothing to happen
        self.assertAllClose(y0, new_y0)
        t = 231.232
        self.assertAllClose(rhs(t, y0), self.lindblad_model(t, y0))
        self.assertFalse(model_in_frame_basis)

    def test_lindblad_setup(self):
        """Test functions for LindbladModel with frame."""

        lindblad_model = self.lindblad_model.copy()
        lindblad_model.rotating_frame = self.frame_op

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]])
        _, rhs, new_y0, _ = setup_generator_model_rhs_y0_in_frame_basis(lindblad_model, y0)

        # expect nothing to happen
        self.assertAllClose(self.Uadj @ y0 @ self.U, new_y0)
        t = 231.232
        self.assertTrue(lindblad_model.in_frame_basis)
        self.assertAllClose(rhs(t, y0), lindblad_model(t, y0))

    def test_vectorized_lindblad_setup_no_frame(self):
        """Test functions for vectorized LindbladModel with no frame."""

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]]).flatten()
        gen, rhs, new_y0, _ = setup_generator_model_rhs_y0_in_frame_basis(
            self.vec_lindblad_model, y0
        )

        # expect nothing to happen
        self.assertAllClose(y0, new_y0)
        t = 231.232
        self.assertTrue(self.vec_lindblad_model.in_frame_basis)
        self.assertAllClose(gen(t), self.vec_lindblad_model(t))
        self.assertAllClose(rhs(t, y0), self.vec_lindblad_model(t, y0))

    def test_vectorized_lindblad_setup(self):
        """Test functions for vectorized LindbladModel with frame."""

        vec_lindblad_model = self.vec_lindblad_model.copy()
        vec_lindblad_model.rotating_frame = self.frame_op

        y0 = np.array([[3.43, 1.31], [3.0, 1.23]]).flatten()
        gen, rhs, new_y0, _ = setup_generator_model_rhs_y0_in_frame_basis(vec_lindblad_model, y0)

        # expect nothing to happen
        self.assertAllClose(np.kron(self.Uadj.conj(), self.Uadj) @ y0, new_y0)
        t = 231.232
        self.assertTrue(vec_lindblad_model.in_frame_basis)
        self.assertAllClose(gen(t), vec_lindblad_model(t))
        self.assertAllClose(rhs(t, y0), vec_lindblad_model(t, y0))

    def test_hamiltonian_results_conversion_no_frame(self):
        """Test hamiltonian results conversion with no frame."""

        # test 1d input
        results_y = np.array([[1.0, 23.3], [2.32, 1.232]])
        output = results_y_out_of_frame_basis(self.ham_model, results_y, y0_ndim=1)
        self.assertAllClose(results_y, output)

        # test 2d input
        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(self.ham_model, results_y, y0_ndim=2)
        self.assertAllClose(results_y, output)

    def test_hamiltonian_results_conversion(self):
        """Test hamiltonian results conversion."""

        ham_model = self.ham_model.copy()
        ham_model.rotating_frame = self.frame_op

        # test 1d input
        results_y = np.array([[1.0, 23.3], [2.32, 1.232]])
        output = results_y_out_of_frame_basis(ham_model, results_y, y0_ndim=1)
        expected = [self.U @ y for y in results_y]
        self.assertAllClose(expected, output)

        # test 2d input
        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(ham_model, results_y, y0_ndim=2)
        expected = [self.U @ y for y in results_y]
        self.assertAllClose(expected, output)

    def test_lindblad_results_conversion_no_frame(self):
        """Test lindblad results conversion with no frame."""

        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(self.lindblad_model, results_y, y0_ndim=2)
        self.assertAllClose(results_y, output)

    def test_lindblad_results_conversion(self):
        """Test lindblad results conversion."""

        lindblad_model = self.lindblad_model.copy()
        lindblad_model.rotating_frame = self.frame_op

        results_y = np.array([[[1.0, 23.3], [23, 231j]], [[2.32, 1.232], [1j, 2.0 + 3j]]])
        output = results_y_out_of_frame_basis(lindblad_model, results_y, y0_ndim=2)
        expected = [self.U @ y @ self.Uadj for y in results_y]
        self.assertAllClose(expected, output)

    def test_vectorized_lindblad_results_conversion_no_frame(self):
        """Test vectorized lindblad results conversion with no frame."""

        # test 1d input
        results_y = np.array([[1.0, 23.3, 1.23, 0.123], [2.32, 1.232, 1j, 21.2]])
        output = results_y_out_of_frame_basis(self.vec_lindblad_model, results_y, y0_ndim=1)
        self.assertAllClose(results_y, output)

        # test 2d input
        results_y = np.array(
            [
                [[1.0, 23.3], [23, 231j], [2.231, 32.32], [2.321, 2.231]],
                [[2.32, 1.232], [1j, 2.0 + 3j], [2.32, 2.12314], [334.0, 34.3]],
            ]
        )
        output = results_y_out_of_frame_basis(self.vec_lindblad_model, results_y, y0_ndim=2)
        self.assertAllClose(results_y, output)

    def test_vectorized_lindblad_results_conversion(self):
        """Test vectorized lindblad results conversion."""

        vec_lindblad_model = self.vec_lindblad_model.copy()
        vec_lindblad_model.rotating_frame = self.frame_op
        P = np.kron(self.U.conj(), self.U)

        # test 1d input
        results_y = np.array([[1.0, 23.3, 1.23, 0.123], [2.32, 1.232, 1j, 21.2]])
        output = results_y_out_of_frame_basis(vec_lindblad_model, results_y, y0_ndim=1)
        expected = [P @ y for y in results_y]
        self.assertAllClose(expected, output)

        # test 2d input
        results_y = np.array(
            [
                [[1.0, 23.3], [23, 231j], [2.231, 32.32], [2.321, 2.231]],
                [[2.32, 1.232], [1j, 2.0 + 3j], [2.32, 2.12314], [334.0, 34.3]],
            ]
        )
        output = results_y_out_of_frame_basis(vec_lindblad_model, results_y, y0_ndim=2)
        expected = [P @ y for y in results_y]
        self.assertAllClose(expected, output)
