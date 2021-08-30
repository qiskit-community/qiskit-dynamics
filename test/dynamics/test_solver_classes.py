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

"""
Tests for solver classes module.
"""

from .common import QiskitDynamicsTestCase, TestJaxBase

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info import Operator, Statevector, SuperOp, DensityMatrix

from qiskit_dynamics import Solver

class TestSolverExceptions(QiskitDynamicsTestCase):
    """Tests for Solver exception raising based on input types."""

    def setUp(self):
        X = Operator.from_label('X')
        self.ham_solver = Solver(hamiltonian_operators=[X],
                                 hamiltonian_signals=[1.])

        self.lindblad_solver = Solver(hamiltonian_operators=[X],
                                      hamiltonian_signals=[1.],
                                      dissipator_operators=[X])

        self.vec_lindblad_solver = Solver(hamiltonian_operators=[X],
                                      hamiltonian_signals=[1.],
                                      dissipator_operators=[X],
                                      evaluation_mode='dense_vectorized')

    def test_hamiltonian_shape_error(self):
        """Test error raising if invalid shape for Hamiltonian model."""

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve([0., 1.], np.array([1., 0., 0.]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve([0., 1.], np.array([[[1., 0., 0.]]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.ham_solver.solve([0., 1.], Statevector(np.array([1., 0., 0.])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_lindblad_shape_error(self):
        """Test error raising if invalid shape for Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0., 1.], np.array([1., 0., 0.]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0., 1.], np.array([[[1., 0., 0.]]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0., 1.], Statevector(np.array([1., 0., 0.])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_vectorized_lindblad_shape_error(self):
        """Test error raising if invalid shape for vectorized Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve([0., 1.], np.array([[1., 0.], [0., 1.]]))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve([0., 1.], DensityMatrix(np.array([1., 0., 0.])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            self.vec_lindblad_solver.solve([0., 1.], Statevector(np.array([1., 0., 0.])))
        self.assertTrue("Shape mismatch" in str(qe.exception))

    def test_non_vectorized_SuperOp_error(self):
        """Test SuperOp simulation attempt for non-vectorized Lindblad model."""

        with self.assertRaises(QiskitError) as qe:
            self.lindblad_solver.solve([0., 1.], SuperOp(np.eye(4)))
        self.assertTrue("Simulating SuperOp" in str(qe.exception))



class TestSolver(QiskitDynamicsTestCase):
    pass

class TestSolverJax(TestSolver, TestJaxBase):
    """JAX version of TestSolver."""
    pass
