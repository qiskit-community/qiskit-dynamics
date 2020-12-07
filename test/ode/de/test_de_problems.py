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

"""tests for DE_Problems.py"""

import unittest
import numpy as np
from scipy.linalg import expm

from qiskit.quantum_info.operators import Operator
from qiskit_ode.dispatch import Array
from qiskit_ode.models.signals import Constant, Signal
from qiskit_ode.models.operator_models import OperatorModel
from qiskit_ode.models.quantum_models import HamiltonianModel, LindbladModel
from qiskit_ode.de.de_problems import (ODEProblem, LMDEProblem,
                                       OperatorModelProblem, SchrodingerProblem,
                                       DensityMatrixProblem)
from qiskit_ode.type_utils import (vec_commutator, vec_dissipator)

from ..test_jax_base import TestJaxBase


class TestODEProblem(unittest.TestCase):
    """Basic tests for ODEProblem."""

    def test_set_rhs_numpy(self):
        """Test wrapping of RHS."""

        # pylint: disable=unused-argument
        def rhs(t, y):
            return y**2

        ode_problem = ODEProblem(rhs)

        output = ode_problem(0., Array([1., 2.]))

        self.assertTrue(isinstance(output, Array))
        self.assertTrue(np.allclose(output, Array([1., 4.])))


class TestLMDEProblem(unittest.TestCase):
    """Basic tests for LMDEProblem."""

    def test_set_generator(self):
        """Test wrapping of generator."""

        def generator(t):
            return t * np.array([[0., -1j], [1j, 0.]])

        lmde_problem = LMDEProblem(generator)

        output = lmde_problem.generator(3.)

        self.assertTrue(isinstance(output, Array))
        self.assertTrue(np.allclose(output, 3 * Array([[0., -1j], [1j, 0.]])))


class TestOperatorModelProblem(unittest.TestCase):
    """Base class for testing OperatorModelProblem with different backends."""

    def setUp(self):
        # wrote it this way as not integrated with terra at time of writing
        self.X = Array([[0., 1.], [1., 0.]], dtype=complex)
        self.Y = Array([[0., -1j], [1j, 0.]], dtype=complex)
        self.Z = Array([[1., 0.], [0., -1.]], dtype=complex)

        # define a basic model
        w = 2.
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2,
                     -1j * 2 * np.pi * r * self.X / 2]
        signals = [Constant(w), Signal(1., w)]

        self.w = 2
        self.r = r
        self.basic_model = OperatorModel(operators=operators, signals=signals)

        self.y0 = Array([1., 0.], dtype=complex)

    def test_auto_frame_handling(self):
        """Test automatic setting of frames."""
        self.basic_model.frame = self.X

        problem = OperatorModelProblem(generator=self.basic_model)

        self.assertTrue(np.allclose(problem.user_frame.frame_operator,
                                    Array([[0., 1.], [1., 0.]], dtype=complex)))
        self.assertTrue(np.allclose(problem.generator_model.frame.frame_operator,
                                    -1j * 2 * np.pi * self.w * self.Z / 2))

    def test_user_state_to_problem(self):
        """Test user_state_to_problem.

        Note: To go from a user facing state to a problem state requires:
            - transformation out of user frame
            - transformation into problem frame
            - transformation into problem basis
        """
        problem = OperatorModelProblem(generator=self.basic_model,
                                       user_frame=self.X)

        # force use of numpy for computing expected output
        y = np.array([0., 1.], dtype=complex)
        t = 0.124
        drift_frame_op = 2 * np.pi * self.w * self.Z / 2
        expected = (self.X.data @ expm(1j * t * np.array(drift_frame_op.data)) @
                    expm(-1j * t * np.array(self.X.data)) @ y)

        output = problem.user_state_to_problem(t, y)

        self.assertTrue(np.allclose(expected, output))

    def test_problem_state_to_user(self):
        """Test problem_state_to_user.

        Note: To go from a user facing state to a problem state requires:
            - transformation out of problem basis
            - transformation out of problem frame
            - transformation into user frame
        """
        problem = OperatorModelProblem(generator=self.basic_model,
                                       user_frame=self.X)

        # force use of numpy for computing expected output
        y = np.array([0., 1.], dtype=complex)
        t = 0.094882
        drift_frame_op = 2 * np.pi * self.w * self.Z / 2
        expected = (expm(1j * t * np.array(self.X.data)) @
                    expm(-1j * t * np.array(drift_frame_op.data)) @
                    self.X.data @ y)

        output = problem.problem_state_to_user(t, y)

        self.assertTrue(np.allclose(expected, output))

    def test_generator(self):
        """Test correct evaluation of generator.
        The generator is evaluated in the solver frame in a basis in which the
        frame operator is diagonal.
        """
        problem = OperatorModelProblem(generator=self.basic_model,
                                       solver_frame=self.X)

        t = 13.1231

        output = problem.generator(t).data

        X = np.array(self.X.data)
        X_diag, U = np.linalg.eigh(X)
        Uadj = U.conj().transpose()
        gen = -1j * 2 * np.pi * (self.w * np.array(self.Z.data) / 2 +
                                 self.r * np.cos(2 * np.pi * self.w * t) *
                                 X / 2)
        expected = (Uadj @ expm(1j * t * X) @ gen @ expm(-1j * t * X) @ U
                    + 1j * np.diag(X_diag))

        self.assertTrue(np.allclose(expected, output))

    def test_rhs(self):
        """Test correct evaluation of rhs.
        The generator is evaluated in the solver frame in a basis in which the
        frame operator is diagonal.
        """
        problem = OperatorModelProblem(generator=self.basic_model,
                                       solver_frame=self.X)

        t = 13.1231
        y = np.eye(2, dtype=complex)

        output = problem.rhs(t, y).data

        X = np.array(self.X.data)
        X_diag, U = np.linalg.eigh(X)
        Uadj = U.conj().transpose()
        gen = -1j * 2 * np.pi * (self.w * np.array(self.Z.data) / 2 +
                                 self.r * np.cos(2 * np.pi * self.w * t) *
                                 X / 2)
        expected = (Uadj @ expm(1j * t * X) @ gen @ expm(-1j * t * X) @ U
                    + 1j * np.diag(X_diag)) @ y

        self.assertTrue(np.allclose(expected, output))

    def test_solver_cutoff_freq(self):
        """Test correct setting of solver cutoff freq.
        """
        problem = OperatorModelProblem(generator=self.basic_model,
                                       solver_cutoff_freq=2 * self.w)

        self.assertTrue(problem.generator_model.cutoff_freq == 2 * self.w)
        self.assertTrue(self.basic_model.cutoff_freq is None)


class TestOperatorModelProblemJax(TestOperatorModelProblem, TestJaxBase):
    """Jax version of OperatorModelProblem tests.

    Note: This class has no body but contains tests due to inheritance.
    """


class TestSchrodingerProblem(unittest.TestCase):
    """Test SchrodingerProblem."""

    def setUp(self):
        self.X = Array(Operator.from_label('X').data)
        self.Y = Array(Operator.from_label('Y').data)
        self.Z = Array(Operator.from_label('Z').data)

        # define a basic model
        w = 2.
        r = 0.5
        operators = [2 * np.pi * self.Z / 2,
                     2 * np.pi * r * self.X / 2]
        signals = [Constant(w), Signal(1., w)]

        self.w = 2
        self.r = r
        self.basic_hamiltonian = HamiltonianModel(operators=operators,
                                                  signals=signals)

    def test_generator_construction(self):
        """Test correct construction of the Schrodinger generator
        :math:`-iH(t)`.
        """

        se_prob = SchrodingerProblem(self.basic_hamiltonian)

        self.assertTrue(np.allclose(se_prob.generator_model.operators[0],
                                    -1j * 2 * np.pi * self.Z / 2))
        self.assertTrue(np.allclose(se_prob.generator_model.operators[1],
                                    -1j * 2 * np.pi * self.r * self.X / 2))


class TestSchrodingerProblemJax(TestSchrodingerProblem, TestJaxBase):
    """Jax version of SchrodingerProblem tests.

    Note: This class has no body but contains tests due to inheritance.
    """


# pylint: disable=too-many-instance-attributes
class TestDensityMatrixProblem(unittest.TestCase):
    """Test DensityMatrixProblem."""

    def setUp(self):
        self.X = Array(Operator.from_label('X').data)
        self.Y = Array(Operator.from_label('Y').data)
        self.Z = Array(Operator.from_label('Z').data)

        # define a basic model
        w = 2.
        r = 0.5
        operators = [2 * np.pi * self.Z / 2,
                     2 * np.pi * r * self.X / 2]
        signals = [Constant(w), Signal(1., w)]

        self.w = 2
        self.r = r
        hamiltonian = HamiltonianModel(operators=operators,
                                       signals=signals)

        self.noise_ops = [Array([[0., 1.], [0., 0.]], dtype=complex),
                          Array([[0., 0.], [1., 0.]], dtype=complex)]

        self.basic_lindblad_model = LindbladModel.from_hamiltonian(hamiltonian=hamiltonian,
                                                                   noise_operators=self.noise_ops)

        # not a valid density matrix but can be used for testing
        self.y0 = Array([[1., 2.], [3., 4.]], dtype=complex)

    def test_basic_generatoroperators(self):
        """Test correct construction of the operators in the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model)

        # validate generator matrices
        self.assertTrue(np.allclose(l_prob.generator_model.operators[0],
                                    vec_commutator(-1j * 2 * np.pi * self.Z / 2)))
        self.assertTrue(np.allclose(l_prob.generator_model.operators[1],
                                    vec_commutator(-1j * 2 * np.pi * self.r * self.X / 2)))
        self.assertTrue(np.allclose(l_prob.generator_model.operators[2],
                                    vec_dissipator(self.noise_ops[0])))
        self.assertTrue(np.allclose(l_prob.generator_model.operators[3],
                                    vec_dissipator(self.noise_ops[1])))

    def test_basic_generator_signals(self):
        """Test correct construction of the signals in the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model)

        # validate generator signals
        t = 0.12314
        sig_vals = l_prob.generator_model.signals.value(t)
        expected = np.array([self.w,
                             np.exp(1j * 2 * np.pi * self.w * t),
                             1.,
                             1.])

        self.assertTrue(np.allclose(sig_vals, expected))

    def test_basic_generator_frame(self):
        """Test correct construction of the frame for the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model)

        frame_op = l_prob.generator_model.frame.frame_operator
        expected = vec_commutator(-1j * 2 * np.pi * self.w * self.Z / 2)
        self.assertTrue(np.allclose(frame_op, expected))

    def test_state_type_converter(self):
        """Test correct construction state_type_converter for density matrix
        problem.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model)

        # ensure that converter correctly flattens states
        self.assertTrue(np.allclose(l_prob.state_type_converter.outer_to_inner(self.y0),
                                    self.y0.flatten(order='F')))

        # ensure that converter correctly unflattens states
        # pylint: disable=line-too-long
        self.assertTrue(np.allclose(l_prob.state_type_converter.inner_to_outer(Array([1., 2.,
                                                                                      3., 4.])),
                                    Array([1., 2., 3., 4.]).reshape((2, 2), order='F')))


class TestDensityMatrixProblemJax(TestDensityMatrixProblem, TestJaxBase):
    """Jax version of TestDensityMatrixProblem tests.

    Note: This class has no body but contains tests due to inheritance.
    """


# pylint: disable=too-many-instance-attributes
class TestSuperOpProblem(unittest.TestCase):
    """Test SuperOpProblem.

    For now same tests as for DensityMatrixProblem.
    """

    def setUp(self):
        self.X = Array(Operator.from_label('X').data)
        self.Y = Array(Operator.from_label('Y').data)
        self.Z = Array(Operator.from_label('Z').data)

        # define a basic model
        w = 2.
        r = 0.5
        operators = [2 * np.pi * self.Z / 2,
                     2 * np.pi * r * self.X / 2]
        signals = [Constant(w), Signal(1., w)]

        self.w = 2
        self.r = r
        hamiltonian = HamiltonianModel(operators=operators,
                                       signals=signals)

        self.noise_ops = [Array([[0., 1.], [0., 0.]]),
                          Array([[0., 0.], [1., 0.]])]

        self.basic_lindblad_model = LindbladModel.from_hamiltonian(hamiltonian=hamiltonian,
                                                                   noise_operators=self.noise_ops)

        # not a valid density matrix but can be used for testing
        self.y0 = np.eye(4, dtype=complex)

    def test_basic_generator_operators(self):
        """Test correct construction of the operators in the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model)

        # validate generator matrices
        self.assertTrue(np.allclose(l_prob.generator_model.operators[0],
                                    vec_commutator(-1j * 2 * np.pi * self.Z / 2)))
        self.assertTrue(np.allclose(l_prob.generator_model.operators[1],
                                    vec_commutator(-1j * 2 * np.pi * self.r * self.X / 2)))
        self.assertTrue(np.allclose(l_prob.generator_model.operators[2],
                                    vec_dissipator(self.noise_ops[0])))
        self.assertTrue(np.allclose(l_prob.generator_model.operators[3],
                                    vec_dissipator(self.noise_ops[1])))

    def test_basic_generator_signals(self):
        """Test correct construction of the signals in the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model)

        # validate generator signals
        t = 0.12314
        sig_vals = l_prob.generator_model.signals.value(t)
        expected = np.array([self.w,
                             np.exp(1j * 2 * np.pi * self.w * t),
                             1.,
                             1.])

        self.assertTrue(np.allclose(sig_vals, expected))

    def test_basic_generator_frame(self):
        """Test correct construction of the frame for the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model)

        frame_op = l_prob.generator_model.frame.frame_operator
        expected = vec_commutator(-1j * 2 * np.pi * self.w * self.Z / 2)
        self.assertTrue(np.allclose(frame_op, expected))


class TestSuperOpProblemJax(TestSuperOpProblem, TestJaxBase):
    """Jax version of TestSuperOpProblem tests.

    Note: This class has no body but contains tests due to inheritance.
    """
