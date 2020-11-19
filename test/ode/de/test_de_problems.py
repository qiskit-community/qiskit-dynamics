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
"""tests for DE_Problems.py"""

import unittest
import warnings
import numpy as np

from qiskit.quantum_info.operators import Operator
from qiskit_ode.models.signals import Constant, Signal
from qiskit_ode.models.operator_models import OperatorModel
from qiskit_ode.models.quantum_models import HamiltonianModel, LindbladModel
from qiskit_ode.de.DE_Problems import (BMDE_Problem,
                                       SchrodingerProblem,
                                       DensityMatrixProblem,
                                       SuperOpProblem)
from qiskit_ode.type_utils import (vec_commutator, vec_dissipator)

class TestBMDE_Problem(unittest.TestCase):

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

        # define a basic model
        w = 2.
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2,
                     -1j * 2 * np.pi * r * self.X / 2]
        signals = [Constant(w), Signal(1., w)]

        self.w = 2
        self.r = r
        self.basic_model = OperatorModel(operators=operators, signals=signals)

        self.y0 = np.array([1., 0.])

    def test_t0_interval_error(self):
        """Test exception raising for specifying both t0 and an interval."""

        try:
            bmde_problem = BMDE_Problem(generator=self.basic_model,
                                        y0=self.y0,
                                        t0=0.,
                                        interval=[0.,1.])
        except Exception as e:
            self.assertTrue('t0 or interval.' in str(e))

    def test_generator_copied(self):
        """Ensure that the generator in the bmde_problem is a copy."""
        bmde_problem = BMDE_Problem(generator=self.basic_model,
                                    y0=self.y0,
                                    t0=0.,
                                    frame=None)
        self.basic_model.frame = 1j * np.array([1, -1])

        self.assertTrue(bmde_problem._generator.frame.frame_operator is None)

    def test_user_in_frame(self):
        """Test correct setting of _user_in_frame."""
        bmde_problem = BMDE_Problem(generator=self.basic_model,
                                    y0=self.y0,
                                    t0=0.)

        self.assertTrue(not bmde_problem._user_in_frame)

        self.basic_model.frame = np.array(-1j * np.array([-1,1]))

        bmde_problem = BMDE_Problem(generator=self.basic_model,
                                    y0=self.y0,
                                    t0=0.)

        self.assertTrue(bmde_problem._user_in_frame)

    def test_frame_auto(self):
        """Test auto setting of frame."""
        bmde_problem = BMDE_Problem(generator=self.basic_model,
                                    y0=self.y0,
                                    t0=0.)

        self.assertAlmostEqual(bmde_problem._generator.frame.frame_operator,
                               self.basic_model.drift)

    def test_cutoff_freq_error(self):
        """Test cutoff frequency error."""
        self.basic_model.cutoff_freq = 2.
        try:
            bmde_problem = BMDE_Problem(generator=self.basic_model,
                                        y0=self.y0,
                                        t0=0.,
                                        cutoff_freq=1.)
        except Exception as e:
            self.assertTrue('Cutoff frequency' in str(e))

    def test_double_frame_warning(self):
        """Test that specifying a frame in the model and when constructing
        the BMDE problem raises a warning.
        """
        self.basic_model.frame = np.array(-1j * np.array([-1, 1]))
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            bmde_problem = BMDE_Problem(generator=self.basic_model,
                                        y0=self.y0,
                                        t0=0.,
                                        frame=None)
            self.assertEqual(len(ws), 1)
            self.assertTrue('A frame' in str(ws[-1].message))


    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)

class TestSchrodingerProblem(unittest.TestCase):
    """Test SchrodingerProblem.
    """

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

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

        self.y0 = np.array([1., 0.])

    def test_generator_construction(self):
        """Test correct construction of the Schrodinger generator
        :math:`-iH(t)`.
        """

        se_prob = SchrodingerProblem(self.basic_hamiltonian,
                                     self.y0,
                                     t0=0.)

        self.assertAlmostEqual(se_prob._generator._operators[0].data,
                         -1j * 2 * np.pi * self.Z.data / 2)
        self.assertAlmostEqual(se_prob._generator._operators[1].data,
                         -1j * 2 * np.pi * self.r * self.X.data / 2)

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)

class TestDensityMatrixProblem(unittest.TestCase):
    """Test DensityMatrixProblem.
    """

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

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

        self.noise_ops =[Operator(np.array([[0., 1.], [0., 0.]])),
                         Operator(np.array([[0., 0.], [1., 0.]]))]

        self.basic_lindblad_model = LindbladModel.from_hamiltonian(hamiltonian=hamiltonian,
                                                                   noise_operators=self.noise_ops)

        # not a valid density matrix but can be used for testing
        self.y0 = np.array([[1., 2.], [3., 4.]])

    def test_basic_generator_operators(self):
        """Test correct construction of the operators in the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model,
                                      self.y0,
                                      t0=0.)

        # validate generator matrices
        self.assertAlmostEqual(l_prob._generator._operators[0].data,
                               vec_commutator(-1j * 2 * np.pi * self.Z.data / 2))
        self.assertAlmostEqual(l_prob._generator._operators[1].data,
                               vec_commutator(-1j * 2 * np.pi * self.r * self.X.data / 2))
        self.assertAlmostEqual(l_prob._generator._operators[2].data,
                               vec_dissipator(self.noise_ops[0].data))
        self.assertAlmostEqual(l_prob._generator._operators[3].data,
                               vec_dissipator(self.noise_ops[1].data))

    def test_basic_generator_signals(self):
        """Test correct construction of the signals in the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model,
                                      self.y0,
                                      t0=0.)

        # validate generator signals
        t = 0.12314
        sig_vals = l_prob._generator.signals.value(t)
        expected = np.array([self.w,
                             np.exp(1j * 2 * np.pi * self.w * t),
                             1.,
                             1.])

        self.assertAlmostEqual(sig_vals, expected)

    def test_basic_generator_frame(self):
        """Test correct construction of the frame for the vectorized
        Lindblad generator.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model,
                                      self.y0,
                                      t0=0.)

        frame_op = l_prob._generator.frame.frame_operator
        expected = vec_commutator(-1j * 2 * np.pi * self.w * self.Z.data / 2)
        self.assertAlmostEqual(frame_op, expected)

    def test_state_type_converter(self):
        """Test correct construction state_type_converter for density matrix
        problem.
        """

        l_prob = DensityMatrixProblem(self.basic_lindblad_model,
                                      self.y0,
                                      t0=0.)

        # ensure that stored state is correct
        self.assertAlmostEqual(l_prob.y0, self.y0)

        # ensure that converter correctly flattens states
        self.assertAlmostEqual(l_prob._state_type_converter.outer_to_inner(l_prob.y0),
                               self.y0.flatten(order='F'))

        # ensure that converter correctly unflattens states
        self.assertAlmostEqual(l_prob._state_type_converter.inner_to_outer(np.array([1.,2.,3.,4.])),
                               np.array([1.,2.,3.,4.]).reshape((2,2),order='F'))

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)


class TestSuperOpProblem(unittest.TestCase):
    """Test SuperOpProblem.

    For now same tests as for DensityMatrixProblem.
    """

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

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

        self.noise_ops =[Operator(np.array([[0., 1.], [0., 0.]])),
                         Operator(np.array([[0., 0.], [1., 0.]]))]

        self.basic_lindblad_model = LindbladModel.from_hamiltonian(hamiltonian=hamiltonian,
                                                                   noise_operators=self.noise_ops)

        # not a valid density matrix but can be used for testing
        self.y0 = np.eye(4, dtype=complex)

    def test_basic_generator_operators(self):
        """Test correct construction of the operators in the vectorized
        Lindblad generator.
        """

        l_prob = SuperOpProblem(self.basic_lindblad_model,
                                self.y0,
                                t0=0.)

        # validate generator matrices
        self.assertAlmostEqual(l_prob._generator._operators[0].data,
                               vec_commutator(-1j * 2 * np.pi * self.Z.data / 2))
        self.assertAlmostEqual(l_prob._generator._operators[1].data,
                               vec_commutator(-1j * 2 * np.pi * self.r * self.X.data / 2))
        self.assertAlmostEqual(l_prob._generator._operators[2].data,
                               vec_dissipator(self.noise_ops[0].data))
        self.assertAlmostEqual(l_prob._generator._operators[3].data,
                               vec_dissipator(self.noise_ops[1].data))

    def test_basic_generator_signals(self):
        """Test correct construction of the signals in the vectorized
        Lindblad generator.
        """

        l_prob = SuperOpProblem(self.basic_lindblad_model,
                                self.y0,
                                t0=0.)

        # validate generator signals
        t = 0.12314
        sig_vals = l_prob._generator.signals.value(t)
        expected = np.array([self.w,
                             np.exp(1j * 2 * np.pi * self.w * t),
                             1.,
                             1.])

        self.assertAlmostEqual(sig_vals, expected)

    def test_basic_generator_frame(self):
        """Test correct construction of the frame for the vectorized
        Lindblad generator.
        """

        l_prob = SuperOpProblem(self.basic_lindblad_model,
                                self.y0,
                                t0=0.)

        frame_op = l_prob._generator.frame.frame_operator
        expected = vec_commutator(-1j * 2 * np.pi * self.w * self.Z.data / 2)

        self.assertAlmostEqual(frame_op, expected)

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
