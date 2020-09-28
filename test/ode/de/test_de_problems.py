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
from qiskit.providers.aer.pulse_new.models.signals import Constant, Signal
from qiskit.providers.aer.pulse_new.models.operator_models import OperatorModel
from qiskit.providers.aer.pulse_new.de.DE_Problems import BMDE_Problem

class TestDE_Problems(unittest.TestCase):

    def setUp(self):
        self.X = Operator.from_label('X')
        self.Y = Operator.from_label('Y')
        self.Z = Operator.from_label('Z')

        # define a basic model
        w = 2.
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2,
                     -1j * 2 * np.pi *  r * self.X / 2]
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
