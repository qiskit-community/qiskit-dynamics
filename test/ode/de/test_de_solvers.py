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
"""tests for DE_Solvers.py"""

import unittest
import numpy as np

from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.pulse_new.models.signals import Constant, Signal
from qiskit.providers.aer.pulse_new.models.operator_models import OperatorModel
from qiskit.providers.aer.pulse_new.de.DE_Problems import BMDE_Problem
from qiskit.providers.aer.pulse_new.de.DE_Solvers import BMDE_Solver

class TestDE_Solvers(unittest.TestCase):

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

    def test_integrate_w_basic_model(self):
        """Test integration with the basic model"""

        de_problem = BMDE_Problem(generator=self.basic_model,
                                  y0=self.y0,
                                  t0=0.)
        solver = BMDE_Solver(de_problem)

        solver.integrate(1. / self.r)

        probs = np.abs(solver.y)**2
        self.assertTrue(probs[1] > 0.999)

    def assertAlmostEqual(self, A, B, tol=10**-15):
        self.assertTrue(np.abs(A - B).max() < tol)
