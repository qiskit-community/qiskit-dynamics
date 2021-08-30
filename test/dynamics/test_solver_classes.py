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

"""
Tests for solver classes module.
"""

from .common import QiskitDynamicsTestCase, TestJaxBase

import numpy as np

from qiskit_dynamics import Solver

class TestSolver(QiskitDynamicsTestCase):
    pass

class TestSolverJax(TestSolver, TestJaxBase):
    """JAX version of TestSolver."""
    pass
