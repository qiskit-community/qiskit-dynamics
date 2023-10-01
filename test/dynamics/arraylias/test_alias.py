# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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
Test global alias instances.
"""

from functools import partial

from ..common import QiskitDynamicsTestCase, test_array_backends

import numpy as np

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_SCIPY as usp


@partial(test_array_backends, backends=["numpy", "jax", "array_numpy", "array_jax"])
class TestDynamicsNumpy(QiskitDynamicsTestCase):

    def test_simple_case(self):
        """Validate correct type and output."""
        a = self.asarray([1., 2., 3.])
        output = unp.exp(a)
        self.assertTrue(isinstance(output, type(a)))

        expected = np.exp(np.array([1., 2., 3.]))
        self.assertAllClose(output, expected)


@partial(test_array_backends, backends=["numpy", "jax", "array_numpy", "array_jax"])
class TestDynamicsNumpy(QiskitDynamicsTestCase):

    def test_simple_case(self):
        """Validate correct type and output."""
        a = self.asarray([1., 2., 3.])
        output = unp.exp(a)
        self.assertTrue(isinstance(output, type(a)))

        expected = np.exp(np.array([1., 2., 3.]))
        self.assertAllClose(output, expected)