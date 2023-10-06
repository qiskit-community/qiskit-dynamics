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
# pylint: disable=invalid-name,no-member

"""
Test global alias instances.
"""

from functools import partial

import numpy as np
import scipy as sp

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_SCIPY as usp

from ..common import QiskitDynamicsTestCase, test_array_backends


@partial(test_array_backends, array_libraries=["numpy", "jax", "array_numpy", "array_jax"])
class TestDynamicsNumpy(QiskitDynamicsTestCase):
    """Test cases for global numpy configuration."""

    def test_simple_case(self):
        """Validate correct type and output."""
        a = self.asarray([1.0, 2.0, 3.0])
        output = unp.exp(a)
        self.assertArrayType(output)

        expected = np.exp(np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(output, expected)


@test_array_backends
class TestDynamicsScipy(QiskitDynamicsTestCase):
    """Test cases for global scipy configuration."""

    def test_simple_case(self):
        """Validate correct type and output."""
        a = self.asarray([1.0, 2.0, 3.0])
        output = usp.fft.dct(a)
        self.assertArrayType(output)

        expected = sp.fft.dct(np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(output, expected)
