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

"""
Base class for Jax tests.
"""

import unittest
from qiskit_ode import dispatch


class TestJaxBase(unittest.TestCase):
    """Base class with setUpClass and tearDownClass for setting jax as the
    default backend.

    Test cases that inherit from this class will automatically work with jax
    backend.
    """

    @classmethod
    def setUpClass(cls):
        try:
            # pylint: disable=import-outside-toplevel
            from jax import config
            config.update("jax_enable_x64", True)
        except Exception:
            raise unittest.SkipTest('Skipping jax tests.')

        dispatch.set_default_backend('jax')

    @classmethod
    def tearDownClass(cls):
        """Set numpy back to the default backend."""
        dispatch.set_default_backend('numpy')
