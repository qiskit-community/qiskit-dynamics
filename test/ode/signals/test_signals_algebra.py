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
# pylint: disable=invalid-name

"""
Tests for algebraic operations on signals.
"""

import numpy as np

from qiskit_ode.signals import Signal, Constant, DiscreteSignal
from qiskit_ode.signals.signals import SignalSum, DiscreteSignalSum
from qiskit_ode.dispatch import Array

from ..common import QiskitOdeTestCase, TestJaxBase

try:
    from jax import jit, grad
    import jax.numpy as jnp
except ImportError:
    pass
