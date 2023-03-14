# -*- coding: utf-8 -*-

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

"""Configure custom instance of numpy alias for Dynamics."""

from typing import Optional

from arraylias import numpy_alias
from .array import Array


DYNAMICS_ALIAS = numpy_alias()

# Set qiskit_dynamics.array.Array to be dispatched to numpy
DYNAMICS_ALIAS.register_type(Array, "numpy")

DYNAMICS_NUMPY = DYNAMICS_ALIAS()