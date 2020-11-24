# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Array with numpy compatibility and multiple backend support.

This is a prototype of the qiskit.quantum_info.dispatch module and
should eventually be imported from there once it has been added to qiskit-terra.
"""

# Import Array
from .array import Array

# Import wrapper function
from .wrap import wrap

# Import dispatch utilities
from .dispatch import (set_default_backend,
                       default_backend,
                       available_backends,
                       backend_types,
                       asarray)

# Register backends
from .backends import *

# If only one backend is available, set it as the default
if len(available_backends()) == 1:
    set_default_backend(available_backends()[0])

# Monkey patch quantum info
from .patch_qi import *
