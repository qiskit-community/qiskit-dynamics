# -*- coding: utf-8 -*-

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
=================================================
Dispatch Module (:mod:`qiskit_dynamics.dispatch`)
=================================================

.. currentmodule:: qiskit_dynamics.dispatch

This module contains a common interface for working with array types from
multiple array libraries.

Dispatching Functions
=====================

.. autosummary::
    :toctree: ../stubs/

    dispatcher
    dispatch_function


Additional Functions
====================

.. autosummary::
    :toctree: ../stubs/

    infer_library
    registered_libraries
    validate_library
    requires_library
    asarray

Registering Array Libraries
===========================

.. autosummary::
    :toctree: ../stubs/

    registered_libraries
    is_registered_library
    registered_types
    is_registered_type
    register_function
    register_module
    register_type
"""

# Dispatcher
from .dispatcher import (
    dispatcher,
    dispatch_function,
    infer_library,
)

from .functions import (
    requires_library,
    asarray,
)

from .register import (
    registered_libraries,
    is_registered_library,
    registered_types,
    is_registered_type,
    register_function,
    register_module,
    register_type,
)

# Register built-in supported libraries
from .dispatcher_libraries import *

# Depreacted dispatch functions
from .dispatch import (
    set_default_backend,
    default_backend,
    available_backends,
    backend_types,
    asarray,
    requires_backend,
)

# Deprecated import paths
from .array import Array

from .wrap import wrap
