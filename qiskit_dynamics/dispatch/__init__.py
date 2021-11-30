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

Dispatcher Object
=================

.. autosummary::
    :toctree: ../stubs/

    default_dispatcher


Utility Functions
=================

.. autosummary::
    :toctree: ../stubs/

    array
    asarray
    infer_library
    requires_library


Dispatcher Classes
==================

    DynamicDispatcher
    StaticDispatcher
"""

from .default_dispatcher import default_dispatcher
from .static_dispatcher import StaticDispatcher
from .dynamic_dispatcher import DynamicDispatcher

from .functions import (
    array,
    asarray,
    infer_library,
    requires_library,
)

# Register built-in supported libraries
from .dispatcher_libraries import *

# DEPRECATED

# Depreacted dispatch functions
from .dispatch import (
    set_default_backend,
    default_backend,
    available_backends,
    backend_types,
    requires_backend,
)

# Deprecated import paths
from .array import Array

from .wrap import wrap
