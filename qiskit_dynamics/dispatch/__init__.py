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


Functions
=========

.. autosummary::
    :toctree: ../stubs/

    default_dispatcher


Classes
=======

.. autosummary::
    :toctree: ../stubs/

    Dispatcher
    DispatcherError
"""
from .dispatcher import Dispatcher, DispatcherError
from .default_dispatcher import default_dispatcher


# DEPRECATED
from .dispatch import (
    set_default_backend,
    default_backend,
    available_backends,
    backend_types,
    requires_backend,
)
from .array import Array
from .wrap import wrap
