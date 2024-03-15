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

.. warning::

    The ``array`` and ``dispatch`` submodules of Qiskit Dynamics have been deprecated as of version
    0.5.0. The use of the ``Array`` class is no longer required to work with different array
    libraries in Qiskit Dynamics, and is broken in some cases. Refer to the :ref:`user guide entry
    on using different array libraries with Qiskit Dynamics <how-to use different array libraries>`.
    Users can now work directly with the supported array type of their choice, without the need to
    wrap them to enable dispatching. The ``array`` and ``dispatch`` submodules will be removed in
    version 0.6.0.


This module contains dispatch methods used by the :class:`~qiskit_dynamics.array.Array` class.


Dispatch Functions
==================

.. autosummary::
    :toctree: ../stubs/

    asarray
    requires_backend
"""

# Import dispatch utilities
from .dispatch import (
    asarray,
    requires_backend,
)

# Register backends
from .backends import *
