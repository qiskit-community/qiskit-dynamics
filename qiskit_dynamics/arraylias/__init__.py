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

r"""
============================================
Arraylias (:mod:`qiskit_dynamics.arraylias`)
============================================

.. currentmodule:: qiskit_dynamics.arraylias

Qiskit Dynamics uses `Arraylias <https://qiskit-extensions.github.io/arraylias/>`_ to manage
dispatching of array operations for different array types coming from different array libraries.

This module contains Qiskit Dynamics-global extensions of the default NumPy and SciPy aliases
provided by Arraylias <https://qiskit-extensions.github.io/arraylias/>`_, which have been configured
to support the `JAX <https://jax.readthedocs.io/en/latest/>`_ `BCOO` sparse array type, as well as
the sparse types offered by SciPy.

.. autosummary::
   :toctree: ../stubs/

   DYNAMICS_NUMPY_ALIAS
   DYNAMICS_SCIPY_ALIAS
   DYNAMICS_NUMPY
   DYNAMICS_SCIPY
"""

from .alias import (
    DYNAMICS_NUMPY_ALIAS,
    DYNAMICS_SCIPY_ALIAS,
    DYNAMICS_NUMPY,
    DYNAMICS_SCIPY,
    ArrayLike,
    requires_array_library,
)
