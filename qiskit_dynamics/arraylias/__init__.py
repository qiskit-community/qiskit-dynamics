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

This module contains Qiskit Dynamics-global extensions of the default NumPy and SciPy aliases
provided by `Arraylias <https://qiskit-extensions.github.io/arraylias/>`_. These are used to manage
dispatching of array operations for the different array types supported by Qiskit Dynamics. They
have been configured beyond the `Arraylias <https://qiskit-extensions.github.io/arraylias/>`_
defaults to additionally support both `JAX <https://jax.readthedocs.io/en/latest/>`_ and SciPy
sparse types. The following table summarizes the registered libraries and respective types.

.. list-table:: Supported libraries
    :widths: 10 70
    :header-rows: 1

    * - ``array_library``
      - Registered types
    * - ``"numpy"``
      - Default supported by the Arraylias NumPy and SciPy aliases.
    * - ``"jax"``
      - Default supported by the Arraylias NumPy and SciPy aliases.
    * - ``"jax_sparse"``
      - The JAX ``jax.experimental.sparse.BCOO`` array type.
    * - ``"scipy_sparse"``
      - Subclasses of the ``scipy.sparse.spmatrix`` sparse base class. When instantiating SciPy
        sparse arrays, the alias will specifically create ``scipy.sparse.csr_matrix`` instances.

The global configured aliases and the aliased libraries can be imported from
``qiskit_dynamics.arraylias``, and are summarized by the following table.

.. list-table:: Configured Arraylias objects
    :widths: 10 70
    :header-rows: 1

    * - Arraylias object
      - Description
    * - ``DYNAMICS_NUMPY_ALIAS``
      - Qiskit Dynamics-global NumPy alias.
    * - ``DYNAMICS_SCIPY_ALIAS``
      - Qiskit Dynamics-global SciPy alias.
    * - ``DYNAMICS_NUMPY``
      - Qiskit Dynamics-global aliased NumPy library.
    * - ``DYNAMICS_SCIPY``
      - Qiskit Dynamics-global aliased SciPy library.
"""

from .alias import (
    DYNAMICS_NUMPY_ALIAS,
    DYNAMICS_SCIPY_ALIAS,
    DYNAMICS_NUMPY,
    DYNAMICS_SCIPY,
    ArrayLike,
    requires_array_library,
)
