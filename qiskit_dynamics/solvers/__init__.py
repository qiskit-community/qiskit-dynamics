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

r"""
========================================
Solvers (:mod:`qiskit_dynamics.solvers`)
========================================

.. currentmodule:: qiskit_dynamics.solvers

This module provides classes and functions for solving differential equations.

The following table summarizes the solver interfaces exposed in this module.
Broadly, the *solver functions* are low-level interfaces exposing numerical methods for
solving particular classes of differential equations, while the *solver classes*
provide high level interfaces for solving models of quantum systems.

.. list-table:: Solver interfaces
   :widths: 10 50
   :header-rows: 1

   * - Object
     - Description
   * - :class:`~qiskit_dynamics.solvers.Solver`
     - High level solver class for both Hamiltonian and Lindblad dynamics.
       Automatically constructs the relevant model type based on system details, and
       the :meth:`~qiskit_dynamics.solvers.Solver.solve` method automatically handles
       ``qiskit.quantum_info`` input types.
   * - :func:`~qiskit_dynamics.solvers.solve_ode`
     - Low level solver function for ordinary differential equations:

       .. math::

            \dot{y}(t) = f(t, y(t)),

       for :math:`y(t)` arrays of arbitrary shape.
   * - :func:`~qiskit_dynamics.solvers.solve_lmde`
     - Low level solver function for linear matrix differential equations in *standard form*:

       .. math::
            \dot{y}(t) = G(t)y(t),

       where :math:`G(t)` is either a callable or a ``qiskit_dynamics``
       model type, and :math:`y(t)` arrays of suitable shape for the matrix multiplication above.


Solver classes
==============

.. autosummary::
   :toctree: ../stubs/

   Solver

Solver functions
================

.. autosummary::
   :toctree: ../stubs/

   solve_ode
   solve_lmde
"""

from .solver_functions import solve_ode, solve_lmde
from .solver_classes import Solver
