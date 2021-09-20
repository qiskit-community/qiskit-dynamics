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

This module provides functions and classes for solving differential equations.

The class :class:`qiskit_dynamics.solvers.Solver` provides a higher level
interface for solving both Schrodinger and Lindblad master equation dynamics.
After instantiating with the details of the system, the
:meth:`~qiskit_dynamics.solvers.Solver.solve` method provides automatic handling of input
state types based on the underlying model (e.g. a :class:`qiskit.quantum_info.Statevector`
will be automatically converted to a :class:`qiskit.quantum_info.DensityMatrix` if
simulating Lindblad dynamics).

By contrast, the solver functions provide a low level interface for solving different
types of differential equations. Modelled after the interface of ``scipy.integrate.solve_ivp``,
these functions provide access to various underlying numerical methods, with the input
state assumed to be an array.

.. list-table:: Solver functions
   :widths: 10 50
   :header-rows: 1

   * - Function
     - Class of differential equations
   * - :func:`~qiskit_dynamics.solvers.solve_ode`
     - Solves ordinary differential equations:

       .. math::

            \dot{y}(t) = f(t, y(t)),

       for :math:`y(t)` of arbitrary shape.
   * - :func:`~qiskit_dynamics.solvers.solve_lmde`
     - Solves linear matrix differential equations in *standard form*:

       .. math::
            \dot{y}(t) = G(t)y(t),

       where :math:`G(t)` is either a callable or a ``qiskit_dynamics``
       model type, and :math:`y(t)` has shape suitable for the matrix multiplication above.


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
