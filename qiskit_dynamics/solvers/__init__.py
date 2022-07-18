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

Table :ref:`1 <standard solver table>` summarizes the standard solver interfaces exposed in
this module. It includes a high level class :class:`~qiskit_dynamics.solvers.Solver`
for solving models of quantum systems, as well as low-level functions for solving both
ordinary differential equations :math:`\dot{y}(t) = f(t, y(t))` and linear matrix differential
equations :math:`\dot{y}(t) = G(t)y(t)`.

Additionally, this module contains more specialized solvers for linear matrix differential
equations based on perturbative expansions, described :ref:`below <perturbative solvers>`.

.. _standard solver table:

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

       for :math:`y(t)` arrays of arbitrary shape and :math:`f` specified as an arbitrary callable.
   * - :func:`~qiskit_dynamics.solvers.solve_lmde`
     - Low level solver function for linear matrix differential equations in *standard form*:

       .. math::
            \dot{y}(t) = G(t)y(t),

       where :math:`G(t)` is either a callable or a ``qiskit_dynamics``
       model type, and :math:`y(t)` arrays of suitable shape for the matrix multiplication above.

.. _perturbative solvers:

Perturbative Solvers
====================

The classes :class:`~qiskit_dynamics.solvers.DysonSolver` and
:class:`~qiskit_dynamics.solvers.MagnusSolver` implement advanced solvers
detailed in [:footcite:`puzzuoli_sensitivity_2022`], with the
:class:`~qiskit_dynamics.solvers.DysonSolver` implementing a variant of the *Dysolve*
algorithm originally introduced in [:footcite:p:`shillito_fast_2020`].

The solvers are specialized to linear matrix differential equations with :math:`G(t)`
decomposed as:

.. math::

    G(t) = G_0 + \sum_j Re[f_j(t)e^{i2\pi\nu_jt}]G_j,

and are fixed step with a pre-defined step size :math:`\Delta t`. The differential equation is
solved by either computing a truncated Dyson series, or taking the exponential of a truncated
Magnus expansion.

Add reference to both userguide and perturbation theory module documentation.


Solver classes
==============

.. autosummary::
   :toctree: ../stubs/

   Solver
   DysonSolver
   MagnusSolver

Solver functions
================

.. autosummary::
   :toctree: ../stubs/

   solve_ode
   solve_lmde

.. footbibliography::
"""

from .solver_functions import solve_ode, solve_lmde
from .solver_classes import Solver
from .perturbative_solvers.dyson_solver import DysonSolver
from .perturbative_solvers.magnus_solver import MagnusSolver
