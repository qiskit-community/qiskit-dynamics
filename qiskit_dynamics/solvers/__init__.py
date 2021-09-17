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
===============================================================
Differential equations solvers (:mod:`qiskit_dynamics.solvers`)
===============================================================

This module provides high level functions for solving classes of
Differential Equations (DEs), described below.

1. Ordinary Differential Equations (ODEs)
#########################################

The most general class of DEs we consider are ODEs, which are of the form:

.. math::

    \dot{y}(t) = f(t, y(t)),

Where :math:`f` is called the Right-Hand Side (RHS) function.
ODEs can be solved by calling the :meth:`~qiskit_dynamics.solve_ode` function.

2. Linear Matrix Differential Equations (LMDEs)
###############################################

LMDEs are a specialized subclass of ODEs of importance in quantum theory. Most generally,
an LMDE is an ODE for which the the RHS function :math:`f(t, y)` is *linear* in the second
argument. Numerical methods for LMDEs typically assume a *standard form*

.. math::

    f(t, y) = G(t)y,

where :math:`G(t)` is a square matrix-valued function called the *generator*, and
the state :math:`y(t)` must be an array of appropriate shape. Note that any LMDE in the more
general sense (not in *standard form*) can be restructured into one of standard form via suitable
vectorization.

The function :meth:`~qiskit_dynamics.de.solve_lmde` provides access to solvers for LMDEs in
standard form, specified in terms of a representation of the generator :math:`G(t)`,
either as a Python ``Callable`` function or subclasses of
:class:`~qiskit_dynamics.models.generator_models.BaseGeneratorModel`.

Note that the numerical methods available via :meth:`~qiskit_dynamics.solve_ode`
are also available through :meth:`~qiskit_dynamics.de.solve_lmde`:

    * If the generator is supplied as a ``Callable``, the standard RHS function
      :math:`f(t, y) = G(t)y` is automatically constructed.
    * If the generator supplied is a subclass of
      :class:`~qiskit_dynamics.models.generator_models.BaseGeneratorModel` which is not in standard
      form, it is delegated to :meth:`~qiskit_dynamics.solve_ode`.


.. currentmodule:: qiskit_dynamics.solvers

.. autosummary::
   :toctree: ../stubs/

   solve_ode
   solve_lmde
   Solver
"""

from .solver_functions import solve_ode, solve_lmde
from .solver_classes import Solver
