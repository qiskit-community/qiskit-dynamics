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
====================================================
Differential equations module (:mod:`qiskit_ode.de`)
====================================================

.. currentmodule:: qiskit_ode.de

DE Problems
===========

.. currentmodule:: qiskit_ode.de.de_problems

.. autosummary::
   :toctree: ../stubs/

   ODEProblem
   LMDEProblem

Solvers
=======

.. currentmodule:: qiskit_ode.de.solve

.. autosummary::
   :toctree: ../stubs/

   solve
"""

from .de_problems import ODEProblem, LMDEProblem
from .solve import solve
