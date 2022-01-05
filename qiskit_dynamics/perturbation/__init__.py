# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
=========================================================
Perturbation Theory (:mod:`qiskit_dynamics.perturbation`)
=========================================================

.. currentmodule:: qiskit_dynamics.perturbation

This module contains tools for computing and utilizing perturbation theory terms.

Power series utilities
======================

.. autosummary::
    :toctree: ../stubs/

    MatrixPolynomial

Perturbation theory computation
===============================

.. autosummary::
    :toctree: ../stubs/

    solve_lmde_perturbation
    PerturbationResults
"""

from .power_series_utils import MatrixPolynomial
from .solve_lmde_perturbation import solve_lmde_perturbation
from .perturbation_results import PerturbationResults
