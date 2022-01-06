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

Power series
============

As parts of this module deal with perturbation theory theory expansions as
power series, it is necessary to choose both mathematical notation and
data-structure representations of power series to work with throughout.
The chosen conventions are outlined here.

.. note::

    The power series are always assumed to be *matrix-valued* power-series of complex variables -
    i.e. the coefficients are matrices, and the variables are complex scalars.

Mathematically, we represent a matrix-valued power-series using an
*index multiset* notation. Let :math:`\mathcal{I}_k(r)` represent the set of multisets of
size :math:`k` whose elements are drawn from :math:`\{1, \dots, r\}`. Furthermore, given
:math:`r` scalars :math:`c_1, \dots, c_r`, and an index multiset :math:`I \in \mathcal{I}_k(r)`,
with elements :math:`I = (i_1, \dots, i_k)`, denote

.. math::

    c_I = c_{i_1} \times \dots \times c_{i_k}.

I.e. :math:`c_I` is the :math:`k`-fold product of the scalars :math:`c_j` whose indices
are given by the multiset :math:`I`. Some example usages of this notation are:

    - :math:`c_{(1, 2)} = c_1 c_2`,
    - :math:`c_{(1, 1)} = c_1^2`, and
    - :math:`c_{(1, 2, 2, 3)} = c_1 c_2^2 c_3`.

With this notation, we concisely represent a power series in :math:`r` variables as:

.. math::

    f(c_1, \dots, c_r) = M_0 + \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I M_I,

where :math:`M_0` is the constant term, and the :math:`M_I` are the matrix-valued power-series
coefficients.

Within this module, index multisets are represented as lists of integers, with the canonical
representation of an index multiset assumed to be sorted.
The class :class:`~qiskit_dynamics.perturbation.MatrixPolynomial` is
used for representing and evaluating truncated power series.


Time dependent perturbation theory
==================================

Dyson and Magnus!

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
