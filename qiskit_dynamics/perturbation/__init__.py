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

This module contains tools for numerically computing and utilizing perturbation theory terms.
Perturbation theory is an advanced topic; a brief review of the concepts and notation required
to understand the contents in this module are given in the
:ref:`Time-dependent perturbation theory and multi-variable
series expansions review <perturbation review>` discussion.

.. _td perturbation theory:

Time-dependent perturbation theory
==================================

The function :func:`.solve_lmde_perturbation` computes
Dyson series :footcite:`dyson_radiation_1949` and
Magnus expansion :footcite:`magnus_exponential_1954,blanes_magnus_2009` terms
in a multi-variable setting via algorithms in :footcite:`puzzuoli_sensitivity_2022`.
It can also be used to compute Dyson-like integrals using the algorithm in
:footcite:`haas_engineering_2019`. Results are returned in either a :class:`PowerSeriesData`
or :class:`DysonLikeData` class, which are data classes with functionality for indexing
and accessing specific perturbation terms. See the function documentation for further details.

Truncated power-series representation and multisets
===================================================

The class :class:`.ArrayPolynomial` represents an array-valued multivariable polynomial
(i.e. a truncated power series), and provides functionality for
both evaluating and transforming array-valued polynomials.

This module makes use of the `multiset package <https://pypi.org/project/multiset/>`_ for
indexing multi-variable power series. See the
:ref:`multiset and power series notation section <multiset power series>`
of the perturbation review for an explanation of this convention.


Perturbation module functions
=============================

.. autosummary::
    :toctree: ../stubs/

    solve_lmde_perturbation


Perturbation module classes
===========================

.. autosummary::
    :toctree: ../stubs/

    ArrayPolynomial
    PowerSeriesData
    DysonLikeData

.. footbibliography::
"""

from .array_polynomial import ArrayPolynomial
from .solve_lmde_perturbation import solve_lmde_perturbation
from .perturbation_data import PowerSeriesData, DysonLikeData
