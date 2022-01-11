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

This module contains tools for computing and utilizing time-dependent perturbation theory terms.
Due to the advanced nature of this topic, the documentation below outlines the
mathematical notation and data-structure conventions of the module, and also gives
a detailed overview of the purposes of the user-facing components.


Power series
============

Perturbative expansions are typically expressed as power series decompositions,
and the conventions for representing power series in this module are detailed here.

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

With this notation, we represent a power series in :math:`r` variables as:

.. math::

    f(c_1, \dots, c_r) = M_0 + \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I M_I,

where :math:`M_0` is the constant term, and the :math:`M_I` are the matrix-valued power-series
coefficients.

.. note::

    Throughout this module, index multisets are represented as a ``List`` of ``int``\s,
    with the canonical representation of an index multiset assumed to be sorted in non-decreasing
    order.

The :class:`~qiskit_dynamics.perturbation.MatrixPolynomial` class represents a matrix-valued
multivariable polynomial (i.e. a truncated power series), with methods for evaluation.


.. _td perturbation theory:

Time dependent perturbation theory
==================================

The function :func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation`
computes various time-dependent perturbation theory terms related to the Dyson series
[:footcite:`dyson_radiation_1949`] and Magnus expansion [:footcite:p:`magnus_exponential_1954`],
used in matrix differential equations (LMDEs). Using the power series notation of the
previous section, the general setting supported by this function involves LMDE generators
with power series decompositions of the form:

.. math::

    G(t, c_1, \dots, c_r) = G_0(t) + \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I A_I(t),

where

    - :math:`G_0(t)` is the unperturbed generator,
    - The :math:`A_I(t)` give the time-dependent operator form of the perturbations, and
    - The expansion parameters :math:`c_1, \dots, c_r` are viewed as the perturbation parameters.

.. note::

    The above is written as an infinite power series, but of course, in practice,
    the function assumes only a finite number of the :math:`A_I(t)` are specified as being
    non-zero.

:func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation` enables computation of a
finite number of power series decomposition coefficients of either the solution itself,
or of a time-averaged generator *in the toggling frame of the unperturbed generator* :math:`G_0(t)`
[:footcite:`evans_timedependent_1967`, :footcite:`haeberlen_1968`].
Denoting :math:`V(t_0, t)` the solution of the LMDE with generator :math:`G_0(t)`
over the interval :math:`[t_0, t]`, the generator :math:`G` in the toggling frame of :math:`G_0(t)`
is given by:

.. math::

    \tilde{G}(t, c_1, \dots, c_r) =
            \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \tilde{A}_I(t),

with :math:`\tilde{A}_I(t) = V(t_0, t)^\dagger A_I(t)V(t_0, t)`.


:func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation` may be used to compute
terms in either the symmetric Dyson series or symmetric Magnus expansion
[forthcoming].
Denoting :math:`U(t_0, t_f, c_1, \dots, c_r)` the solution of the LMDE with generator
:math:`\tilde{G}` over the interval :math:`[t_0, t_f]`, the symmetric Dyson series
directly expands the solution as a power series in the :math:`c_1, \dots, c_r`:

.. math::

    U(t_0, t_f, c_1, \dots, c_r) =
            \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \mathcal{D}_I(t_0, t_f).

The symmetric Magnus expansion similarly gives a power series decomposition of the
time-averaged generator:

.. math::

    \Omega(t_0, t_f, c_1, \dots, c_r) =
            \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \mathcal{O}_I(t_0, t_f),

which satisfies :math:`U(t_0, t_f, c_1, \dots, c_r) = \exp(\Omega(t_0, t_f, c_1, \dots, c_r))`
under certain conditions [:footcite:`magnus_exponential_1954`, :footcite:`blanes_magnus_2009`].

:func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation` numerically computes a desired
list of the :math:`\mathcal{D}_I(t_0, t_f)` or :math:`\mathcal{O}_I(t_0, t_f)`
using the algorithm in [forthcoming]. It may also be used to compute Dyson-like
integrals using the algorithm in [:footcite:`haas_engineering_2019`]. Results are returned in a
:class:`PerturbationResults` objects which is a data container with some functionality for
indexing and accessing specific perturbation terms. See the function documentation for further
details.


Perturbative Solvers
====================

The :class:`~qiskit_dynamics.perturbation.PerturbativeSolver` class provides two
solvers built using the symmetric Dyson and Magnus expansions, as outlined in [forthcoming].

.. note::

    The principles and core ideas of the methods were outlined in the Dyson-based *Dysolve*
    algorithm given in [:footcite:p:`shillito_fast_2020`], however the Magnus version and
    specific algorithms and framing of the problem are as given in [forthcoming].

The methods are specialized to LMDEs whose generators are decomposed as:

.. math::

    G(t) = G_0 + \sum_j Re[f_j(t)e^{i2\pi\nu_jt}]G_j,

and take time steps of a pre-defined fixed size :math:`\Delta t` by either computing
a truncated symmetric Dyson series, or taking the exponential of a truncated
symmetric Magnus expansion.


Perturbation module API
=======================

.. autosummary::
    :toctree: ../stubs/

    MatrixPolynomial
    solve_lmde_perturbation
    PerturbationResults
    PerturbativeSolver

.. footbibliography::
"""

from .power_series_utils import MatrixPolynomial
from .solve_lmde_perturbation import solve_lmde_perturbation
from .perturbation_results import PerturbationResults
from .perturbative_solver import PerturbativeSolver
