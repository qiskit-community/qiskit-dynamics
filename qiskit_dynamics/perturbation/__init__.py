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

The function :func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation`
computes various time-dependent perturbation theory terms in the context of linear
matrix differential equations (LMDEs). Using the power series notation of the previous section,
the general setting supported by this function involves LMDE generators with power
series decompositions of the form:

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
or of a time-averaged generator *in the toggling frame of the unperturbed generator* :math:`G_0(t)`.
Denoting :math:`V(t_0, t)` the solution of the LMDE with generator :math:`G_0(t)`
over the interval :math:`[t_0, t]`, the generator :math:`G` in the toggling frame of :math:`G_0(t)`
is given by:

.. math::

    \tilde{G}(t, c_1, \dots, c_r) =
            \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \tilde{A}_I(t),

with :math:`\tilde{A}_I(t) = V(t_0, t)^\dagger A_I(t)V(t_0, t)`.

Denoting :math:`U(t_0, t_f, c_1, \dots, c_r)` the solution of the LMDE with generator
:math:`\tilde{G}` over the interval :math:`[t_0, t_f]`, the Dyson series provides a means of
expanding :math:`U(t_0, t_f, c_1, \dots, c_r)` in 'powers' of :math:`\tilde{G}`, and similarly
the Magnus expansion provides an expansion for a matrix :math:`\Omega(t_0, t_f, c_1, \dots, c_r)`
satisfying :math:`U(t_0, t_f, c_1, \dots, c_r) = \exp(\Omega(t_0, t_f, c_1, \dots, c_r))`,
also in 'powers' of :math:`\tilde{G}`. By further expanding :math:`\tilde{G}` into its
power series decomposition and combining terms in the Dyson and Magnus expansions
according to the perturbation parameters, the symmetric Dyson series gives:

.. math::

    U(t_0, t_f, c_1, \dots, c_r) =
            \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \mathcal{D}_I(t_0, t_f)


and similarly, the symmetric Magnus expansion gives:

.. math::

    \Omega(t_0, t_f, c_1, \dots, c_r) =
            \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \mathcal{O}_I(t_0, t_f)


Given specification of the functions :math:`G_0(t)` and the non-zero perturbation terms
:math:`A_I(t)` specified as python callables,
:func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation` numerically computes a desired
list of the :math:`\mathcal{D}_I(t_0, t_f)` or :math:`\mathcal{O}_I(t_0, t_f)`. It may also
be used to compute Dyson-like integrals using the algorithm in []. See the function doc string
for details.



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
