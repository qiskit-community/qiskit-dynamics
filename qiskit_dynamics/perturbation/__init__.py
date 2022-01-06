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

Tools for computing and utilizing time-dependent perturbation theory terms.
This module-level documentation outlines mathematical notation and data-structure conventions
needed to understand the functions and classes contained herein.


Power series
============

Perturbative expansions are typically expressed as power series decompositions,
and we outline here the conventions used to represent power series in this module.

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

    Throughout this module, index multisets are represented as lists of integers,
    with the canonical representation of an index multiset assumed to be sorted in non-decreasing
    order.

The :class:`~qiskit_dynamics.perturbation.MatrixPolynomial` class represents a matrix-valued
multivariable polynomial (i.e. a truncated power series), with methods for evaluation.


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


:func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation` may be used to compute
terms in either the symmetric Dyson series or symmetric Magnus expansion [5].
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
under certain conditions [2, 3].

:func:`~qiskit_dynamics.perturbation.solve_lmde_perturbation` numerically computes a desired
list of the :math:`\mathcal{D}_I(t_0, t_f)` or :math:`\mathcal{O}_I(t_0, t_f)`. It may also
be used to compute Dyson-like integrals using the algorithm in [4]. Results are returned in a
:class:`PerturbationResults` objects which is a data container with some functionality for
indexing and accessing specific perturbation terms. See the function documentation for details.


Perturbative Solvers
====================

Perturbative solvers!


References
==========

This is a test. Why isn't this rendering as a legitimate section?

1. F. Dyson, *The radiation theories of Tomonaga, Schwinger, and Feynman*,
   Phys. Rev. 75, 486-502
2. W. Magnus, *On the exponential solution of differential equations*
   *for a linear operator*, Commun. Pure Appl. Math. 7, 649-73
3. S. Blanes, F. Casas, J. Oteo, J. Ros, *The Magnus expansion and some*
   *of its applications*, Phys. Rep. 470, 151-238
4. H. Haas, D. Puzzuoli, F. Zhang, D. Cory, *Engineering Effective Hamiltonians*,
   New J. Phys. 21, 103011 (2019).
5. Forthcoming

Perturbation module API
=======================

.. autosummary::
    :toctree: ../stubs/

    MatrixPolynomial
    solve_lmde_perturbation
    PerturbationResults
    PerturbativeSolver
"""

from .power_series_utils import MatrixPolynomial
from .solve_lmde_perturbation import solve_lmde_perturbation
from .perturbation_results import PerturbationResults
from .perturbative_solvers import PerturbativeSolver
