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
# pylint: disable=invalid-name

"""
Dyson series based solver.
"""

from typing import Optional, List, Union

from scipy.integrate._ivp.ivp import OdeResult

from multiset import Multiset

from qiskit.quantum_info import Operator

from qiskit_dynamics import Signal, RotatingFrame
from qiskit_dynamics.array import Array

from .expansion_model import ExpansionModel
from .perturbative_solver import _PerturbativeSolver, _perturbative_solve, _perturbative_solve_jax


class DysonSolver(_PerturbativeSolver):
    r"""Solver for linear matrix differential equations based on the Dyson series.

    This class implements the Dyson-series based solver presented in
    [:footcite:`puzzuoli_sensitivity_2022`], which is a variant of the *Dysolve* algorithm
    originally introduced in [:footcite:p:`shillito_fast_2020`].
    This solver applies to linear matrix differential equations with generators of the form:

    .. math::

        G(t) = G_0 + \sum_{j=1}^s \textnormal{Re}[f_j(t) e^{i 2 \pi \nu_j t}]G_j,

    and solves the LMDE in the rotating frame of :math:`G_0`, which is assumed to be
    anti-Hermitian. I.e. it solves the LMDE with generator:

    .. math::

        \tilde{G}(t) = \sum_{j=1}^s \textnormal{Re}[f_j(t) e^{i 2 \pi \nu_j t}]\tilde{G}_j(t),

    with :math:`\tilde{G}_i(t) = e^{-t G_0} G_i e^{tG_0}`. The solver is *fixed-step*,
    with step size :math:`\Delta t` being defined at instantiation,
    and solves over each step by computing a truncated Dyson series. See the
    :ref:`Time-dependent perturbation theory and multi-variable
    series expansions review <perturbation review>` for a description of the Dyson series.

    At instantiation, the following parameters, which define the structure of the Dyson series
    used, are fixed:

    - The step size :math:`\Delta t`,
    - The operator structure :math:`G_0`, :math:`G_i`,
    - The reference frequencies :math:`\nu_j`,
    - Approximation schemes for the envelopes :math:`f_j` over each time step (see below), and
    - The Dyson series terms to keep in the truncation.

    A 'compilation' or 'pre-computation' step computing the truncated expansion occurs
    at instantiation. Once instantiated, the LMDE can be solved repeatedly for different lists of
    envelopes :math:`f_1(t), \dots, f_s(t)` by calling the :meth:`solve` method with the
    initial time ``t0`` and number of time-steps ``n_steps`` of size :math:`\Delta t`.
    The list of envelopes are specified as :class:`~qiskit_dynamics.signals.Signal` objects,
    whose carrier frequencies are automatically shifted to the reference frequencies :math:`\nu_j`,
    with the frequency difference, and any phase, being absorbed into the envelope.

    More explicitly, the process of solving over an interval :math:`[t_0, t_0 + \Delta t]`
    is as follows. After shifting the carrier frequencies, the resulting envelopes are approximated
    using a discrete Chebyshev transform, whose orders for each signal is given by
    ``chebyshev_orders``. That is, for :math:`t \in [t_0, t_0 + \Delta t]`, each envelope is
    approximated as:

    .. math::

        f_j(t) \approx \sum_{m=0}^{d_j} f_{j,m}T_m(t-t_0)

    where :math:`T_m(\cdot)` are the Chebyshev polynomials over the interval,
    :math:`f_{j,m}` are the approximation coefficients attained via Discrete Chebyshev Transform,
    and :math:`d_j` is the order of approximation used for the given envelope.
    Using:

    .. math::

        \textnormal{Re}[f_{j,m}T_m(t-t_0)e^{i2 \pi \nu_j t}] =
        \textnormal{Re}[f_{j,m}e^{i 2 \pi \nu_j t_0}] \cos(2 \pi \nu_j (t-t_0))T_m(t-t_0) \\
        + \textnormal{Im}[f_{j,m}e^{i 2 \pi \nu_j t_0}] \sin(-2 \pi \nu_j (t-t_0))T_m(t-t_0)

    The generator is approximately decomposed as

    .. math::

        \tilde{G}(t) \approx \sum_{j=1}^s \sum_{m=0}^{d_j}
        \textnormal{Re}[f_{j,m}e^{i 2 \pi \nu_j t_0}]
        \cos(2 \pi \nu_j (t-t_0))T_m(t-t_0) \tilde{G}_j \\
        + \sum_{j=1}^s \sum_{m=0}^{d_j}
        \textnormal{Im}[f_{j,m}e^{i 2 \pi \nu_j t_0}]
        \sin(- 2 \pi \nu_j (t-t_0))T_m(t-t_0) \tilde{G}_j

    The multivariable Dyson series is then computed relative to the
    above decomposition, with the variables being the
    :math:`\textnormal{Re}[f_{j,m}e^{i 2 \pi \nu_j t_0}]` and
    :math:`\textnormal{Im}[f_{j,m}e^{i 2 \pi \nu_j t_0}]`, and the operators being
    :math:`\cos(2 \pi \nu_j (t-t_0))T_m(t-t_0) G_j` and
    :math:`\sin(- 2 \pi \nu_j (t-t_0))T_m(t-t_0) G_j`. As shown in
    [:footcite:`puzzuoli_sensitivity_2022`, :footcite:p:`shillito_fast_2020`],
    the multivariable Dyson series for intervals of length :math:`\Delta t`
    with different starting times are related via a simple frame change, and as such
    these need only be computed once, and this makes up the 'pre-computation' step of this
    object.

    The optional argument ``include_imag`` can be used to control, on a signal by signal basis,
    whether or not the imaginary terms

    .. math::

        \textnormal{Im}[f_{j,m}e^{i 2 \pi \nu_j t_0}]
        \sin(- 2 \pi \nu_j (t-t_0))T_m(t-t_0) \tilde{G}_j

    are included in the scheme. In generality they are required, but in special cases they are not
    necessary, such as when :math:`\nu_j = 0`, or if :math:`f_j(t)`, including phase, is purely
    real. By default all such terms are included.

    .. footbibliography::
    """

    def __init__(
        self,
        operators: List[Operator],
        rotating_frame: Union[Array, Operator, RotatingFrame, None],
        dt: float,
        carrier_freqs: Array,
        chebyshev_orders: List[int],
        expansion_order: Optional[int] = None,
        expansion_labels: Optional[List[Multiset]] = None,
        integration_method: Optional[str] = None,
        include_imag: Optional[List[bool]] = None,
        **kwargs,
    ):
        r"""Initialize.

        Args:
            operators: List of constant operators specifying the operators with signal coefficients.
            rotating_frame: Rotating frame to setup the solver in.
                            Must be Hermitian or anti-Hermitian.
            dt: Fixed step size to compile to.
            carrier_freqs: Carrier frequencies of the signals in the generator decomposition.
            chebyshev_orders: Approximation degrees for each signal over the interval [0, dt].
            expansion_order: Order of perturbation terms to compute up to. Specifying this
                             argument results in computation of all terms up to the given order.
                             Can be used in conjunction with ``expansion_terms``.
            expansion_labels: Specific perturbation terms to compute. If both ``expansion_order``
                              and ``expansion_terms`` are specified, then all terms up to
                              ``expansion_order`` are computed, along with the additional terms
                              specified in ``expansion_terms``. Labels are specified either as
                              ``Multiset`` or as valid arguments to the ``Multiset`` constructor.
                              This function further requires that ``Multiset``\s consist only of
                              non-negative integers.
            integration_method: ODE solver method to use when computing perturbation terms.
            include_imag: List of bools determining whether to keep imaginary components in
                          the signal approximation. Defaults to True for all signals.
            kwargs: Additional arguments to pass to the solver when computing perturbation terms.
        """
        model = ExpansionModel(
            operators=operators,
            rotating_frame=rotating_frame,
            dt=dt,
            carrier_freqs=carrier_freqs,
            chebyshev_orders=chebyshev_orders,
            expansion_method="dyson",
            expansion_order=expansion_order,
            expansion_labels=expansion_labels,
            integration_method=integration_method,
            include_imag=include_imag,
            **kwargs,
        )
        super().__init__(model=model)

    def _solve(self, t0: float, n_steps: int, y0: Array, signals: List[Signal]) -> OdeResult:
        ys = None
        if Array.default_backend() == "jax":
            single_step = lambda x: self.model.evaluate(x).data
            ys = [y0, _perturbative_solve_jax(single_step, self.model, signals, y0, t0, n_steps)]
        else:
            single_step = lambda coeffs, y: self.model.evaluate(coeffs) @ y
            ys = [y0, _perturbative_solve(single_step, self.model, signals, y0, t0, n_steps)]

        return OdeResult(t=[t0, t0 + n_steps * self.model.dt], y=ys)
