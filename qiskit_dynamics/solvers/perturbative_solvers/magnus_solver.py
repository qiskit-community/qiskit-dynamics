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
Magnus expansion based solver.
"""

from typing import Optional, List, Union

from scipy.linalg import expm
from scipy.integrate._ivp.ivp import OdeResult

from multiset import Multiset

from qiskit.quantum_info import Operator

from qiskit_dynamics import Signal, RotatingFrame
from qiskit_dynamics.array import Array

from .expansion_model import ExpansionModel
from .perturbative_solver import _PerturbativeSolver, _perturbative_solve, _perturbative_solve_jax

try:
    from jax.scipy.linalg import expm as jexpm
except ImportError:
    pass


class MagnusSolver(_PerturbativeSolver):
    """Solver for linear matrix differential equations based on the Magnus expansion.

    This class implements the Magnus expansion-based solver presented in
    [:footcite:`puzzuoli_sensitivity_2022`], which is a Magnus expansion variant of the
    *Dysolve* algorithm originally introduced in [:footcite:p:`shillito_fast_2020`]. Its
    setup and behaviour are the same as as the :class:`~qiskit_dynamics.solvers.DysonSolver`
    class, with the sole exception being that it uses a truncated Magnus expansion
    and matrix exponentiation to solve over a single time step. See the
    :ref:`Time-dependent perturbation theory and multi-variable
    series expansions review <perturbation review>` for a description of the Magnus expansion,
    and the documentation for :class:`~qiskit_dynamics.solvers.DysonSolver` for more detailed
    behaviour of this class.
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
            expansion_method="magnus",
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
            single_step = lambda x: self.model.Udt @ jexpm(self.model.evaluate(x).data)
            ys = [y0, _perturbative_solve_jax(single_step, self.model, signals, y0, t0, n_steps)]
        else:
            single_step = lambda coeffs, y: self.model.Udt @ expm(self.model.evaluate(coeffs)) @ y
            ys = [y0, _perturbative_solve(single_step, self.model, signals, y0, t0, n_steps)]

        return OdeResult(t=[t0, t0 + n_steps * self.model.dt], y=ys)
