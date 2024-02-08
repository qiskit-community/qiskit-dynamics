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
# pylint: disable=invalid-name

"""
Wrapper for jax.experimental.ode.odeint
"""

from typing import Callable, Optional
from scipy.integrate._ivp.ivp import OdeResult

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics.arraylias import ArrayLike, requires_array_library

from .solver_utils import merge_t_args_jax, trim_t_results_jax

try:
    from jax.experimental.ode import odeint
except ImportError:
    pass


@requires_array_library("jax")
def jax_odeint(
    rhs: Callable,
    t_span: ArrayLike,
    y0: ArrayLike,
    t_eval: Optional[ArrayLike] = None,
    **kwargs,
):
    """Routine for calling `jax.experimental.ode.odeint`

    Args:
        rhs: Callable of the form :math:`f(t, y)`
        t_span: Interval to solve over.
        y0: Initial state.
        t_eval: Optional list of time points at which to return the solution.
        **kwargs: Optional arguments to be passed to ``odeint``.

    Returns:
        OdeResult: Results object.
    """

    t_list = merge_t_args_jax(t_span, t_eval)

    # determine direction of integration
    t_direction = unp.sign(unp.asarray(t_list[-1] - t_list[0], dtype=complex))

    results = odeint(
        lambda y, t: rhs(unp.real(t_direction * t), y) * t_direction,
        y0=unp.asarray(y0, dtype=complex),
        t=unp.real(t_direction) * unp.asarray(t_list),
        **kwargs,
    )

    results = OdeResult(t=t_list, y=results)

    return trim_t_results_jax(results, t_eval)
