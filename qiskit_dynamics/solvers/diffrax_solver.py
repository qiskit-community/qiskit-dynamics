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

from typing import Callable, Optional, Union, Tuple, List
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit_dynamics.dispatch import requires_backend
from qiskit_dynamics.array import Array, wrap
from diffrax import ODETerm, Dopri5, PIDController
from diffrax import diffeqsolve as _diffeqsolve

from diffrax.solver import AbstractSolver

from .solver_utils import merge_t_args, trim_t_results
import jax.numpy as jnp


@requires_backend("jax")
def diffrax_dopri5(
    rhs: Callable,
    t_span: Array,
    y0: Array,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
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
    raise NotImplementedError

    t_list = merge_t_args(t_span, t_eval)
    # if t_eval is none, doesn't matter, but if t_eval is specified, merge assumes t_span is also np array
    # Check if diffrax handles backwards integration
    # t_list = t_list.data

    # determine direction of integration
    # t_direction = np.sign(Array(t_list[-1] - t_list[0], backend="jax", dtype=float))
    # t_direction = 1
    # t_direction = jnp.array(1)
    rhs = wrap(rhs)

    stepsize_controller = PIDController(rtol=kwargs["rtol"], atol=kwargs["atol"])

    # term = ODETerm(lambda y, t, _: (rhs(np.real(t_direction * t), y) * t_direction).data)
    term = ODETerm(lambda t, y, _: Array(rhs(t.real, y)).data)
    solver = Dopri5()

    results = diffeqsolve(
        term,
        solver,
        t0=t_list[0],
        t1=t_list[-1],
        dt0=None,
        y0=y0,
        stepsize_controller=stepsize_controller,
    )  # **kwargs

    # results = odeint(
    #     lambda y, t: rhs(np.real(t_direction * t), y) * t_direction,
    #     y0=Array(y0, dtype=float),
    #     t=np.real(t_direction) * Array(t_list),
    #     **kwargs,
    # )

    results = OdeResult(t=t_list, y=Array(results.ys, backend="jax", dtype=float))

    return trim_t_results(results, t_span, t_eval)


@requires_backend("jax")
def diffrax_solver(
    rhs: Callable,
    t_span: Array,
    y0: Array,
    method: Optional[AbstractSolver] = Dopri5(),
    t_eval: Optional[Union[Tuple, List, Array]] = None,
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
    if isinstance(method, type) and issubclass(method, AbstractSolver):
        solver = method()

    t_list = merge_t_args(t_span, t_eval)
    # if t_eval is none, doesn't matter, but if t_eval is specified, merge assumes t_span is also np array
    # Check if diffrax handles backwards integration
    # t_list = t_list.data

    # determine direction of integration
    # t_direction = np.sign(Array(t_list[-1] - t_list[0], backend="jax", dtype=float))


    # convert rhs and y0 to real
    rhs = real_rhs(rhs)
    y0 = c2r(y0)
    # rhs = wrap(rhs)
    stepsize_controller = PIDController(rtol=kwargs["rtol"], atol=kwargs["atol"])

    # term = ODETerm(lambda y, t, _: (rhs(np.real(t_direction * t), y) * t_direction).data)
    term = ODETerm(lambda t, y, _: Array(rhs(t.real, y), dtype=float).data)

    diffeqsolve = wrap(_diffeqsolve)

    results = diffeqsolve(
        term,
        solver=solver,
        t0=t_list[0],
        t1=t_list[-1],
        dt0=None,
        y0=Array(y0),
        stepsize_controller=stepsize_controller,
    )  # **kwargs


    ys = r2c(results.ys)
    results = OdeResult(t=t_list, y=Array(ys, backend="jax", dtype=complex))

    return trim_t_results(results, t_span, t_eval)

def real_rhs(rhs):
    """Convert complex RHS to real RHS function"""

    def _real_rhs(t, y):
        return c2r(rhs(t, r2c(y)))

    return _real_rhs


def c2r(arr):
    """Convert complex array to a real array"""
    return jnp.concatenate([jnp.real(Array(arr).data), jnp.imag(Array(arr).data)])


def r2c(arr):
    """Convert a real array to a complex array"""
    size = arr.shape[0] // 2
    return arr[:size] + 1j * arr[size:]
