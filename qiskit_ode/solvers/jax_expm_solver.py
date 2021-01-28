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
Custom jax expm-based solver.
"""

from typing import Callable, Optional, Union, Tuple, List
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit_ode.dispatch import requires_backend, Array

from .solver_utils import merge_t_args, trim_t_results

try:
    import jax.numpy as jnp
    from jax.lax import scan, cond
    from jax.scipy.linalg import expm as jexpm
except ImportError:
    pass


@requires_backend('jax')
def jax_expm_solver(generator: Callable,
                    t_span: Array,
                    y0: Array,
                    max_dt: float,
                    t_eval: Optional[Union[Tuple, List, Array]] = None):
    """Fixed-step size matrix exponential based solver implemented with ``jax``.
    Solves the specified problem by taking steps of size no larger than ``max_dt``.

    Args:
        generator: Callable of the form :math:`G(t)`
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    def gen(t):
        return Array(generator(t)).data

    y0 = Array(y0, backend='jax').data

    # time args are non-differentiable
    t_span = Array(t_span, backend='numpy').data
    max_dt = Array(max_dt, backend='numpy').data
    t_list = np.array(merge_t_args(t_span, t_eval))

    delta_t_list = np.diff(t_list)
    n_steps_list = np.abs(delta_t_list / max_dt).astype(int)

    # correct potential rounding errors
    # pylint: disable=consider-using-enumerate
    for k in range(len(n_steps_list)):
        if n_steps_list[k] == 0:
            n_steps_list[k] = 1
        # absolute value to handle backwards integration
        if np.abs(delta_t_list[k] / n_steps_list[k]) > max_dt:
            n_steps_list[k] = n_steps_list[k] + 1

    max_steps = n_steps_list.max()
    h_list = np.array(delta_t_list / n_steps_list)

    def scan_f(carry, x):
        current_t, h, n_steps = x
        current_y = carry
        next_y = step_integrate(gen, current_y, current_t, h, n_steps, max_steps)
        return next_y, next_y

    ys = scan(scan_f, init=y0, xs=(jnp.array(t_list[:-1]),
                                   jnp.array(h_list),
                                   jnp.array(n_steps_list)))[1]

    ys = jnp.append(jnp.expand_dims(y0, axis=0), ys, axis=0)

    results = OdeResult(t=t_list, y=Array(ys, backend='jax'))

    return trim_t_results(results, t_span, t_eval)


def step_integrate(generator: Callable,
                   y0: Array,
                   t0: float,
                   h: float,
                   n_steps: int,
                   max_steps: int):
    """Integrate starting at ``t0`` for ``n_steps`` of length ``h``.

    Args:
        generator: Callable representing the generator.
        y0: State at ``t0``.
        t0: Initial time.
        h: Step size.
        n_steps: Number of steps to take.
        max_steps: Upper bound on ``n_steps``.

    Returns:
        Array: State after the ``n_steps``.
    """

    def take_step(step, y):
        eval_time = t0 + (h * step) + h / 2
        return jexpm(generator(eval_time) * h) @ y

    def identity(y):
        return y

    def scan_func(carry, step):
        y = cond(step < n_steps,
                 lambda y: take_step(step, y),
                 identity,
                 carry)
        return (y, None)

    return scan(scan_func, y0, jnp.arange(max_steps))[0]
