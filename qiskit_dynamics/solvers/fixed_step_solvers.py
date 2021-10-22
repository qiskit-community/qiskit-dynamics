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
Custom fixed step solvers.
"""

from typing import Callable, Optional, Union, Tuple, List
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult
from scipy.linalg import expm

from qiskit_dynamics.dispatch import requires_backend, Array

try:
    from jax import vmap
    import jax.numpy as jnp
    from jax.lax import scan, cond, associative_scan
    from jax.scipy.linalg import expm as jexpm
except ImportError:
    pass

from .solver_utils import merge_t_args, trim_t_results


def scipy_expm_solver(
    generator: Callable,
    t_span: Array,
    y0: Array,
    max_dt: float,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
):
    """Fixed-step size matrix exponential based solver implemented with
    ``scipy.linalg.expm``. Solves the specified problem by taking steps of
    size no larger than ``max_dt``.

    Args:
        generator: Callable, either a generator rhs
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    def take_step(generator, t0, y, h):
        eval_time = t0 + (h / 2)
        return expm(generator(eval_time) * h) @ y

    return fixed_step_solver_template(
        take_step, rhs_func=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


@requires_backend("jax")
def jax_expm_solver(
    generator: Callable,
    t_span: Array,
    y0: Array,
    max_dt: float,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
):
    """Fixed-step size matrix exponential based solver implemented with ``jax``.
    Solves the specified problem by taking steps of size no larger than ``max_dt``.

    Args:
        generator: Callable, either a generator rhs
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    def take_step(generator, t, y, h):
        eval_time = t + (h / 2)
        return jexpm(generator(eval_time) * h) @ y

    return fixed_step_solver_template_jax(
        take_step, rhs_func=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )

@requires_backend("jax")
def jax_expm_parallel_solver(
    generator: Callable,
    t_span: Array,
    y0: Array,
    max_dt: float,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
):
    """Parallel version of jax_expm_solver
    """

    def take_step(generator, t, y, h):
        eval_time = t + (h / 2)
        return jexpm(generator(eval_time) * h) @ y

    return fixed_step_lmde_solver_parallel_template_jax(
        take_step, generator=generator, t_span=t_span, y0=y0, max_dt=max_dt, t_eval=t_eval
    )


def fixed_step_solver_template(
    take_step: Callable,
    rhs_func: Callable,
    t_span: Array,
    y0: Array,
    max_dt: float,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
):
    """Helper function for implementing fixed-step solvers supporting both
    ``t_span`` and ``max_dt`` arguments. ``take_step`` is assumed to be a
    function implementing a single step of size h of a fixed-step method.
    The signature of ``take_step`` is assumed to be:
        - rhs_func: Either a generator :math:`G(t)` or RHS function :math:`f(t,y)`.
        - t0: The current time.
        - y0: The current state.
        - h: The size of the step to take.

    It returns:
        - y: The state of the DE at time t0 + h.

    ``take_step`` is used to integrate the DE specified by ``rhs_func``
    through all points in ``t_eval``, taking steps no larger than ``max_dt``.
    Each interval in ``t_eval`` is divided into the least number of sub-intervals
    of equal length so that the sub-intervals are smaller than ``max_dt``.

    Args:
        take_step: Callable for fixed step integration.
        rhs_func: Callable, either a generator or rhs function.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    # ensure the output of rhs_func is a raw array
    def wrapped_rhs_func(*args):
        return Array(rhs_func(*args)).data

    y0 = Array(y0).data

    t_list, h_list, n_steps_list = get_fixed_step_sizes(t_span, t_eval, max_dt)

    ys = [y0]
    for current_t, h, n_steps in zip(t_list, h_list, n_steps_list):
        y = ys[-1]
        inner_t = current_t
        for _ in range(n_steps):
            y = take_step(wrapped_rhs_func, inner_t, y, h)
            inner_t = inner_t + h
        ys.append(y)
    ys = Array(ys)

    results = OdeResult(t=t_list, y=ys)

    return trim_t_results(results, t_span, t_eval)


def fixed_step_solver_template_jax(
    take_step: Callable,
    rhs_func: Callable,
    t_span: Array,
    y0: Array,
    max_dt: float,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
):
    """This function is the jax control-flow version of
    :meth:`fixed_step_solver_template`. See the documentation of :meth:`fixed_step_solver_template`
    for details.

    Args:
        take_step: Callable for fixed step integration.
        rhs_func: Callable, either a generator or rhs function.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    # ensure the output of rhs_func is a raw array
    def wrapped_rhs_func(*args):
        return Array(rhs_func(*args), backend="jax").data

    y0 = Array(y0).data

    t_list, h_list, n_steps_list = get_fixed_step_sizes(t_span, t_eval, max_dt)

    # if jax, need bound on number of iterations in each interval
    max_steps = n_steps_list.max()

    def identity(y):
        return y

    # interval integrator set up for jax.lax.scan
    def scan_interval_integrate(carry, x):
        current_t, h, n_steps = x
        current_y = carry

        def scan_take_step(carry, step):
            t, y = carry
            y = cond(step < n_steps, lambda y: take_step(wrapped_rhs_func, t, y, h), identity, y)
            t = t + h
            return (t, y), None

        next_y = scan(scan_take_step, (current_t, current_y), jnp.arange(max_steps))[0][1]

        return next_y, next_y

    ys = scan(
        scan_interval_integrate,
        init=y0,
        xs=(jnp.array(t_list[:-1]), jnp.array(h_list), jnp.array(n_steps_list)),
    )[1]

    ys = Array(jnp.append(jnp.expand_dims(y0, axis=0), ys, axis=0), backend="jax")

    results = OdeResult(t=t_list, y=ys)

    return trim_t_results(results, t_span, t_eval)


def fixed_step_lmde_solver_parallel_template_jax(
    take_step: Callable,
    generator: Callable,
    t_span: Array,
    y0: Array,
    max_dt: float,
    t_eval: Optional[Union[Tuple, List, Array]] = None,
):
    """Parallelized and LMDE specific version of fixed_step_solver_template_jax.

    Assuming the structure of an LMDE:
    * Computes all propagators over each individual time-step in parallel using ``jax.vmap``.
    * Computes all propagators from t_span[0] to each intermediate time point in parallel
      using ``jax.lax.associative_scan``.
    * Applies results to y0 and extracts the desired time points from ``t_eval``.

    The above logic is slightly varied to save some operations is ``y0`` is square.

    Args:
        take_step: Fixed step integration rule.
        generator: Generator for the LMDE.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Maximum step size.
        t_eval: Optional list of time points at which to return the solution.

    Returns:
        OdeResult: Results object.
    """

    # ensure the output of rhs_func is a raw array
    def wrapped_rhs_func(*args):
        return Array(rhs_func(*args), backend="jax").data

    t_list, h_list, n_steps_list = get_fixed_step_sizes(t_span, t_eval, max_dt)

    # if jax, need bound on number of iterations in each interval
    max_steps = n_steps_list.max()

    # set things up to compute propagators in reverse order for the purposes of scanning
    all_times_reverse = [] # all stepping points given in reverse order
    all_h_reverse = [] # step sizes for each point above
    t_list_locations = [-1] # ordered list of locations in all_times_reverse that are in t_list
    for t, h, n_steps in zip(t_list, h_list, n_steps_list):
        all_times_reverse = np.append(t + h * np.flip(np.arange(n_steps)), all_times_reverse)
        all_h_reverse = np.append(h * np.ones(n_steps), all_h_reverse)
        t_list_locations = np.append(t_list_locations, [t_list_locations[-1] - n_steps])

    # compute propagators over each time step in parallel
    id = jnp.eye(y0.shape[-1], dtype=complex)
    step_propagators = vmap(lambda t, h: take_step(generator, t, id, h))(all_times_reverse, all_h_reverse)

    # multiply propagators together in parallel
    ys = None
    reverse_mul = lambda A, B: jnp.matmul(B, A)
    if y0.ndim == 2 and y0.shape[0] == y0.shape[1]:
        # if square, append y0 as the first step propagator, scan, and extract
        intermediate_props = associative_scan(reverse_mul, jnp.append(step_propagators, jnp.array([y0]), axis=0), reverse=True, axis=0)
        ys = Array(intermediate_props[t_list_locations], backend='jax')
    else:
        # if not square, scan propagators, extract relevant time points, multiply by y0,
        # then append y0
        intermediate_props = associative_scan(reverse_mul, step_propagators, reverse=True, axis=0)
        intermediate_y = intermediate_props[t_list_locations[1:]] @ y0
        ys = Array(jnp.append(jnp.array([y0]), intermediate_y, axis=0), backend='jax')

    results = OdeResult(t=t_list, y=ys)

    return trim_t_results(results, t_span, t_eval)


def get_fixed_step_sizes(t_span: Array, t_eval: Array, max_dt: float) -> Tuple[Array, Array, Array]:
    """Merge ``t_span`` and ``t_eval``, and determine the number of time steps and
    and step sizes (no larger than ``max_dt``) required to fixed-step integrate between
    each time point.

    Args:
        t_span: Total interval of integration.
        t_eval: Time points within t_span at which the solution should be returned.
        max_dt: Max size step to take.

    Returns:
        Tuple[Array, Array, Array]: with merged time point list, list of step sizes to take
        between time points, and list of corresponding number of steps to take between time steps.
    """

    # time args are non-differentiable
    t_span = Array(t_span, backend="numpy").data
    max_dt = Array(max_dt, backend="numpy").data
    t_list = np.array(merge_t_args(t_span, t_eval))

    # set the number of time steps required in each interval so that
    # no steps larger than max_dt are taken
    delta_t_list = np.diff(t_list)
    n_steps_list = np.abs(delta_t_list / max_dt).astype(int)

    # correct potential rounding errors
    for idx, (delta_t, n_steps)  in enumerate(zip(delta_t_list, n_steps_list)):
        if n_steps == 0:
            n_steps = 1
        # absolute value to handle backwards integration
        if np.abs(delta_t / n_steps) / max_dt > 1 + 1e-15:
            n_steps_list[idx] = n_steps + 1

    # step size in each interval
    h_list = np.array(delta_t_list / n_steps_list)

    return t_list, h_list, n_steps_list
