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
# pylint: disable=invalid-name,no-member,attribute-defined-outside-init

"""
Module for solving interfaces.
"""

from typing import Optional, Union, Callable
import inspect

import numpy as np
from scipy.integrate import solve_ivp, OdeSolver

from qiskit import QiskitError
from qiskit_ode.dispatch import Array, requires_backend
from .de_solvers import jax_expm
from .de_problems import ODEProblem, LMDEProblem
from ..type_utils import StateTypeConverter

# supported scipy methods
SOLVE_IVP_METHODS = ['RK45', 'RK23', 'BDF', 'DOP853']


def solve(problem: Union[ODEProblem, Callable],
          t_span: Array,
          y0: Array,
          method: Optional[Union[str, OdeSolver]] = 'DOP853',
          **kwargs):
    """General solver interface for :class:`ODEProblem` instances.

    This method exposes a variety of underlying ODE solvers, which can be
    accessed via the `method` argument. Optional arguments for any of the
    solver routines can be passed via `kwargs`.

        - `scipy.integrate.solve_ivp` - supports methods
          `['RK45', 'RK23', 'BDF', 'DOP853']` or by passing a valid
          `scipy` :class:`OdeSolver` instance.
        - `jax.experimental.ode.odeint` - accessed via passing
          `method='jax_odeint'`.
        - 'jax_expm' - a `jax`-based exponential solver. Requires argument
          `max_dt` passed in `kwargs`.

    Results are returned as a :class:`SolveResult` object.

    Args:
        problem: ODE specification or rhs function
        t_span: tuple or list of initial and final time
        y0: state at initial time
        method: solving method to use
        kwargs: additional arguments to pass to the solver

    Returns:
        SolveResult: results object

    Raises:
        QiskitError: if specified method does not exist
    """

    t_span = Array(t_span)
    y0 = Array(y0)

    # if a callable put into an ODEProblem
    if not isinstance(problem, ODEProblem):
        problem = ODEProblem(problem)

    # convert initial state into internal problem format
    y0 = problem.user_state_to_problem(t_span[0], y0)

    # solve the problem using specified method
    results = None
    if (method in SOLVE_IVP_METHODS or (inspect.isclass(method) and
                                        issubclass(method, OdeSolver))):
        results = _call_scipy_solve_ivp(problem,
                                        t_span,
                                        y0,
                                        method,
                                        **kwargs)

    elif isinstance(method, str) and method == 'jax_odeint':
        results = _call_jax_odeint(problem,
                                   t_span,
                                   y0,
                                   **kwargs)

    elif isinstance(method, str) and method == 'jax_expm':
        if not isinstance(problem, LMDEProblem):
            raise QiskitError('jax_expm requires a generator.')

        results = _call_jax_expm(problem.generator,
                                 t_span,
                                 y0,
                                 **kwargs)

    else:
        raise QiskitError("""Specified method does not exist or is
                             not supported.""")

    # convert resulting states from problem format into user format
    output_states = []
    for idx in range(len(results.y)):
        output_states.append(problem.problem_state_to_user(results.t[idx],
                                                           results.y[idx]).data)
    results.y = Array(output_states)

    # return the results
    return results


def _call_scipy_solve_ivp(rhs: Callable,
                          t_span: Array,
                          y0: Array,
                          method: Union[str, OdeSolver],
                          **kwargs):
    """Routine for calling `scipy.integrate.solve_ivp`

    Args:
        rhs: Callable of the form :math:`f(t, y)`
        t_span: Interval to solve over.
        y0: Initial state.
        method: solver method.
        kwargs: optional arguments for `solve_ivp`.

    Returns:
        SolveResult: results object

    Raises:
        QiskitError: if unsupported kwarg present
    """

    if 'dense_output' in kwargs and kwargs['dense_output'] is True:
        raise QiskitError('dense_output not supported for solve_ivp.')

    # solve_ivp requires 1d arrays internally
    internal_state_spec = {'type': 'array', 'ndim': 1}
    type_converter = StateTypeConverter.from_outer_instance_inner_type_spec(y0,
                                                                            internal_state_spec)

    # modify the rhs to work with 1d arrays
    rhs = type_converter.rhs_outer_to_inner(rhs)

    # convert y0 to the flattened version
    y0 = type_converter.outer_to_inner(y0)

    results = solve_ivp(rhs,
                        t_span=t_span.data,
                        y0=y0.data,
                        method=method,
                        **kwargs)

    # convert to the standardized results format
    # solve_ivp returns the states as a 2d array with columns being the states
    results.y = results.y.transpose()
    results.y = Array([type_converter.inner_to_outer(y) for y in results.y])

    return SolveResult(**dict(results))


@requires_backend('jax')
def _call_jax_odeint(rhs: Callable,
                     t_span: Array,
                     y0: Array,
                     **kwargs):
    """Routine for calling `jax.experimental.ode.odeint`

    Args:
        rhs: Callable of the form :math:`f(t, y)`
        t_span: Interval to solve over.
        y0: Initial state.
        kwargs: Optional arguments for `odeint`.

    Returns:
        SolveResult: results object
    """

    from jax.experimental.ode import odeint

    times = None
    if 't_eval' in kwargs:
        t_eval = kwargs['t_eval']
        times = Array(t_eval, dtype=float).data
        if t_span[0] < t_eval[0]:
            times = np.append(Array(t_span[0], dtype=float), times)

        if t_span[-1] > t_eval[-1]:
            times = np.append(times, Array(t_span[-1], dtype=float))
    else:
        times = Array(t_span, dtype=float)

    results = odeint(lambda t, y: Array(rhs(y, t)).data,
                     y0=Array(y0).data,
                     t=Array(times).data,
                     **kwargs)

    return SolveResult(t=times, y=Array(results))


@requires_backend('jax')
def _call_jax_expm(generator: Callable, t_span: Array, y0: Array, **kwargs):
    """Routine for calling `qiskit_ode.de.de_solvers.jax_expm`

    Args:
        generator: Callable of the form :math:`G(t)`
        t_span: Interval to solve over.
        y0: Initial state.
        kwargs: Must contain `max_dt`.

    Returns:
        SolveResult: results object

    Raises:
        QiskitError: if required kwarg `max_dt` is not present
    """

    if 'max_dt' not in kwargs:
        raise QiskitError('jax_expm solver requires specification of max_dt.')

    yf = jax_expm(lambda t: Array(generator(t)).data,
                  Array(t_span, backend='numpy').data,
                  Array(y0).data,
                  Array(kwargs.get('max_dt'), backend='numpy').data)

    return SolveResult(t=Array(t_span), y=[y0, yf])


class SolveResult:
    """Object for storing results of `solve`, with attribute access.
    The particular attributes vary depending on which method is used
    in `solve`, but at a minimum, a :class`SolveResult` instance
    should contain the attributes:
        - `t`: a list of times.
        - `y`: a list of states at the corresponding times, indexed by the
               leading axis.
    """

    def __init__(self, **kwargs):
        self.__dict__ = kwargs

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string_rep = ''
        for key, value in self.__dict__.items():
            string_rep += key + ': ' + str(value) + '\n'

        return string_rep

    def keys(self):
        """Check available attributes."""
        return self.__dict__.keys()
