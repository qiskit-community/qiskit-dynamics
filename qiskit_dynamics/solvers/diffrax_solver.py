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
Wrapper for diffrax solvers
"""

from typing import Callable, Optional, Union, Tuple, List
from scipy.integrate._ivp.ivp import OdeResult
from qiskit import QiskitError

from qiskit_dynamics.dispatch import requires_backend
from qiskit_dynamics.array import Array, wrap

from .solver_utils import merge_t_args


try:
    from diffrax import ODETerm, SaveAt
    from diffrax import diffeqsolve as _diffeqsolve

    from diffrax.solver import AbstractSolver
    import jax.numpy as jnp
except ImportError as err:
    pass


@requires_backend("jax")
def diffrax_solver(
    rhs: Callable,
    t_span: Array,
    y0: Array,
    method: "AbstractSolver",
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    **kwargs,
):
    """Routine for calling ``diffrax.diffeqsolve``

    Args:
        rhs: Callable of the form :math:`f(t, y)`.
        t_span: Interval to solve over.
        y0: Initial state.
        method: Which diffeq solving method to use.
        t_eval: Optional list of time points at which to return the solution.
        **kwargs: Optional arguments to be passed to ``diffeqsolve``.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: Passing both `SaveAt` argument and `t_eval` argument.
    """
    if isinstance(method, type) and issubclass(method, AbstractSolver):
        solver = method()
    else:
        solver = method

    t_list = merge_t_args(t_span, t_eval)

    # convert rhs and y0 to real
    rhs = real_rhs(rhs)
    y0 = c2r(y0)

    term = ODETerm(lambda t, y, _: Array(rhs(t.real, y), dtype=float).data)

    diffeqsolve = wrap(_diffeqsolve)

    if "saveat" in kwargs and t_eval is not None:
        raise QiskitError(
            """Saveat argument and t_eval both passed to diffrax solver.
            Use only one, t_eval can be contained in the saveat instance."""
        )

    if t_eval is not None:
        saveat = SaveAt(ts=t_eval)
        results = diffeqsolve(
            term,
            solver=solver,
            t0=t_list[0],
            t1=t_list[-1],
            dt0=None,
            y0=Array(y0, dtype=float),
            saveat=saveat,
            **kwargs,
        )
    else:
        results = diffeqsolve(
            term,
            solver=solver,
            t0=t_list[0],
            t1=t_list[-1],
            dt0=None,
            y0=Array(y0, dtype=float),
            **kwargs,
        )

    ys = jnp.array([r2c(y) for y in results.ys])
    results = OdeResult(t=t_eval, y=Array(ys, backend="jax", dtype=complex))
    return results


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
