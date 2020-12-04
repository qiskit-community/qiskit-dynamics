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

"""
Module for custom solvers.
"""

from typing import Callable
import numpy as np

from qiskit_ode.dispatch import Array

def jax_expm(generator: Callable,
             t_span: Array,
             y0: Array,
             max_dt: float):
    """Fixed-step size matrix exponential based solver implemented with `jax`.
    This routine splits the interval `t_span` into equally sized steps of size
    no larger than `max_dt`, and solves the ODE by exponentiating the generator
    at every step.

    Args:
        generator: Callable function for the generator.
        t_span: Interval to solve over.
        y0: Initial state.
        max_dt: Upper bound on step size.

    Returns:
        Array: The final state.
    """

    try:
        from jax.scipy.linalg import expm as jexpm
        from jax.lax import scan
        import jax.numpy as jnp
    except ImportError as e:
        raise e

    delta_t = t_span[1] - t_span[0]

    # Note: casting this specifically as a standard numpy array is required
    # for forcing this value to be treated as static during function
    # transformations
    steps = np.array(delta_t // max_dt).astype(int)
    h = delta_t / steps

    def scan_func(carry, step):
        eval_time = t_span[0] + (h * step) + h/2
        return (jexpm(generator(eval_time) * h) @ carry, None)

    yf = scan(scan_func, y0, jnp.arange(steps))[0]
    return yf
