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

from typing import Callable
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError
from qiskit_ode.dispatch import requires_backend, Array


@requires_backend('jax')
def solve_jax_expm(generator: Callable, t_span: Array, y0: Array, **kwargs):
    """Fixed-step size matrix exponential based solver implemented with ``jax``.
    This routine splits the interval ``t_span`` into equally sized steps of size
    no larger than ``max_dt``, and solves the ODE by exponentiating the generator
    at every step.

    Args:
        generator: Callable of the form :math:`G(t)`
        t_span: Interval to solve over.
        y0: Initial state.
        kwargs: Must contain ``max_dt``.

    Returns:
        OdeResult: results object

    Raises:
        QiskitError: if required kwarg ``max_dt`` is not present
    """

    if 'max_dt' not in kwargs:
        raise QiskitError('jax_expm solver requires specification of max_dt.')

    from jax.scipy.linalg import expm as jexpm
    from jax.lax import scan
    import jax.numpy as jnp

    def gen(t):
        return Array(generator(t)).data

    t_span = Array(t_span, backend='numpy').data
    y0 = Array(y0, backend='jax').data
    max_dt = Array(kwargs.get('max_dt'), backend='numpy').data

    delta_t = t_span[1] - t_span[0]

    # Note: casting this specifically as a standard numpy array is required
    # for forcing this value to be treated as static during function
    # transformations
    steps = np.array(delta_t // max_dt).astype(int)
    h = delta_t / steps

    def scan_func(carry, step):
        eval_time = t_span[0] + (h * step) + h/2
        return (jexpm(gen(eval_time) * h) @ carry, None)

    yf = scan(scan_func, y0, jnp.arange(steps))[0]

    return OdeResult(t=Array(t_span), y=[y0, yf])
