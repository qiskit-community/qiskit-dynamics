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

from typing import Callable
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit_ode.dispatch import requires_backend, Array


@requires_backend('jax')
def jax_odeint(rhs: Callable, t_span: Array, y0: Array, **kwargs):
    """Routine for calling `jax.experimental.ode.odeint`

    Args:
        rhs: Callable of the form :math:`f(t, y)`
        t_span: Interval to solve over.
        y0: Initial state.
        kwargs: Optional arguments for `odeint`.

    Returns:
        OdeResult: results object
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

    return OdeResult(t=times, y=Array(results))
