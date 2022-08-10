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
Perturbative solver base class.
"""

from abc import ABC, abstractmethod
from typing import Callable, List, Union

import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit_dynamics import Signal
from qiskit_dynamics.signals import SignalList
from qiskit_dynamics.array import Array
from qiskit_dynamics.dispatch.dispatch import Dispatch

from .expansion_model import ExpansionModel
from ..solver_utils import setup_args_lists

try:
    import jax.numpy as jnp
    from jax import vmap
    from jax.lax import associative_scan
except ImportError:
    pass


class _PerturbativeSolver(ABC):
    """Abstract base class for perturbation-based solvers."""

    @abstractmethod
    def __init__(self, model: ExpansionModel):
        r"""Initialize.

        Args:
            model: ExpansionModel defining solver behaviour.
        """
        self._model = model

    @property
    def model(self) -> ExpansionModel:
        """Model object storing expansion details."""
        return self._model

    def solve(
        self,
        t0: Union[float, List[float]],
        n_steps: Union[int, List[int]],
        y0: Union[Array, List[Array]],
        signals: Union[List[Signal], List[List[Signal]]],
    ) -> Union[OdeResult, List[OdeResult]]:
        """Solve given an initial time, number of steps, signals, and initial state.

        Note that this method can be used to solve a list of simulations at once, by specifying
        one or more of the arguments ``t0``, ``n_steps``, ``y0``, or ``signals`` as a list of
        valid inputs. For this mode of operation, all of these arguments must be either lists of
        the same length, or a single valid input, which will be used repeatedly.

        Args:
            t0: Initial time.
            n_steps: Number of time steps to solve for.
            y0: Initial state at time t0.
            signals: List of signals.

        Returns:
            OdeResult: Results object, or list of results objects.

        Raises:
            QiskitError: If improperly formatted arguments.
        """

        # validate and setup list of simulations
        [t0_list, n_steps_list, y0_list, signals_list], multiple_sims = setup_args_lists(
            args_list=[t0, n_steps, y0, signals],
            args_names=["t0", "n_steps", "y0", "signals"],
            args_to_list=[
                lambda x: _scalar_to_list(x, "t0"),
                lambda x: _scalar_to_list(x, "n_steps"),
                _y0_to_list,
                _signals_to_list,
            ],
        )

        all_results = []
        for args in zip(t0_list, n_steps_list, y0_list, signals_list):
            if len(args[-1]) != len(self.model.operators):
                raise QiskitError("Signals must be the same length as the operators in the model.")
            all_results.append(
                self._solve(t0=args[0], n_steps=args[1], y0=args[2], signals=args[3])
            )

        if multiple_sims is False:
            return all_results[0]

        return all_results

    @abstractmethod
    def _solve(self, t0: float, n_steps: int, y0: Array, signals: List[Signal]) -> OdeResult:
        """Solve once for a list of signals, initial state, initial time, and number of steps.

        Args:
            t0: Initial time.
            n_steps: Number of time steps to solve for.
            y0: Initial state at time t0.
            signals: List of signals.

        Returns:
            OdeResult: Results object.
        """
        pass


def _perturbative_solve(
    single_step: Callable,
    model: "ExpansionModel",
    signals: List[Signal],
    y0: np.ndarray,
    t0: float,
    n_steps: int,
) -> np.ndarray:
    """Standard python logic version of perturbative solving routine."""
    U0 = model.rotating_frame.state_out_of_frame(t0, np.eye(model.Udt.shape[0], dtype=complex))
    Uf = model.rotating_frame.state_into_frame(
        t0 + n_steps * model.dt, np.eye(U0.shape[0], dtype=complex)
    )

    sig_cheb_coeffs = model.approximate_signals(signals, t0, n_steps)

    y = U0 @ y0
    for k in range(n_steps):
        y = single_step(sig_cheb_coeffs[:, k], y)

    return Uf @ y


def _perturbative_solve_jax(
    single_step: Callable,
    model: "ExpansionModel",
    signals: List[Signal],
    y0: np.ndarray,
    t0: float,
    n_steps: int,
) -> np.ndarray:
    """JAX version of _perturbative_solve."""
    U0 = model.rotating_frame.state_out_of_frame(
        t0, jnp.eye(model.Udt.shape[0], dtype=complex)
    ).data
    Uf = model.rotating_frame.state_into_frame(
        t0 + n_steps * model.dt, jnp.eye(U0.shape[0], dtype=complex)
    ).data

    sig_cheb_coeffs = model.approximate_signals(signals, t0, n_steps).data

    y = U0 @ y0

    step_propagators = vmap(single_step)(jnp.flip(sig_cheb_coeffs.transpose(), axis=0))
    y = associative_scan(jnp.matmul, step_propagators, axis=0)[-1] @ y

    return Uf @ y


def _scalar_to_list(x, name):
    """Check if x is a scalar or a list of scalars, and convert to a list in either case."""
    was_list = False
    x_ndim = _nested_ndim(x)
    if x_ndim > 1:
        raise QiskitError(f"{name} must be either 0d or 1d.")

    if x_ndim == 1:
        was_list = True
    else:
        x = [x]

    return x, was_list


def _y0_to_list(y0):
    """Check if y0 is a single array or list of arrays, and return as a list in either case."""
    was_list = False
    if not isinstance(y0, list):
        y0 = [y0]
    else:
        was_list = True

    return y0, was_list


def _signals_to_list(signals):
    """Check if signals is a single signal specification or a list of
    such specifications, and return as a list in either case.
    """
    was_list = False
    if signals is None:
        signals = [signals]
    elif isinstance(signals, list) and isinstance(signals[0], (list, SignalList)):
        # multiple lists
        was_list = True
    elif isinstance(signals, SignalList) or (
        isinstance(signals, list) and not isinstance(signals[0], (list, SignalList))
    ):
        # single signals list
        signals = [signals]
    else:
        raise QiskitError("Signals specified in invalid format.")

    return signals, was_list


def _nested_ndim(x):
    """Determine the 'ndim' of x, which could be composed of nested lists and array types."""
    if isinstance(x, (list, tuple)):
        return 1 + _nested_ndim(x[0])
    elif issubclass(type(x), Dispatch.REGISTERED_TYPES) or isinstance(x, Array):
        return x.ndim

    # assume scalar
    return 0
