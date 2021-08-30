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

r"""
=============================================================
Differential equations solvers (:mod:`qiskit_dynamics.solve`)
=============================================================

This module provides high level functions for solving classes of
Differential Equations (DEs), described below.

1. Ordinary Differential Equations (ODEs)
#########################################

The most general class of DEs we consider are ODEs, which are of the form:

.. math::

    \dot{y}(t) = f(t, y(t)),

Where :math:`f` is called the Right-Hand Side (RHS) function.
ODEs can be solved by calling the :meth:`~qiskit_dynamics.solve_ode` function.

2. Linear Matrix Differential Equations (LMDEs)
###############################################

LMDEs are a specialized subclass of ODEs of importance in quantum theory. Most generally,
an LMDE is an ODE for which the the RHS function :math:`f(t, y)` is *linear* in the second
argument. In this package we work with a *standard form* of LMDEs, in which we assume:

.. math::

    f(t, y) = G(t)y,

where :math:`G(t)` is a square matrix-valued function called the *generator*, and
the state :math:`y(t)` must be an array of appropriate shape. Note that any LMDE in the more
general sense (not in *standard form*) can be restructured into one of standard form via suitable
vectorization.

The function :meth:`~qiskit_dynamics.de.solve_lmde` solves LMDEs in standard form, specified
in terms of a representation of the generator :math:`G(t)`, either as a ``Callable`` function
or a subclass of :class:`~qiskit_dynamics.models.generator_models.BaseGeneratorModel`.

.. currentmodule:: qiskit_dynamics.solve

.. autosummary::
   :toctree: ../stubs/

   solve_ode
   solve_lmde
"""

from typing import Optional, Union, Callable, Tuple, List
import inspect

from scipy.integrate import OdeSolver

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator

from qiskit import QiskitError
from qiskit_dynamics import dispatch
from qiskit_dynamics.dispatch import Array, requires_backend

from .solvers.fixed_step_solvers import scipy_expm_solver, jax_expm_solver
from .solvers.scipy_solve_ivp import scipy_solve_ivp, SOLVE_IVP_METHODS
from .solvers.jax_odeint import jax_odeint

from .models.rotating_frame import RotatingFrame
from .models.rotating_wave_approximation import rotating_wave_approximation
from .models.generator_models import BaseGeneratorModel, GeneratorModel
from .models import HamiltonianModel, LindbladModel

try:
    from jax.lax import scan
except ImportError:
    pass


def solve_ode(
    rhs: Callable,
    t_span: Array,
    y0: Array,
    method: Optional[Union[str, OdeSolver]] = "DOP853",
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    **kwargs,
):
    r"""General interface for solving Ordinary Differential Equations (ODEs).
    ODEs are differential equations of the form

    .. math::

        \dot{y}(t) = f(t, y(t)),

    where :math:`f` is a callable function and the state :math:`y(t)` is an
    arbitrarily-shaped complex :class:`Array`.

    The ``method`` argument exposes a variety of underlying ODE solvers. Optional
    arguments for any of the solver routines can be passed via ``kwargs``.
    Available methods are:

    - ``scipy.integrate.solve_ivp`` - supports methods
      ``['RK45', 'RK23', 'BDF', 'DOP853', 'Radau', 'LSODA']`` or by passing a valid
      ``scipy`` :class:`OdeSolver` instance.
    - ``jax.experimental.ode.odeint`` - accessed via passing
      ``method='jax_odeint'``.

    Results are returned as a :class:`OdeResult` object.

    Args:
        rhs: RHS function :math:`f(t, y)`.
        t_span: ``Tuple`` or ``list`` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
        t_eval: Times at which to return the solution. Must lie within ``t_span``. If unspecified,
                the solution will be returned at the points in ``t_span``.
        kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If specified method does not exist.
    """
    t_span = Array(t_span)
    y0 = Array(y0)

    rhs = dispatch.wrap(rhs)

    # solve the problem using specified method
    results = None
    if method in SOLVE_IVP_METHODS or (inspect.isclass(method) and issubclass(method, OdeSolver)):
        results = scipy_solve_ivp(rhs, t_span, y0, method, t_eval, **kwargs)
    elif isinstance(method, str) and method == "jax_odeint":
        results = jax_odeint(rhs, t_span, y0, t_eval, **kwargs)
    else:
        raise QiskitError("""Specified method is not a supported ODE method.""")

    return results


def solve_lmde(
    generator: Union[Callable, BaseGeneratorModel],
    t_span: Array,
    y0: Array,
    method: Optional[Union[str, OdeSolver]] = "DOP853",
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    **kwargs,
):
    r"""General interface for solving Linear Matrix Differential Equations (LMDEs).
    Most generally, LMDEs are a special subclass of ODEs for which the RHS function
    :math:`f(t, y)` is linear in :math:`y` for all :math:`t`, however here we
    restrict the definition to a standard form:

    .. math::

        \dot{y}(t) = G(t)y(t),

    where :math:`G(t)` is a square matrix valued-function called the *generator*,
    and :math:`y(t)` is an :class:`Array` of appropriate shape. Any LMDE in the more
    general sense can be rewritten in the above form using a suitable vectorization,
    and so no generality is lost.

    The generator :math:`G(t)` may either be specified as a `Callable` function,
    or as an instance of a :class:`BaseGeneratorModel` subclass, which defines an
    interface for accessing standard transformations on LMDEs.

    The ``method`` argument exposes solvers specialized to both LMDEs, as
    well as general ODE solvers. If the method is not specific to LMDEs,
    the problem will be passed to :meth:`solve_ode`.
    Optional arguments for any of the solver routines can be passed via ``kwargs``.
    Available LMDE-specific methods are:

    - ``'scipy_expm'``: A matrix-exponential solver using ``scipy.linalg.expm``.
                        Requires additional kwarg ``max_dt``.
    - ``'jax_expm'``: A ``jax``-based exponential solver. Requires additional kwarg ``max_dt``.

    Results are returned as a :class:`OdeResult` object.

    Args:
        generator: Representaiton of generator function :math:`G(t)`.
        t_span: ``Tuple`` or `list` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
        t_eval: Times at which to return the solution. Must lie within ``t_span``. If unspecified,
                the solution will be returned at the points in ``t_span``.
        kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If specified method does not exist, or if dimension of y0 is incompatible
                     with generator dimension.
    """

    # raise error that lmde-specific methods can't be used with LindbladModel unless
    # it is vectorized
    if (
        isinstance(generator, LindbladModel)
        and ("vectorized" not in generator.evaluation_mode)
        and (method in ["scipy_expm", "jax_expm"])
    ):
        raise QiskitError(
            """LMDE-specific methods with LindbladModel requires setting a
               vectorized evaluation mode."""
        )

    t_span = Array(t_span)
    y0 = Array(y0)

    # setup generator and rhs functions
    if (
        isinstance(generator, BaseGeneratorModel)
        and generator.rotating_frame.frame_operator is not None
    ):
        # for case of BaseGeneratorModels, setup to solve in frame basis
        if isinstance(generator, LindbladModel) and "vectorized" in generator.evaluation_mode:
            if generator.rotating_frame.frame_basis is not None:
                y0 = generator.rotating_frame.vectorized_frame_basis_adjoint @ y0
        elif isinstance(generator, LindbladModel):
            y0 = generator.rotating_frame.operator_into_frame_basis(y0)
        elif isinstance(generator, GeneratorModel):
            y0 = generator.rotating_frame.state_into_frame_basis(y0)

        # define rhs functions in frame basis
        def solver_generator(t):
            return generator(t, in_frame_basis=True)

        def solver_rhs(t, y):
            return generator(t, y, in_frame_basis=True)

    else:
        # if generator is not a BaseGeneratorModel, treat it purely as a function
        solver_generator = generator

        def solver_rhs(t, y):
            return generator(t) @ y

    if method == "scipy_expm":
        results = scipy_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    elif method == "jax_expm":
        results = jax_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    else:
        # method is not LMDE-specific, so pass to solve_ode using rhs
        results = solve_ode(solver_rhs, t_span, y0, method=method, t_eval=t_eval, **kwargs)

    # convert results to correct basis if necessary
    if (
        isinstance(generator, BaseGeneratorModel)
        and generator.rotating_frame.frame_operator is not None
    ):
        # for left multiplication cases, if number of input dimensions is 1
        # vectorized basis transformation requires transposing before and after
        if y0.ndim == 1:
            results.y = results.y.T

        if isinstance(generator, LindbladModel) and "vectorized" in generator.evaluation_mode:
            if generator.rotating_frame.frame_basis is not None:
                results.y = generator.rotating_frame.vectorized_frame_basis @ results.y
        elif isinstance(generator, LindbladModel):
            results.y = generator.rotating_frame.operator_out_of_frame_basis(results.y)
        elif isinstance(generator, GeneratorModel):
            results.y = generator.rotating_frame.state_out_of_frame_basis(results.y)

        if y0.ndim == 1:
            results.y = results.y.T

    return results
