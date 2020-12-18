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
========================================================
Differential equations solvers (:mod:`qiskit_ode.solve`)
========================================================

This module provides high level functions for solving classes of
Differential Equations (DEs), described below.

1. Ordinary Differential Equations (ODEs)
#########################################

The most general class of DEs we consider are ODEs, which are of the form:

.. math::

    \dot{y}(t) = f(t, y(t)),

Where :math:`f` is called the Right-Hand Side (RHS) function.
ODEs can be solved by calling the :meth:`~qiskit_ode.de.solve_ode` function.

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

The function :meth:`~qiskit_ode.de.solve_lmde` solves LMDEs in standard form, specified
in terms of a representation of the generator :math:`G(t)`, either as a ``Callable`` function
or a subclass of :class:`~qiskit_ode.models.generator_models.BaseGeneratorModel`.

.. currentmodule:: qiskit_ode.solve

.. autosummary::
   :toctree: ../stubs/

   solve_ode
   solve_lmde
"""

from typing import Optional, Union, Callable, Tuple
import inspect

from scipy.integrate import OdeSolver
# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError
from qiskit_ode import dispatch
from qiskit_ode.dispatch import Array

from .solvers.solve_jax_expm import solve_jax_expm
from .solvers.scipy_solve_ivp import scipy_solve_ivp, SOLVE_IVP_METHODS
from .solvers.jax_odeint import jax_odeint

from .models.frame import Frame
from .models.generator_models import BaseGeneratorModel, CallableGenerator
from .models import HamiltonianModel


def solve_ode(rhs: Callable,
              t_span: Array,
              y0: Array,
              method: Optional[Union[str, OdeSolver]] = 'DOP853',
              **kwargs):
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
      ``['RK45', 'RK23', 'BDF', 'DOP853']`` or by passing a valid
      ``scipy`` :class:`OdeSolver` instance.
    - ``jax.experimental.ode.odeint`` - accessed via passing
      ``method='jax_odeint'``.

    Results are returned as a :class:`OdeResult` object.

    Args:
        rhs: RHS function :math:`f(t, y)`.
        t_span: ``Tuple`` or ``list`` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
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
    if (method in SOLVE_IVP_METHODS or (inspect.isclass(method) and issubclass(method, OdeSolver))):
        results = scipy_solve_ivp(rhs, t_span, y0, method, **kwargs)
    elif isinstance(method, str) and method == 'jax_odeint':
        results = jax_odeint(rhs, t_span, y0, **kwargs)
    else:
        raise QiskitError("""Specified method is not a supported ODE method.""")

    return results


def solve_lmde(generator: Union[Callable, BaseGeneratorModel],
               t_span: Array,
               y0: Array,
               method: Optional[Union[str, OdeSolver]] = 'DOP853',
               input_frame: Optional[Union[str, Array]] = 'auto',
               solver_frame: Optional[Union[str, Array]] = 'auto',
               output_frame: Optional[Union[str, Array]] = 'auto',
               solver_cutoff_freq: Optional[float] = None,
               **kwargs):
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

    - ``'jax_expm'`` - a ``jax``-based exponential solver. Requires argument
      ``max_dt`` passed in ``kwargs``.

    Results are returned as a :class:`OdeResult` object.

    Args:
        generator: Representaiton of generator function :math:`G(t)`.
        t_span: ``Tuple`` or `list` of initial and final time.
        y0: State at initial time.
        method: Solving method to use.
        input_frame: Frame that the initial state is specified in. If ``input_frame == 'auto'``,
                     defaults to using the frame the generator is specified in.
        solver_frame: Frame to solve the system in. If ``solver_frame == 'auto'``, defaults to
                      using the drift of the generator when specified as a
                      :class:`BaseGeneratorModel`.
        output_frame: Frame to return the results in. If ``output_frame == 'auto'``,
                     defaults to using the frame the generator is specified in.
        solver_cutoff_freq: Cutoff frequency to use (if any) for doing the rotating
                            wave approximation.
        kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If specified method does not exist, or if dimension of y0 is incompatible
                     with generator dimension.
    """

    # setup input frame, output frame, and the internal solver generator based on args
    input_frame, output_frame, generator = \
        setup_lmde_frames_and_generator(input_generator=generator,
                                        input_frame=input_frame,
                                        solver_frame=solver_frame,
                                        output_frame=output_frame,
                                        solver_cutoff_freq=solver_cutoff_freq)

    # store shape of y0, and reshape y0 if necessary
    return_shape = y0.shape
    y0 = lmde_y0_reshape(generator_dim=generator(t_span[0]).shape[0],
                         y0=y0)

    # map y0 from input frame into solver frame and basis
    y0 = input_frame.state_out_of_frame(t_span[0], y0)
    y0 = generator.frame.state_into_frame(t_span[0], y0, return_in_frame_basis=True)

    # define rhs functions in frame basis
    def solver_generator(t):
        return generator(t, in_frame_basis=True)

    def solver_rhs(t, y):
        return generator(t, y, in_frame_basis=True)

    # call correct method
    results = None
    if isinstance(method, str) and method == 'jax_expm':
        results = solve_jax_expm(solver_generator,
                                 t_span,
                                 y0,
                                 **kwargs)

    # if results is None, method is not LMDE-specific, so pass to solve_ode using rhs
    if results is None:
        results = solve_ode(solver_rhs, t_span, y0, method=method, **kwargs)

    # convert any states in results to correct basis/frame
    output_states = []
    for idx in range(len(results.y)):
        time = results.t[idx]
        out_y = results.y[idx]

        # transform out of solver frame/basis into output frame
        out_y = generator.frame.state_out_of_frame(time, out_y, y_in_frame_basis=True)
        out_y = output_frame.state_into_frame(time, out_y)

        # reshape to match input shape if necessary
        output_states.append(Array(out_y.reshape(return_shape, order='F')).data)

    results.y = Array(output_states)

    return results


def setup_lmde_frames_and_generator(input_generator: Union[Callable, BaseGeneratorModel],
                                    input_frame: Optional[Union[str, Array]] = 'auto',
                                    solver_frame: Optional[Union[str, Array]] = 'auto',
                                    output_frame: Optional[Union[str, Array]] = 'auto',
                                    solver_cutoff_freq: Optional[float] = None) \
                                    -> Tuple[Frame, Frame, BaseGeneratorModel]:
    """Helper function for setting up internally used :class:`BaseGeneratorModel`
    for :meth:`solve_lmde`.

    Args:
        input_generator: User-supplied generator.
        input_frame: Input frame for the problem.
        solver_frame: Frame to solve in.
        output_frame: Output frame for the problem.
        solver_cutoff_freq: Cutoff frequency to use when solving.

    Returns:
        Frame, Frame, BaseGeneratorModel: input frame, output frame, and BaseGeneratorModel
    """

    generator = None

    # if not an instance of a subclass of BaseGeneratorModel assume Callable
    if not isinstance(input_generator, BaseGeneratorModel):
        generator = CallableGenerator(input_generator)
    else:
        generator = input_generator.copy()

    # set input and output frames
    if isinstance(input_frame, str) and input_frame == 'auto':
        input_frame = generator.frame
    else:
        input_frame = Frame(input_frame)

    if isinstance(output_frame, str) and output_frame == 'auto':
        output_frame = generator.frame
    else:
        output_frame = Frame(output_frame)

    # set solver frame
    # this must be done after input/output frames as it modifies the generator itself
    if isinstance(solver_frame, str) and solver_frame == 'auto':
        # if auto, set it to the anti-hermitian part of the drift
        generator.frame = None

        if isinstance(generator, HamiltonianModel):
            generator.frame = -1j * generator.drift
        else:
            generator.frame = anti_herm_part(generator.drift)
    else:
        generator.frame = Frame(solver_frame)

    generator.cutoff_freq = solver_cutoff_freq

    return input_frame, output_frame, generator


def lmde_y0_reshape(generator_dim: int, y0: Array) -> Array:
    """Either: G(t)y0 is already well defined, or we assume that y0 is the input state of
    the more general form of lmde f(t, y) with f linear in y, and we assume the generator
    has been vectorized in column stacking convention.

    Args:
        generator_dim: dimension of the generator
        y0: input state

    Return:
        y0: Appropriately reshaped input state.

    Raises:
        QiskitError: If shape of y0 does not conform to any interpretation of the generator dim.
    """

    if y0.shape[0] != generator_dim:
        if y0.shape[0] * y0.shape[1] == generator_dim:
            y0 = y0.flatten(order='F')
        else:
            raise QiskitError('y0.shape is incompatible with specified generator.')

    return y0


def anti_herm_part(mat: Array) -> Array:
    """Get the anti-hermitian part of an operator."""
    if mat is None:
        return None

    return 0.5 * (mat - mat.conj().transpose())
