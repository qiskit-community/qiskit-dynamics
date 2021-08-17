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

from typing import Optional, Union, Callable, Tuple, Any, Type, List
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
from .models.generator_models import BaseGeneratorModel, CallableGenerator
from .models import HamiltonianModel

try:
    from jax.lax import scan
except ImportError:
    pass


def solve_ode(
    rhs: Callable,
    t_span: Array,
    y0: Union[Array, QuantumState, BaseOperator],
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
    y0, y0_cls = initial_state_converter(y0, return_class=True)

    rhs = dispatch.wrap(rhs)

    # solve the problem using specified method
    results = None
    if method in SOLVE_IVP_METHODS or (inspect.isclass(method) and issubclass(method, OdeSolver)):
        results = scipy_solve_ivp(rhs, t_span, y0, method, t_eval, **kwargs)
    elif isinstance(method, str) and method == "jax_odeint":
        results = jax_odeint(rhs, t_span, y0, t_eval, **kwargs)
    else:
        raise QiskitError("""Specified method is not a supported ODE method.""")
    if y0_cls is not None:
        results.y = [final_state_converter(i, y0_cls) for i in results.y]
    return results


def solve_lmde(
    generator: Union[Callable, BaseGeneratorModel],
    t_span: Array,
    y0: Union[Array, QuantumState, BaseOperator],
    method: Optional[Union[str, OdeSolver]] = "DOP853",
    t_eval: Optional[Union[Tuple, List, Array]] = None,
    input_frame: Optional[Union[str, Array]] = "auto",
    solver_frame: Optional[Union[str, Array]] = "auto",
    output_frame: Optional[Union[str, Array]] = "auto",
    solver_cutoff_freq: Optional[float] = None,
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
        input_frame: RotatingFrame that the initial state is specified in. If ``input_frame == 'auto'``,
                     defaults to using the frame the generator is specified in.
        solver_frame: RotatingFrame to solve the system in. If ``solver_frame == 'auto'``, defaults to
                      using the drift of the generator when specified as a
                      :class:`BaseGeneratorModel`.
        output_frame: RotatingFrame to return the results in. If ``output_frame == 'auto'``,
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
    t_span = Array(t_span)
    y0, y0_cls = initial_state_converter(y0, return_class=True)

    # setup input frame, output frame, and the internal solver generator based on args
    input_frame, output_frame, generator = setup_lmde_frames_and_generator(
        input_generator=generator,
        input_frame=input_frame,
        solver_frame=solver_frame,
        output_frame=output_frame,
        solver_cutoff_freq=solver_cutoff_freq,
    )

    # store shape of y0, and reshape y0 if necessary
    return_shape = y0.shape
    y0 = lmde_y0_reshape(generator_dim=generator(t_span[0]).shape[0], y0=y0)

    # map y0 from input frame into solver frame and basis
    y0 = input_frame.state_out_of_frame(t_span[0], y0)
    y0 = generator.rotating_frame.state_into_frame(t_span[0], y0, return_in_frame_basis=True)

    # define rhs functions in frame basis
    def solver_generator(t):
        return generator(t, in_frame_basis=True)

    def solver_rhs(t, y):
        return generator(t, y, in_frame_basis=True)

    if method == "scipy_expm":
        results = scipy_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    elif method == "jax_expm":
        results = jax_expm_solver(solver_generator, t_span, y0, t_eval=t_eval, **kwargs)
    else:
        # method is not LMDE-specific, so pass to solve_ode using rhs
        results = solve_ode(solver_rhs, t_span, y0, method=method, t_eval=t_eval, **kwargs)

    # convert any states in results to correct basis/frame
    output_states = None

    # pylint: disable=too-many-boolean-expressions
    if (
        results.y.backend == "jax"
        and (
            generator.rotating_frame.frame_diag is None
            or generator.rotating_frame.frame_diag.backend == "jax"
        )
        and (output_frame.frame_diag is None or output_frame.frame_diag.backend == "jax")
        and y0_cls is None
    ):
        # if all relevant objects are jax-compatible, run jax-customized version
        output_states = _jax_lmde_output_state_converter(
            results.t, results.y, generator.rotating_frame, output_frame, return_shape, y0_cls
        )
    else:
        output_states = []
        for idx in range(len(results.y)):
            time = results.t[idx]
            out_y = results.y[idx]

            # transform out of solver frame/basis into output frame
            out_y = generator.rotating_frame.state_out_of_frame(time, out_y, y_in_frame_basis=True)
            out_y = output_frame.state_into_frame(time, out_y)

            # reshape to match input shape if necessary
            out_y = final_state_converter(out_y.reshape(return_shape, order="F"), y0_cls)
            output_states.append(out_y)

    results.y = output_states

    return results


def setup_lmde_frames_and_generator(
    input_generator: Union[Callable, BaseGeneratorModel],
    input_frame: Optional[Union[str, Array]] = "auto",
    solver_frame: Optional[Union[str, Array]] = "auto",
    output_frame: Optional[Union[str, Array]] = "auto",
    solver_cutoff_freq: Optional[float] = None,
) -> Tuple[RotatingFrame, RotatingFrame, BaseGeneratorModel]:
    """Helper function for setting up internally used :class:`BaseGeneratorModel`
    for :meth:`solve_lmde`.

    Args:
        input_generator: User-supplied generator.
        input_frame: Input frame for the problem.
        solver_frame: RotatingFrame to solve in.
        output_frame: Output frame for the problem.
        solver_cutoff_freq: Cutoff frequency to use when solving.

    Returns:
        RotatingFrame, RotatingFrame, BaseGeneratorModel: input frame, output frame, and BaseGeneratorModel
    """

    generator = None

    # if not an instance of a subclass of BaseGeneratorModel assume Callable
    if not isinstance(input_generator, BaseGeneratorModel):
        generator = CallableGenerator(input_generator)
    else:
        generator = input_generator.copy()

    # set input and output frames
    if isinstance(input_frame, str) and input_frame == "auto":
        input_frame = generator.rotating_frame
    else:
        input_frame = RotatingFrame(input_frame)

    if isinstance(output_frame, str) and output_frame == "auto":
        output_frame = generator.rotating_frame
    else:
        output_frame = RotatingFrame(output_frame)

    # set solver frame
    # this must be done after input/output frames as it modifies the generator itself
    if isinstance(solver_frame, str) and solver_frame == "auto":
        # if auto, set it to the anti-hermitian part of the drift
        generator.rotating_frame = None

        if isinstance(generator, HamiltonianModel):
            generator.rotating_frame = -1j * generator.get_drift()
        else:
            generator.rotating_frame = anti_herm_part(generator.get_drift())
    else:
        generator.rotating_frame = RotatingFrame(solver_frame)

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
            y0 = y0.flatten(order="F")
        else:
            raise QiskitError("y0.shape is incompatible with specified generator.")

    return y0


def anti_herm_part(mat: Array) -> Array:
    """Get the anti-hermitian part of an operator."""
    if mat is None:
        return None

    return 0.5 * (mat - mat.conj().transpose())


def initial_state_converter(
    obj: Any, return_class: bool = False
) -> Union[Array, Tuple[Array, Type]]:
    """Convert initial state object to an Array.

    Args:
        obj: An initial state.
        return_class: Optional. If True return the class to use
                      for converting the output y Array.

    Returns:
        Array: the converted initial state if ``return_class=False``.
        tuple: (Array, class) if ``return_class=True``.
    """
    # pylint: disable=invalid-name
    y0_cls = None
    if isinstance(obj, Array):
        y0, y0_cls = obj, None
    if isinstance(obj, QuantumState):
        y0, y0_cls = Array(obj.data), obj.__class__
    elif isinstance(obj, QuantumChannel):
        y0, y0_cls = Array(SuperOp(obj).data), SuperOp
    elif isinstance(obj, (BaseOperator, Gate, QuantumCircuit)):
        y0, y0_cls = Array(Operator(obj.data)), Operator
    else:
        y0, y0_cls = Array(obj), None
    if return_class:
        return y0, y0_cls
    return y0


def final_state_converter(obj: Any, cls: Optional[Type] = None) -> Any:
    """Convert final state Array to custom class.

    Args:
        obj: final state Array.
        cls: Optional. The class to convert to.

    Returns:
        Any: the final state.
    """
    if cls is None:
        return obj

    if issubclass(cls, (BaseOperator, QuantumState)) and isinstance(obj, Array):
        return cls(obj.data)

    return cls(obj)


@requires_backend("jax")
def _jax_lmde_output_state_converter(
    times: Array,
    ys: Array,
    solver_frame: RotatingFrame,
    output_frame: RotatingFrame,
    return_shape: Tuple,
    y0_cls: object,
) -> Union[List, Array]:
    """Jax control-flow based output state converter for solve_lmde.

    Args:
        times: Array of times.
        ys: Array of output states.
        solver_frame: RotatingFrame of the solver (that the ys are specified in). Assumed
                      to be implemented with Jax backend.
        output_frame: RotatingFrame to be converted to.
        return_shape: Shape for output states.
        y0_cls: Output state return class.

    Returns:
        Union[List, Array]: output states
    """

    def scan_f(_, x):
        time, out_y = x
        out_y = solver_frame.state_out_of_frame(time, out_y, y_in_frame_basis=True)
        out_y = output_frame.state_into_frame(time, out_y)
        out_y = out_y.reshape(return_shape, order="F").data
        return None, out_y

    # scan, ensuring that the times and ys are in fact an Array
    final_states = scan(scan_f, init=None, xs=(Array(times).data, Array(ys).data))[1]

    # final class setting needs to be python-looped, if necessary
    if y0_cls is not None:
        output_states = []
        for state in final_states:
            output_states.append(final_state_converter(state, y0_cls))

        return output_states
    else:
        return Array(final_states)
