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

import numpy as np
from typing import Union, List, Optional, Callable
from warnings import warn

from qiskit.quantum_info.operators import Operator
from ..models.frame import BaseFrame, Frame
from ..models.quantum_models import HamiltonianModel, LindbladModel
from ..models.operator_models import BaseOperatorModel, OperatorModel
from ..type_utils import StateTypeConverter, vec_commutator, to_array
from qiskit_ode.dispatch import Array
import qiskit_ode.dispatch as dispatch

class ODEProblem:
    """:class:`ODEProblem` represents first order ODE problems of the form

    .. math::
        y'(t) = f(t, y)

    in terms of the RHS function :math:`f(t, y)`. It is assumed that the
    state of the system is a complex :class:`Array` of arbitrary shape.

    This class provides a standardized interface for :method:`solve`
    to work with. Aside from storing the RHS function,
    it defines method signatures for transforming ODE states
    from the format the user expects, to a potentially different internal
    format. In this class these functions are trivial (they implement
    the identity function), but this interface facilitates use-cases in which
    an :class:`ODEProblem` subclass may modify the user representation of
    the problem for numerical reasons (e.g. solving in a different frame).
    """

    def __init__(self, rhs: Callable):
        """Initialize with an rhs function `rhs(t, y) = f(t, y)`.

        Args:
            rhs: callable function of the form :math:`f(t, y)``
        """
        self._rhs = dispatch.wrap(rhs, wrap_return=True)

    def user_state_to_problem(self, t: float, y: Array) -> Array:
        """Convert a user specified state at a given time into the internal
        problem format.

        Args:
            t: time
            y: state in user format

        Returns:
            Array state in the problem format
        """

        return Array(y)

    def problem_state_to_user(self, t: float, y: Array) -> Array:
        """Convert a state represented in the internal problem format into the
        user format.

        Args:
            t: time
            y: state in problem format

        Returns:
            Array state in the user format
        """

        return Array(y)

    def rhs(self, t: float, y: Array) -> Array:
        """Evaluate the rhs function.

        Args:
            t: time
            y: state
        """
        return self._rhs(t, y)

    def __call__(self, t: float, y: Array) -> Array:
        """Make the object callable."""
        return self.rhs(t, y)

class LMDEProblem(ODEProblem):
    """:class:`LMDEProblem` is the base class for representing the class of
    first order Linear Matrix Differential Equations (LMDEs), which are
    ODEs of the form:

    .. math::
        \dot{y}(t) = G(t)y(t),

    where :math:`G(t)` is called the generator, and the shapes of the state
    and :math:`G(t)` are such that the matrix multiplication are well defined.

    In addition to the interface inherited from :class:`ODEProblem`, this
    class exposes the generator function itself, which can be used by solvers
    specialized to this class (such as matrix exponentiation-based solvers).
    """

    def __init__(self, generator: Callable):
        """Initialize with the generator function :math:`G(t)`.

        Args:
            generator: the generator of the LMDE
        """
        self._generator = dispatch.wrap(generator, wrap_return=True)

    def rhs(self, t: float, y: Array) -> Array:
        """Standard default implementation of the RHS function given
        the generator.

        Args:
            t: time
            y: state
        """
        return self.generator(t) @ y

    def generator(self, t: float) -> Array:
        """Generator at a given time."""
        return self._generator(t)


class OperatorModelProblem(LMDEProblem):
    """Specialized subclass of :class:`LMDEProblem` in which the generator
    :math:`G(t)` is specified as a :class:`BaseOperatorModel` instance.

    Includes optional settings related to frame and cutoff frequency
    handling provided by :class:`BaseOperatorModel`:
        - Specify a frame to solve in.
        - Specify the frame the user specifies and recieves states.
        - Specify a cutoff frequency to solve with.

    Note: Internally, the system is solved in the basis in which the solver
    frame is diagonal.
    """

    def __init__(self,
                 generator: BaseOperatorModel,
                 solver_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 user_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 solver_cutoff_freq: Optional[float] = None,
                 state_type_converter: Optional[StateTypeConverter] = None):
        """Specify an :class:`LMDEProblem` via a :class:`BaseOperatorModel`.

        Args:
            generator: The generator specified as a :class:`BaseOperatorModel`.
            solver_frame: The frame to solve the system in. Default value of
                          `'auto'` allows this class to choose the frame. A
                          value of `None` implies solving in the lab frame.
            user_frame: The frame in which the user specifies the state of the
                        system. The default value of `'auto'` implies that the
                        state should be specified/returned in the frame of the
                        supplied generator.
            solver_cutoff_freq: Cutoff frequency to use when solving the system.
            state_type_converter: Optional setting for translating between
                                  the state type the user interacts with
                                  to the state type the solver needs.
        """

        self._state_type_converter = state_type_converter

        # copy the generator to preserve state of user's generator
        self._generator = generator.copy()

        # set user_frame
        if isinstance(user_frame, str) and user_frame == 'auto':
            self._user_frame = self._generator.frame
        else:
            self._user_frame = Frame(user_frame)

        # set solver frame
        if isinstance(solver_frame, str) and solver_frame == 'auto':
            # if auto, set it to the anti-hermitian part of the drift
            self._generator.frame = None
            self._generator.frame = anti_herm_part(self._generator.drift)
        else:
            self._generator.frame = Frame(solver_frame)

        # set up cutoff frequency
        self._generator.cutoff_freq = solver_cutoff_freq

    def user_state_to_problem(self, t: float, y: Array) -> Array:
        """Convert state from user side to internal solver representation.
        This requires:
            - Converting out of the `user_framer`.
            - Convert into the `solver_frame`.
            - Convert into the basis in which `solver_frame` is diagonal.

        Args:
            t: time
            y: state
        Returns:
            Array: the transformed state.
        """
        # convert to internal representation
        new_y = None
        if self._state_type_converter is None:
            new_y = y
        else:
            new_y = self._state_type_converter.outer_to_inner(y)

        # convert state from user_frame into lab frame
        new_y = self._user_frame.state_out_of_frame(t, new_y)

        # convert state from lab frame into solver frame, in basis in which
        # frame is diagonal
        return self._generator.frame.state_into_frame(t, new_y,
                                                      return_in_frame_basis=True)

    def problem_state_to_user(self, t: float, y: Array) -> Array:
        """The inverse transformation of :method:`user_state_to_problem`.

        Args:
            t: time
            y: state
        Returns:
            Array: the transformed state.
        """

        # take state out of solver frame and basis into lab frame
        new_y = self._generator.frame.state_out_of_frame(t, y,
                                                         y_in_frame_basis=True)

        # bring state into state_frame
        new_y = self._user_frame.state_into_frame(t, new_y)

        if self._state_type_converter is None:
            return new_y
        else:
            return self._state_type_converter.inner_to_outer(new_y)

    def rhs(self, t, y):
        """RHS function from the :class:`OperatorModel`."""
        return self._generator.lmult(t, y, in_frame_basis=True)

    def generator(self, t):
        """Generator from the :class:`OperatorModel`"""
        return self._generator.evaluate(t, in_frame_basis=True)


class SchrodingerProblem(OperatorModelProblem):
    """A differential equation problem for evolution of a state according
    to the Schrodinger equation: :math:`\dot{A}(t) = -i H(t)A(t)`, where
    :math:`H(t)` is a Hamiltonian given by an instance of
    :class:`HamiltonianModel`.
    """

    def __init__(self,
                 hamiltonian: HamiltonianModel,
                 solver_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 user_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 solver_cutoff_freq: Optional[float] = None):
        """Constructs an :class:`OperatorModelProblem` representing the
        Schrodinger equation.

        Args:
            hamiltonian: The Hamiltonian to simulate.
            solver_frame: The frame to solve the system in. Default value of
                          `'auto'` allows this class to choose the frame. A
                          value of `None` implies solving in the lab frame.
            user_frame: The frame in which the user specifies the state of the
                        system. The default value of `'auto'` implies that the
                        state should be specified/returned in the frame of the
                        supplied generator.
            solver_cutoff_freq: Cutoff frequency to use when solving the system.
        """

        generator = OperatorModel(operators=[-1j * op for
                                             op in hamiltonian._operators],
                                  signals=hamiltonian.signals,
                                  signal_mapping=hamiltonian.signal_mapping,
                                  frame=hamiltonian.frame,
                                  cutoff_freq=hamiltonian.cutoff_freq)

        super().__init__(generator=generator,
                         solver_frame=solver_frame,
                         user_frame=user_frame,
                         solver_cutoff_freq=solver_cutoff_freq)


class StatevectorProblem(SchrodingerProblem):
    """A differential equation problem for evolution of a statevector according
    to the Schrodinger equation: :math:`\dot{y}(t) = -i H(t)y(t)`, where
    :math:`H(t)` is a Hamiltonian given by an instance of
    :class:`HamiltonianModel`.
    """


class UnitaryProblem(SchrodingerProblem):
    """A differential equation problem for evolution of a unitary according
    to the Schrodinger equation: :math:`\dot{U}(t) = -i H(t)U(t)`, where
    :math:`H(t)` is a Hamiltonian given by an instance of
    :class:`HamiltonianModel`.
    """


class DensityMatrixProblem(OperatorModelProblem):
    """Simulate density matrix evolution according to the Lindblad equation:

    .. math::
        \dot{\rho}(t) = -i[H(t), \rho(t)] + \sum_j \gamma_j(t) L_j\rho(t)L_j^\dagger - \frac{1}{2}\{L_j^\daggerL_j, \rho(t)\}

    where:
        - :math:`H(t)` is the Hamiltonian,
        - :math:`L_j` are the noise operators, or dissipators, and
        - :math:`\gamma_j(t)` are the time-dependent dissipator coefficients,
    specified in terms of a :class:`LindbladModel` object.
    """

    def __init__(self,
                 lindblad_model: LindbladModel,
                 solver_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 user_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 solver_cutoff_freq: Optional[float] = None):

        """Constructs an :class:`OperatorModelProblem` representing the
        Lindblad equation, to act on a density matrix.

        Args:
            lindblad_model: The Lindblad model to simulate
            solver_frame: The frame to solve the system in. Default value of
                          `'auto'` allows this class to choose the frame. A
                          value of `None` implies solving in the lab frame.
            user_frame: The frame in which the user specifies the state of the
                        system. The default value of `'auto'` implies that the
                        state should be specified/returned in the frame of the
                        supplied generator.
            solver_cutoff_freq: Cutoff frequency to use when solving the system.
        """

        # handle user specified frames
        if not isinstance(solver_frame, str):
            # wrap in a frame
            solver_frame = Frame(solver_frame)
            # convert to vectorized
            solver_frame = vec_commutator(solver_frame.frame_operator)

        if not isinstance(user_frame, str):
            # wrap in a frame
            user_frame = Frame(user_frame)
            # convert to vectorized
            user_frame = vec_commutator(user_frame.frame_operator)

        # specify the converter to vectorize the density matrix
        dim = int(np.sqrt(lindblad_model._operators[0].shape[0]))
        outer_type_spec = {'type': 'array', 'shape': (dim, dim)}
        inner_type_spec = {'type': 'array', 'shape': (dim**2,)}
        converter = StateTypeConverter(inner_type_spec, outer_type_spec)

        super().__init__(generator=lindblad_model,
                         solver_frame=solver_frame,
                         user_frame=user_frame,
                         solver_cutoff_freq=solver_cutoff_freq,
                         state_type_converter=converter)


class SuperOpProblem(OperatorModelProblem):
    """Simulate super operator evolution according to the Lindblad equation.

    I.e. Construct the Lindblad equation as in :class:`LindbladProblem`,
    but simulate the propagator matrix for the differential equation.
    Solving this equation returns the super operator version of the
    resulting quantum channel (in column vectorization convention).
    """

    def __init__(self,
                 lindblad_model: LindbladModel,
                 solver_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 user_frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 solver_cutoff_freq: Optional[float] = None):

        """Constructs an :class:`OperatorModelProblem` representing the
        Lindblad equation, to act on a density matrix.

        Args:
            lindblad_model: The Lindblad model to simulate
            solver_frame: The frame to solve the system in. Default value of
                          `'auto'` allows this class to choose the frame. A
                          value of `None` implies solving in the lab frame.
            user_frame: The frame in which the user specifies the state of the
                        system. The default value of `'auto'` implies that the
                        state should be specified/returned in the frame of the
                        supplied generator.
            solver_cutoff_freq: Cutoff frequency to use when solving the system.
        """

        # handle user specified frames
        if not isinstance(solver_frame, str):
            # wrap in a frame
            solver_frame = Frame(solver_frame)
            # convert to vectorized
            solver_frame = vec_commutator(solver_frame.frame_operator)

        if not isinstance(user_frame, str):
            # wrap in a frame
            user_frame = Frame(user_frame)
            # convert to vectorized
            user_frame = vec_commutator(user_frame.frame_operator)

        super().__init__(generator=lindblad_model,
                         solver_frame=solver_frame,
                         user_frame=user_frame,
                         solver_cutoff_freq=solver_cutoff_freq)


def anti_herm_part(A: Array) -> Array:
    """Get the anti-hermitian part of an operator."""
    return 0.5 * (A - A.conj().transpose())
