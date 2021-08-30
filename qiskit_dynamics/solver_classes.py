# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

r"""
=============================================================
Solver classes (:mod:`qiskit_dynamics.solver_classes`)
=============================================================

This is temporary, getting docs to build.

.. currentmodule:: qiskit_dynamics.solver_classes

.. autosummary::
   :toctree: ../stubs/

   Solver
"""


from typing import Optional, Union, Callable, Tuple, Any, Type, List
from copy import deepcopy

import numpy as np
from scipy.integrate import OdeSolver

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator, Statevector, DensityMatrix

from qiskit_dynamics.models import (BaseGeneratorModel, HamiltonianModel, LindbladModel,
                                    RotatingFrame, rotating_wave_approximation)
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics import solve_lmde


class Solver:
    """Solver object for simulating both Hamiltonian and Lindblad dynamics.
    Given the components of a Hamiltonian and optional dissipators, this class will
    internally construct either a :class:`HamiltonianModel` or :class:`LindbladModel`
    instance. The evolution given by the model can be simulated by calling :meth:`solve`,
    which automatically handles :mod:`qiskit.quantum_info` state and super operator types,
    and calls :func:`~qiskit_dynamics.solve.solve_lmde` to solve.
    """

    def __init__(self,
                 hamiltonian_operators: Array,
                 hamiltonian_signals: Optional[Union[List[Signal], SignalList]] = None,
                 dissipator_operators: Optional[Array] = None,
                 dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
                 drift: Optional[Array] = None,
                 rotating_frame: Optional[Union[Array, RotatingFrame]]=None,
                 evaluation_mode: Optional[str] = 'dense',
                 rwa_cutoff_freq: Optional[float] = None):
        """Initialize.

        Args:
            hamiltonian_operators: Hamiltonian operators.
            hamiltonian_signals: Coefficients for the Hamiltonian operators.
            dissipator_operators: Optional dissipation operators.
            dissipator_signals: Optional time-dependent coefficients for the dissipators. If
                                ``None``, coefficients are assumed to be the constant ``1.``.
            drift: Hamiltonian drift operator.
            rotating_frame: Rotating frame to transform the model into.
            evaluation_mode: Method for model evaluation. See documentation for
                             :meth:`HamiltonianModel.set_evaluation_mode` or
                             :meth:`LindbladModel.set_evaluation_mode`
                             (if dissipators in model) for valid modes.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency. If ``None``, no
                             approximation is made.
        """

        model = None
        if dissipator_operators is None:
            model = HamiltonianModel(operators=hamiltonian_operators,
                                     signals=hamiltonian_signals,
                                     rotating_frame=rotating_frame,
                                     drift=drift,
                                     evaluation_mode=evaluation_mode)
            self._signals = hamiltonian_signals
        else:
            model = LindbladModel(hamiltonian_operators=hamiltonian_operators,
                                  hamiltonian_signals=hamiltonian_signals,
                                  dissipator_operators=dissipator_operators,
                                  dissipator_signals=dissipator_signals,
                                  rotating_frame=rotating_frame,
                                  drift=drift,
                                  evaluation_mode=evaluation_mode)
            self._signals = (hamiltonian_signals, dissipator_signals)

        self._rwa_signal_map = None
        if rwa_cutoff_freq is not None:
            model, rwa_signal_map = rotating_wave_approximation(model,
                                                                rwa_cutoff_freq,
                                                                return_signal_map=True)
            self._rwa_signal_map = rwa_signal_map

        self._model = model

    @property
    def rotating_frame(self) -> RotatingFrame:
        """Rotating frame of the model."""
        return self.model.rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, new_rotating_frame: Union[Array, RotatingFrame]):
        """Set rotating frame for the model."""
        self.model.rotating_frame = new_rotating_frame

    @property
    def model(self) -> BaseGeneratorModel:
        """The model."""
        return self._model

    @property
    def signals(self) -> SignalList:
        """Solver signals."""
        return self._signals

    @signals.setter
    def signals(self, new_signals: Union[List[Signal], SignalList]):
        """Set signals for the solver, and pass to the model."""
        self._signals = new_signals
        if self._rwa_signal_map is not None:
            new_signals = self._rwa_signal_map(new_signals)
        self.model.signals = new_signals

    @property
    def evaluation_mode(self) -> str:
        """Evaluation mode for the model."""
        return self.model.evaluation_mode

    def set_evaluation_mode(self, new_mode: str):
        """Set model evaluation mode. See documentation for
        :class:`~qiskit_dynamics.models.HamiltonianModel`
        or :class:`~qiskit_dynamics.models.LindbladModel`
        (if dissipators present in model) for valid methods.
        """
        self.model.set_evaluation_mode(new_mode)

    def copy(self) -> 'Solver':
        """Return a copy of self."""
        return deepcopy(self)

    def solve(self,
              t_span: Array,
              y0: Union[Array, QuantumState, BaseOperator],
              **kwargs) -> OdeResult:
        r"""Solve the dynamical problem using :func:`~qiskit_dynamics.solve_lmde`.

        Special handling for various input types:
            - If `y0` is an `Array`, it is passed directly to
            :func:`~qiskit_dynamics.solve_lmde` as is. Acceptable array shapes are
            determined by the model type and evaluation mode.
            - If `y0` is a subclass of :class:`qiskit.quantum_info.QuantumState`:
                - If `self.model` is a :class:`~qiskit_dynamics.models.LindbladModel`,
                `y0` is converted to a :class:`DensityMatrix`. Further, if the model
                evaluation mode is vectorized `y0` will be suitably reshaped for solving.
                - If `self.model` is a :class:`~qiskit_dynamics.models.HamiltonianModel`,
                and `y0` a :class:`DensityMatrix`, the full unitary will be simulated,
                and the evolution of `y0` is attained via conjugation.
            - If `y0` is a subclass of :class`qiskit.quantum_info.QuantumChannel`, the full
            evolution map will be computed and composed with `y0`; either the unitary if
            `self.model` is a :class:`~qiskit_dynamics.models.HamiltonianModel`, or the full
            Lindbladian `SuperOp` if the model is a :class:`~qiskit_dynamics.models.LindbladModel`.

        Returns an `OdeResult` object in the style of `scipy.integrate.solve_ivp`, with results
        formatted to be the same types as the input.

        Args:
            t_span: Time interval to integrate over.
            y0: Initial state.
            kwargs: Keyword args passed to :meth:`~qiskit_dynamics.solve.solve_lmde`.

        Returns:
            OdeResult object with formatted output types.

        Raises:
            QiskitError: Initial state ``y0`` is of invalid shape.
        """

        # convert types
        if isinstance(y0, QuantumState) and isinstance(self.model, LindbladModel):
            y0 = DensityMatrix(y0)

        y0, y0_cls = initial_state_converter(y0, return_class=True)


        # validate types
        if ((y0_cls is SuperOp)
            and isinstance(self.model, LindbladModel)
            and 'vectorized' not in self.evaluation_mode):
            raise QiskitError("""Simulating SuperOp for a LinbladModel requires setting
                                 vectorized evaluation. Set evaluation_mode to a vectorized option.""")

        # validate y0 shape
        if isinstance(self.model, HamiltonianModel):
            if y0.shape[0] != self.model.dim or y0.ndim > 2:
                raise QiskitError("""Shape mismatch for initial state y0 and HamiltonianModel.""")
        elif isinstance(self.model, LindbladModel):
            if (('vectorized' not in self.model.evaluation_mode) and
                (y0.shape[-2:] != (self.model.dim, self.model.dim))
                ):
                raise QiskitError("""Shape mismatch for initial state y0 and LindbladModel.""")
            elif (('vectorized' in self.model.evaluation_mode) and
                  (y0.shape[0] != self.model.dim**2 or y0.ndim > 2)
                  ):
                raise QiskitError("""Shape mismatch for initial state y0 and LindbladModel
                                     in vectorized evaluation mode.""")

        # modify initial state for some custom handling of certain scenarios
        y_input = y0

        # if Simulating density matrix or SuperOp with a HamiltonianModel, simulate the unitary
        if y0_cls in [DensityMatrix, SuperOp] and isinstance(self.model, HamiltonianModel):
            y0 = np.eye(self.model.dim, dtype=complex)
        # if LindbladModel is vectorized and simulating a density matrix, flatten
        elif ((y0_cls is DensityMatrix)
              and isinstance(self.model, LindbladModel)
              and 'vectorized' in self.evaluation_mode):
            y0 = y0.flatten(order="F")

        results = solve_lmde(generator=self.model,
                             t_span=t_span,
                             y0=y0,
                             **kwargs)

        # handle special cases
        if y0_cls is DensityMatrix and isinstance(self.model, HamiltonianModel):
            # conjugate by unitary
            # this is a bit sketch, depends on output types of simplified solve_lmde
            out = Array(results.y)
            results.y = out @ y_input @ out.conj().transpose((0, 2, 1))
        elif y0_cls is SuperOp and isinstance(self.model, HamiltonianModel):
            # convert to SuperOp and compose
            # need to test jax here, and also test if this is workign properly
            out = Array(results.y)
            results.y = np.einsum('nka,nlb->nklab', out.conj(), out).reshape(out.shape[0],
                                                                             out.shape[1]**2,
                                                                             out.shape[1]**2,
                                                                             order='F') @ y_input
        elif ((y0_cls is DensityMatrix)
              and isinstance(self.model, LindbladModel)
              and 'vectorized' in self.evaluation_mode):
              results.y = Array(results.y).reshape((len(results.y),) + y_input.shape, order='F')

        if y0_cls is not None:
            results.y = [final_state_converter(yi, y0_cls) for yi in results.y]

        return results


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
