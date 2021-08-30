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

from typing import Optional, Union, Callable, Tuple, Any, Type, List
import inspect

import numpy as np
from scipy.integrate import OdeSolver

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit.quantum_info import Statevector, DensityMatrix, SuperOp

from qiskit_dynamics.models import (HamiltonianModel, LindbladModel,
                                    RotatingFrame, rotating_wave_approximation)
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics import solve_lmde

# these will likely move
from .solve import initial_state_converter, final_state_converter


class Solver:

    def __init__(self,
                 hamiltonian_operators=None,
                 hamiltonian_signals=None,
                 dissipator_operators=None,
                 dissipator_signals=None,
                 drift=None,
                 rotating_frame=None,
                 evaluation_mode='dense',
                 rwa_cutoff_freq = None):

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
    def rotating_frame(self):
        return self.model.rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, new_rotating_frame):
        self.model.rotating_frame = new_rotating_frame

    @property
    def model(self):
        return self._model

    @property
    def signals(self):
        return self._signals

    @signals.setter
    def signals(self, new_signals):
        self._signals = new_signals
        if self._rwa_signal_map is not None:
            new_signals = self._rwa_signal_map(new_signals)
        self.model.signals = new_signals

    @property
    def evaluation_mode(self):
        return self.model.evaluation_mode

    def set_evaluation_mode(self, new_mode):
        """Set model evaluation mode. What to actually do here?
        """
        self.model.set_evaluation_mode(new_mode)

    def solve(self,
              t_span,
              y0,
              **kwargs) -> OdeResult:
        """Solve the dynamical problem.

        Logic:
            - based on class of y0, want to make some decisions about what to do.
                - input types can be:
                    - array/Array/Operator
                    - StateVector
                    - DensityMatrix
                    - SuperOp
                - if given a SuperOp and/or a DensityMatrix, we can simulate the unitary only
                and then return the results according to the standard formulas
                - For lindblad dynamics, StateVector can be automatically converted to a density
                matrix
                - For array/Array/Operator inputs, do no conversions, it will be assumed to be of a shape
                that can be handled by the given model and solve_lmde

        Args:
            t_span: Time interval to integrate over.
            y0: Initial state.
            kwargs: Keyword args passed to :meth:`solve_lmde`.

        returns:
            OdeResult object with formatted output types.
        """

        # initial conversions
        if isinstance(y0, Statevector) and isinstance(self.model, LindbladModel):
            y0 = DensityMatrix(y0)

        # handle case for Hamiltonian simulation but with density/superop
        y0, y0_cls = initial_state_converter(y0, return_class=True)

        # validate y0 for unwrapped case initial states
        if y0_cls is None:
            if isinstance(self.model, HamiltonianModel):
                if y0.shape[0] != self.model.dim or y0.ndim > 2:
                    raise QiskitError("""Shape mismatch for array y0 and HamiltonianModel.""")
            elif isinstance(self.model, LindbladModel):
                if (('vectorized' not in self.model.evaluation_mode) and
                    (y0.shape[-2:] != (self.model.dim, self.model.dim))
                    ):
                    raise QiskitError("""Shape mismatch for array y0 and LindbladModel.""")
                elif (('vectorized' in self.model.evaluation_mode) and
                      (y0.shape[0] != self.model.dim**2 or y0.ndim > 2)
                      ):
                    raise QiskitError("""Shape mismatch for array y0 and LindbladModel
                                         in vectorized evaluation mode.""")

        if ((y0_cls is SuperOp)
            and isinstance(self.model, LindbladModel)
            and 'vectorized' not in self.evaluation_mode):
            raise QiskitError("""Simulating SuperOp for a LinbladModel requires setting
                                 vectorized evaluation. Set evaluation_mode to a vectorized option.""")

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
                                                                             out.shape[1]**2) @ y_input
        elif ((y0_cls is DensityMatrix)
              and isinstance(self.model, LindbladModel)
              and 'vectorized' in self.evaluation_mode):
              results.y = Array(results.y).reshape((len(results.y),) + y_input.shape, order='F')

        if y0_cls is not None:
            results.y = [final_state_converter(yi, y0_cls) for yi in results.y]

        return results
