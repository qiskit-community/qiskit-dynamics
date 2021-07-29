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
Lindblad models module.
"""

from typing import Union, List, Optional
import numpy as np
from qiskit.exceptions import QiskitError

from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.signals import Signal, SignalList
from .generator_models import GeneratorModel
from .hamiltonian_models import HamiltonianModel
from .operator_collections import DenseLindbladCollection
from .frame import Frame


class LindbladModel(GeneratorModel):
    r"""A model of a quantum system in terms of the Lindblad master equation.

    The Lindblad master equation models the evolution of a density matrix according to:

    .. math::
        \dot{\rho}(t) = -i[H(t), \rho(t)] + \mathcal{D}(t)(\rho(t)),

    where :math:`\mathcal{D}(t)` is the dissipator portion of the equation,
    given by

    .. math::
        \mathcal{D}(t)(\rho(t)) = \sum_j \gamma_j(t) L_j \rho L_j^\dagger
                                  - \frac{1}{2} \{L_j^\dagger L_j, \rho\},

    with :math:`[\cdot, \cdot]` and :math:`\{\cdot, \cdot\}` the
    matrix commutator and anti-commutator, respectively. In the above:

        - :math:`H(t)` denotes the Hamiltonian,
        - :math:`L_j` denotes the :math:`j^{th}` dissipator, or Lindblad,
          operator, and
        - :math:`\gamma_j(t)` denotes the signal corresponding to the
          :math:`j^{th}` Lindblad operator.
    """

    def __init__(
        self,
        hamiltonian_operators: List[Operator],
        hamiltonian_signals: Union[List[Signal], SignalList],
        dissipator_operators: Optional[List[Operator]] = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        drift: Optional[Array] = None,
        frame: Optional[Union[Operator, Array, Frame]] = None,
    ):
        """Initialize.

        Args:
            hamiltonian_operators: list of operators in Hamiltonian
            hamiltonian_signals: list of signals in the Hamiltonian
            dissipator_operators: list of dissipator operators
            dissipator_signals: list of dissipator signals
            drift: Optional, constant term in Hamiltonian
            frame: frame in which calcualtions are to be done.
                If provided, it is assumed that all operators were
                already in the frame basis.

        Raises:
            Exception: if signals incorrectly specified
        """
        if drift is not None:
            drift = Array(drift)
        if dissipator_operators is not None:
            dissipator_operators = Array(dissipator_operators)
        self._operator_collection = DenseLindbladCollection(
            Array(hamiltonian_operators), drift=drift, dissipator_operators=dissipator_operators
        )

        if isinstance(hamiltonian_signals, list):
            hamiltonian_signals = SignalList(hamiltonian_signals)
        elif not isinstance(hamiltonian_signals, SignalList):
            raise Exception(
                """hamiltonian_signals must either be a list of
                             Signals, or a SignalList."""
            )

        if dissipator_signals is None:
            dissipator_signals = SignalList([1.0 for k in dissipator_operators])
        elif isinstance(dissipator_signals, list):
            dissipator_signals = SignalList(dissipator_signals)
        elif not isinstance(dissipator_signals, SignalList):
            raise Exception(
                """dissipator_signals must either be a list of
                                 Signals, or a SignalList."""
            )

        self._hamiltonian_signals = hamiltonian_signals
        self._dissipator_signals = dissipator_signals
        
        self._frame = None
        self.frame = frame

    @classmethod
    def from_hamiltonian(
        cls,
        hamiltonian: HamiltonianModel,
        dissipator_operators: Optional[List[Operator]] = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
    ):
        """Construct from a :class:`HamiltonianModel`.

        Args:
            hamiltonian: the :class:`HamiltonianModel`.
            dissipator_operators: list of dissipator operators.
            dissipator_signals: list of dissipator signals.

        Returns:
            LindbladModel: Linblad model from parameters.
        """

        return cls(
            hamiltonian_operators=hamiltonian.operators,
            hamiltonian_signals=hamiltonian.signals,
            dissipator_operators=dissipator_operators,
            dissipator_signals=dissipator_signals,
            drift=hamiltonian.drift,
        )

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, Array, Frame]):
        if self._frame is not None and self._frame.frame_diag is not None:
            self._operator_collection.drift = self._operator_collection.drift + Array(
                np.diag(self._frame.frame_diag)
            )
            self._operator_collection.apply_function_to_operators(
                self.frame.operator_out_of_frame_basis
            )

        self._frame = Frame(frame)

        if self._frame.frame_diag is not None:
            self._operator_collection.apply_function_to_operators(
                self.frame.operator_into_frame_basis
            )
            self._operator_collection.drift = self._operator_collection.drift - Array(
                np.diag(self._frame.frame_diag)
            )

    def evaluate_without_state(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        raise NotImplementedError("Lindblad models cannot be represented without a given state without vectorization.")

    def evaluate_with_state(
        self, time: Union[float, int], y: Array, in_frame_basis: Optional[bool] = False
    ) -> Array:
        """Evaluates the Lindblad model at a given time.
        time: time at which the model should be evaluated
        y: Density matrix as an (n,n) Array
        in_frame_basis: whether the density matrix is in the
            frame already, and if the final result
            is returned in the frame or not."""

        if not in_frame_basis:
            y = self.frame.operator_into_frame_basis(y)

        hamiltonian_sig_vals = np.real(self._hamiltonian_signals.complex_value(time))
        dissipator_sig_vals = np.real(self._dissipator_signals.complex_value(time))

        # Need to check that I have the differences chosen correctly
        if self.frame.frame_diag is not None:
            frame_eigvals = self.frame.frame_diag
            pexp = np.exp((1j * time) * frame_eigvals)
            nexp = np.exp((-1j * time) * frame_eigvals)
            # Hopefully equivalent to rhs = e^{iFt} \rho e^{-iFt}
            rhs = np.outer(pexp, nexp) * y
            rhs = self._operator_collection.evaluate_with_state(
                [hamiltonian_sig_vals, dissipator_sig_vals], rhs
            )
            # Hopefully equivalent to rhs = e^{-iFt} rhs e^{iFt}
            rhs = np.outer(nexp, pexp) * rhs
        else:
            rhs = self._operator_collection.evaluate_with_state(
                [hamiltonian_sig_vals, dissipator_sig_vals], y
            )

        if not in_frame_basis:
            rhs = self.frame.operator_out_of_frame_basis(rhs)
        return rhs
