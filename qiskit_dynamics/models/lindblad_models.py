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
from .operator_collections import DenseLindbladCollection, DenseVectorizedLindbladCollection
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

    ###Need to add signal setting functionality. Currently does not have it.

    @property
    def signals(self) -> List[Array]:
        return [self._hamiltonian_signals,self._dissipator_signals]

    @property
    def signals(self,new_signals: List[Array]):
        self._hamiltonian_signals,self._dissipator_signals = new_signals

    @property
    def operators(self) -> List[Array]:
        return [self._hamiltonian_operators, self._dissipator_operators]

    @property
    def hilbert_space_dimension(self) -> int:
        return self._hamiltonian_operators.shape[-1]

    @property
    def evaluation_mode(self) -> str:
        return super().evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, new_mode: str):
        """Sets evaluation mode.
        Args:
            new_mode: new mode for evaluation. Supported modes:
                dense_lindblad_collection
                dense_vectorized_lindblad_collection"""
        if new_mode == "dense_lindblad_collection":
            self._operator_collection = DenseLindbladCollection(
                self._hamiltonian_operators,
                drift=self.drift,
                dissipator_operators=self._dissipator_operators,
            )
            self.frame.vectorized_operators=False
        elif new_mode == "dense_vectorized_lindblad_collection":
            self._operator_collection = DenseVectorizedLindbladCollection(
                self._hamiltonian_operators,
                drift=self.drift,
                dissipator_operators=self._dissipator_operators,
            )
            self.frame.vectorized_operators=True
        elif new_mode is None:
            pass
        else:
            raise NotImplementedError("Evaluation mode " + str(new_mode) + " is not supported.")
        self._evaluation_mode = new_mode

    def __init__(
        self,
        hamiltonian_operators: Array,
        hamiltonian_signals: Union[List[Signal], SignalList],
        dissipator_operators: Array = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        drift: Optional[Array] = None,
        frame: Optional[Union[Operator, Array, Frame]] = None,
        evaluation_mode: Optional[str] = "dense_lindblad_collection",
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
        self._operator_collection = None
        self._evaluation_mode = None

        if dissipator_operators is not None:
            dissipator_operators = Array(dissipator_operators)

        self._hamiltonian_operators = Array(np.array(hamiltonian_operators))
        self.drift = drift
        self._dissipator_operators = dissipator_operators

        if isinstance(hamiltonian_signals, list):
            hamiltonian_signals = SignalList(hamiltonian_signals)
        elif not isinstance(hamiltonian_signals, SignalList):
            raise Exception(
                """hamiltonian_signals must either be a list of
                             Signals, or a SignalList."""
            )

        if dissipator_signals is None:
            if dissipator_operators is not None:
                dissipator_signals = SignalList([1.0 for k in dissipator_operators])
            else:
                dissipator_signals = None
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

        self.evaluation_mode = evaluation_mode

    @classmethod
    def from_hamiltonian(
        cls,
        hamiltonian: HamiltonianModel,
        dissipator_operators: Optional[Array] = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        evaluation_mode: Optional[str] = "dense_lindblad_collection",
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
            evaluation_mode=evaluation_mode,
        )

    @property
    def frame(self):
        return self._frame

    @frame.setter
    def frame(self, frame: Union[Operator, Array, Frame]):
        if self._frame is not None and self._frame.frame_diag is not None:
            self.drift = self.drift + Array(np.diag(1j * self._frame.frame_diag))

            self._hamiltonian_operators = self.frame.operator_out_of_frame_basis(
                self._hamiltonian_operators
            )
            if self._dissipator_operators is not None:
                self._dissipator_operators = self.frame.operator_out_of_frame_basis(
                    self._dissipator_operators
                )
            self.drift = self.frame.operator_out_of_frame_basis(self.drift)

        self._frame = Frame(frame)

        if self._frame.frame_diag is not None:
            self._hamiltonian_operators = self.frame.operator_into_frame_basis(
                self._hamiltonian_operators
            )
            if self._dissipator_operators is not None:
                self._dissipator_operators = self.frame.operator_into_frame_basis(
                    self._dissipator_operators
                )
            self.drift = self.frame.operator_into_frame_basis(self.drift)

            self.drift = self.drift - Array(np.diag(1j * self._frame.frame_diag))

        # Ensure these changes are passed on to the operator collection.
        self.evaluation_mode = self.evaluation_mode

    def evaluate_generator(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        if self.evaluation_mode != "dense_vectorized_lindblad_collection":
            raise NotImplementedError(
                "Non-vectorized Lindblad models cannot be represented without a given state without vectorization."
            )
        else:
            return self._operator_collection.evaluate_generator(
                [self._hamiltonian_signals(time), self._dissipator_signals(time)]
            )

    def evaluate_rhs(
        self, time: Union[float, int], y: Array, in_frame_basis: Optional[bool] = False
    ) -> Array:
        """Evaluates the Lindblad model at a given time.
        time: time at which the model should be evaluated
        y: Density matrix as an (n,n) Array if not using a
            vectorized evaluation_mode or an (n^2) Array if
            using vectorized evaluation.
        in_frame_basis: whether the density matrix is in the
            frame already, and if the final result
            is returned in the frame or not."""

        hamiltonian_sig_vals = self._hamiltonian_signals(time)
        if self._dissipator_signals is not None:
            dissipator_sig_vals = self._dissipator_signals(time)
        else:
            dissipator_sig_vals = 0

        if self.frame.frame_diag is not None:

            # Take y out of the frame, but keep in the frame basis
            rhs = self.frame.operator_out_of_frame(
                time, y, operator_in_frame_basis=in_frame_basis, return_in_frame_basis=True
            )

            rhs = self._operator_collection.evaluate_rhs(
                [hamiltonian_sig_vals, dissipator_sig_vals], rhs
            )

            # Put rhs back into the frame, potentially converting its basis.
            rhs = self.frame.operator_into_frame(
                time, rhs, operator_in_frame_basis=True, return_in_frame_basis=in_frame_basis
            )

        else:
            rhs = self._operator_collection.evaluate_rhs(
                [hamiltonian_sig_vals, dissipator_sig_vals], y
            )

        return rhs
