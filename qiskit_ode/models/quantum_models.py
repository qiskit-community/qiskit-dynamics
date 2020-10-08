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

from typing import Callable, Union, List, Optional
import numpy as np
from .signals import VectorSignal, Constant, Signal, BaseSignal
from qiskit.quantum_info.operators import Operator
from .frame import Frame
from .operator_models import OperatorModel
from ..type_utils import vec_commutator, vec_dissipator, to_array

class HamiltonianModel(OperatorModel):
    """A model of a Hamiltonian, i.e. a time-dependent operator of the form

    .. math::

        H(t) = \sum_{i=0}^{k-1} s_i(t) H_i,

    where :math:`H_i` are Hermitian operators, and the :math:`s_i(t)` are
    time-dependent functions represented by :class:`Signal` objects.

    Currently the functionality of this class is as a subclass of
    :class:`OperatorModel`, with the following modifications:
        - The operators in the linear decomposition are verified to be
          Hermitian.
        - Frames are dealt with assuming the structure of the Schrodinger
          equation. I.e. Evaluating the Hamiltonian :math:`H(t)` in a
          frame :math:`F = -iH`, evaluates the expression
          :math:`e^{-tF}H(t)e^{tF} - H`. This is in contrast to
          the base class :class:`OperatorModel`, which would ordinarily
          evaluate :math:`e^{-tF}H(t)e^{tF} - F`.
    """

    def __init__(self,
                 operators: List[Operator],
                 signals: Optional[Union[VectorSignal, List[Signal]]] = None,
                 signal_mapping: Optional[Callable] = None,
                 frame: Optional[Union[Operator, np.array]] = None,
                 cutoff_freq: Optional[float] = None):
        """Initialize, ensuring that the operators are Hermitian.

        Args:
            operators: list of Operator objects.
            signals: Specifiable as either a VectorSignal, a list of
                     Signal objects, or as the inputs to signal_mapping.
                     OperatorModel can be instantiated without specifying
                     signals, but it can not perform any actions without them.
            signal_mapping: a function returning either a
                            VectorSignal or a list of Signal objects.
            frame: Rotating frame operator. If specified with a 1d
                            array, it is interpreted as the diagonal of a
                            diagonal matrix.
            cutoff_freq: Frequency cutoff when evaluating the model.
        """

        # verify operators are Hermitian, and if so instantiate
        for operator in operators:
            if isinstance(operator, Operator):
                operator = operator.data

            if np.linalg.norm((operator.conj().transpose()
                                - operator).data) > 1e-10:
                raise Exception("""HamiltonianModel only accepts Hermitian
                                    operators.""")

        super().__init__(operators=operators,
                         signals=signals,
                         signal_mapping=signal_mapping,
                         frame=frame,
                         cutoff_freq=cutoff_freq)

    def evaluate(self, time: float, in_frame_basis: bool = False) -> np.array:
        """Evaluate the Hamiltonian at a given time.

        Note: This function from :class:`OperatorModel` needs to be overridden,
        due to frames for Hamiltonians being relative to the Schrodinger
        equation, rather than the Hamiltonian itself.
        See the class doc string for details.

        Args:
            time: Time to evaluate the model
            in_frame_basis: Whether to evaluate in the basis in which the frame
                            operator is diagonal

        Returns:
            np.array: the evaluated model
        """

        if self.signals is None:
            raise Exception("""OperatorModel cannot be
                               evaluated without signals.""")

        sig_vals = self.signals.value(time)

        op_combo = self._evaluate_in_frame_basis_with_cutoffs(sig_vals)

        op_to_add_in_fb = None
        if self.frame.frame_operator is not None:
            op_to_add_in_fb = -1j * np.diag(self.frame.frame_diag)


        return self.frame._conjugate_and_add(time,
                                             op_combo,
                                             op_to_add_in_fb=op_to_add_in_fb,
                                             operator_in_frame_basis=True,
                                             return_in_frame_basis=in_frame_basis)


class LindbladModel(OperatorModel):
    """A model of a quantum system, consisting of a hamiltonian
    and an optional description of dissipative dynamics.

    Dissipation terms are understood in terms of the Lindblad master
    equation:

    .. math::
        \dot{\rho}(t) = -i[H(t), \rho(t)] + \mathcal{D}(t)(\rho(t)),

    where :math:`\mathcal{D}(t)` is the dissipator portion of the equation,
    given by

    .. math::
        \mathcal{D}(t)(\rho(t)) = \sum_j \gamma_j(t) L_j \rho L_j^\dagger - \frac{1}{2} \{L_j^\dagger L_j, \rho\},

    with :math:`[\cdot, \cdot]` and :math:`\{\cdot, \cdot\}` the
    matrix commutator and anti-commutator, respectively. In the above:
        - :math:`H(t)` denotes the Hamiltonian,
        - :math:`L_j` denotes the :math:`j^{th}` noise, or Lindblad,
          operator, and
        - :math:`\gamma_j(t)` denotes the signal corresponding to the
          :math:`j^{th}` Lindblad operator.
    """

    def __init__(self,
                 hamiltonian_operators: List[Operator],
                 hamiltonian_signals: Union[List[BaseSignal], VectorSignal],
                 noise_operators: Optional[List[Operator]] = None,
                 noise_signals: Optional[Union[List[BaseSignal], VectorSignal]] = None):
        """Initialize.

        Args:
            hamiltonian_operators: list of operators in Hamiltonian
            hamiltonian_signals: list of signals in the Hamiltonian
            noise_operators: list of noise operators
            noise_signals: list of noise signals
        """

        # combine operators
        vec_ham_ops = -1j * vec_commutator(to_array(hamiltonian_operators))

        full_operators = None
        if noise_operators is not None:
            vec_diss_ops = vec_dissipator(to_array(noise_operators))
            full_operators = np.append(vec_ham_ops, vec_diss_ops, axis=0)
        else:
            full_operators = vec_ham_ops

        # combine signals
        if isinstance(hamiltonian_signals, list):
            hamiltonian_signals = VectorSignal.from_signal_list(hamiltonian_signals)
        elif not isinstance(hamiltonian_signals, VectorSignal):
            raise Exception("""hamiltonian_signals must either be a list of
                             Signals, or a VectorSignal.""")

        full_signals = None
        if noise_operators is None:
            full_signals = hamiltonian_signals
        else:
            if noise_signals is None:
                sig_val = np.ones(len(noise_operators), dtype=complex)
                carrier_freqs = np.zeros(len(noise_operators), dtype=float)
                noise_signals = VectorSignal(envelope=lambda t: sig_val,
                                             carrier_freqs=carrier_freqs)
            elif isinstance(noise_signals, list):
                noise_signals = VectorSignal.from_signal_list(noise_signals)
            elif not isinstance(noise_signals, VectorSignal):
                raise Exception("""noise_signals must either be a list of
                                 Signals, or a VectorSignal.""")


            full_envelope = lambda t: np.append(hamiltonian_signals.envelope(t),
                                                noise_signals.envelope(t))
            full_carrier_freqs = np.append(hamiltonian_signals.carrier_freqs,
                                           noise_signals.carrier_freqs)

            full_drift_array = np.append(hamiltonian_signals.drift_array,
                                         noise_signals.drift_array)

            full_signals = VectorSignal(envelope=full_envelope,
                                        carrier_freqs=full_carrier_freqs,
                                        drift_array=full_drift_array)

        super().__init__(operators=full_operators,
                         signals=full_signals)

    @classmethod
    def from_hamiltonian(cls,
                         hamiltonian: HamiltonianModel,
                         noise_operators: Optional[List[Operator]] = None,
                         noise_signals: Optional[Union[List[BaseSignal], VectorSignal]] = None):
        """Construct from a :class:`HamiltonianModel`.

        Args:
            hamiltonian: the :class:`HamiltonianModel`.
            noise_operators: list of noise operators.
            noise_signals: list of noise signals.
        """

        return cls(hamiltonian_operators=hamiltonian._operators,
                   hamiltonian_signals=hamiltonian.signals,
                   noise_operators=noise_operators,
                   noise_signals=noise_signals)
