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

from typing import Tuple, Union, List, Optional
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.type_utils import to_array
from qiskit_dynamics.signals import Signal, SignalList
from .generator_models import BaseGeneratorModel, GeneratorModel
from .hamiltonian_models import HamiltonianModel, is_hermitian
from .operator_collections import (
    BaseLindbladOperatorCollection
    DenseLindbladCollection,
    DenseVectorizedLindbladCollection,
    SparseLindbladCollection,
    SparseVectorizedLindbladCollection,
)
from .rotating_frame import RotatingFrame


class LindbladModel(BaseGeneratorModel):
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

    Instantiating an instance of :class:`~qiskit_dynamics.models.LindbladModel`
    requires specifying the above decomposition:

    .. code-block:: python

        lindblad_model = LindbladModel(hamiltonian_operators,
                                       hamiltonian_signals,
                                       static_hamiltonian,
                                       dissipator_operators,
                                       dissipator_signals)

    where the arguments ``hamiltonian_operators``, ``hamiltonian_signals``, and
    ``static_hamiltonian`` are for the Hamiltonian decomposition as in
    :class:`~qiskit_dynamics.models.HamiltonianModel`,
    and the ``dissipator_operators`` correspond to the :math:`L_j`, and the ``dissipator_signals``
    the :math:`g_j(t)`, which default to the constant ``1.``.
    """

    def __init__(
        self,
        hamiltonian_operators: Array,
        hamiltonian_signals: Union[List[Signal], SignalList] = None,
        dissipator_operators: Array = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        static_hamiltonian: Optional[Array] = None,
        rotating_frame: Optional[Union[Operator, Array, RotatingFrame]] = None,
        evaluation_mode: Optional[str] = "dense",
        validate: bool = True,
    ):
        """Initialize.

        Args:
            hamiltonian_operators: list of operators in Hamiltonian.
            hamiltonian_signals: list of signals in the Hamiltonian.
            dissipator_operators: list of dissipator operators.
            dissipator_signals: list of dissipator signals.
            static_hamiltonian: Optional, constant term in Hamiltonian.
            rotating_frame: rotating frame in which calcualtions are to be done.
                            If provided, it is assumed that all operators were
                            already in the frame basis.
            evaluation_mode: Evaluation mode to use. See ``LindbladModel.evaluation_mode``
                             for more details.
            validate: If True check input hamiltonian_operators and static_hamiltonian are
                      Hermitian.

        Raises:
            Exception: if signals incorrectly specified.
        """

        static_hamiltonian = to_numeric_matrix_type(static_hamiltonian)
        hamiltonian_operators = to_numeric_matrix_type(hamiltonian_operators)
        if validate:
            if (hamiltonian_operators is not None) and (not is_hermitian(hamiltonian_operators)):
                raise QiskitError("""LinbladModel hamiltonian_operators must be Hermitian.""")
            if (static_hamiltonian is not None) and (not is_hermitian(static_hamiltonian)):
                raise QiskitError("""LinbladModel static_hamiltonian must be Hermitian.""")

        self._operator_collection = self.construct_operator_collection(
            evaluation_mode=evaluation_mode,
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            dissipator_operators=dissipator_operators
        )
        self._evaluation_mode = evaluation_mode

        self._rotating_frame = None
        self.rotating_frame = rotating_frame

        self.signals = (hamiltonian_signals, dissipator_signals)

    @property
    def dim(self) -> int:
        if self._operator_collection.static_hamiltonian is not None:
            return self._operator_collection.static_hamiltonian.shape[-1]
        elif self._operator_collection.hamiltonian_operators is not None:
            return self._operator_collection.hamiltonian_operators[0].shape[-1]
        else:
            return self._operator_collection.dissipator_operators[0].shape[-1]

    @property
    def signals(self) -> Tuple[SignalList]:
        """Gets the Model's Signals.

        Returns:
            Tuple[] with 0th entry storing the Hamiltonian signals
            and the 1st entry storing the dissipator signals.
        """
        return (self._hamiltonian_signals, self._dissipator_signals)

    @signals.setter
    def signals(self, new_signals: Tuple[List[Signal]]):
        hamiltonian_signals, dissipator_signals = new_signals
        ham_ops, diss_ops = self.get_operators()

        if isinstance(hamiltonian_signals, list):
            hamiltonian_signals = SignalList(hamiltonian_signals)

        if hamiltonian_signals is not None:
            if not isinstance(hamiltonian_signals, SignalList):
                raise QiskitError(
                    """hamiltonian_signals must either be a list of
                                 Signals, or a SignalList."""
                )
            if ham_ops is not None and len(hamiltonian_signals) != len(ham_ops):
                raise QiskitError(
                    """hamiltonian_signals must have the same length as the
                       hamiltonian_operators."""
                )

        if dissipator_signals is None:
            if diss_ops is not None:
                dissipator_signals = SignalList([1.0] * len(diss_ops))
            else:
                dissipator_signals = None
        elif isinstance(dissipator_signals, list):
            dissipator_signals = SignalList(dissipator_signals)

        if dissipator_signals is not None:
            if not isinstance(dissipator_signals, SignalList):
                raise QiskitError(
                    """dissipator_signals must either be a list of
                                     Signals, or a SignalList."""
                )
            if diss_ops is not None and len(dissipator_signals) != len(diss_ops):
                raise QiskitError(
                    """dissipator_signals must have the same length as the
                       dissipator_operators."""
                )

        self._hamiltonian_signals = hamiltonian_signals
        self._dissipator_signals = dissipator_signals

    def get_operators(self, in_frame_basis: Optional[bool] = False) -> Tuple[Array]:
        ham_ops = self._operator_collection.hamiltonian_operators
        diss_ops = self._operator_collection.dissipator_operators
        if not in_frame_basis and self.rotating_frame is not None:
            ham_ops = self.rotating_frame.operator_out_of_frame_basis(ham_ops)
            diss_ops = self.rotating_frame.operator_out_of_frame_basis(diss_ops)

        return (ham_ops, diss_ops)

    def get_static_hamiltonian(self, in_frame_basis: Optional[bool] = False) -> Array:
        """Get the constant hamiltonian term.

        Args:
            in_frame_basis: Flag for whether the returned static_operator should be
            in the basis in which the frame is diagonal.
        Returns:
            The static operator term.
        """
        op = self._operator_collection.static_hamiltonian
        if not in_frame_basis and self.rotating_frame is not None:
            return self.rotating_frame.operator_out_of_frame_basis(op)
        else:
            return op

    def set_static_hamiltonian(
        self,
        new_static_hamiltonian: Array,
        operator_in_frame_basis: Optional[bool] = False,
    ):
        """Set the constant Hamiltonian term.
        Note that if the model has a rotating frame this will override
        any contributions to the static term due to the frame transformation.

        Args:
            new_static_hamiltonian: The static operator operator.
            operator_in_frame_basis: Whether `new_static_operator` is already in the rotating
            frame basis.
        """
        if new_static_hamiltonian is None:
            self._operator_collection.static_hamiltonian = None
        else:
            if not operator_in_frame_basis and self.rotating_frame is not None:
                new_static_hamiltonian = self.rotating_frame.operator_into_frame_basis(
                    new_static_hamiltonian
                )

            self._operator_collection.static_hamiltonian = new_static_hamiltonian

    @property
    def evaluation_mode(self) -> str:
        """Numerical evaluation mode of the model.

        Available options:

            - 'dense': Stores Hamiltonian and dissipator terms as dense
               Array types.
            - 'dense_vectorized': Stores the Hamiltonian and dissipator
              terms as (dim^2,dim^2) matrices that acts on a vectorized
              density matrix by left-multiplication. Allows for direct evaluate generator.
            - 'sparse': Like dense, but stores Hamiltonian components with
              `csr_matrix` types. Outputs will be dense if a 2d frame operator is
              used. Not compatible with jax.
            - `sparse_vectorized': Like dense_vectorized, but stores everything as csr_matrices.
        """
        return self._evaluation_mode

    @evaluation_mode.setter
    def evaluation_mode(self, new_mode: str):
        """Sets evaluation mode.

        Args:
            new_mode: String specifying new mode. Available options
                      are 'dense', 'sparse', 'dense_vectorized', and 'sparse_vectorized'.
                      See property doc string for details.

        Raises:
            NotImplementedError: if new_mode is not one of the above
            supported evaluation modes.
        """
        if new_mode != self._evaluation_mode:
            self._operator_collection = self.construct_operator_collection(
                new_mode,
                self._operator_collection.static_hamiltonian,
                self._operator_collection.hamiltonian_operators,
                self._operator_collection.dissipator_operators
            )

        self.vectorized_operators = "vectorized" in new_mode
        self._evaluation_mode = new_mode

    @classmethod
    def from_hamiltonian(
        cls,
        hamiltonian: HamiltonianModel,
        dissipator_operators: Optional[Array] = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        evaluation_mode: Optional[str] = None,
    ):
        """Construct from a :class:`HamiltonianModel`.

        Args:
            hamiltonian: The :class:`HamiltonianModel`.
            dissipator_operators: List of dissipator operators.
            dissipator_signals: List of dissipator signals.
            evaluation_mode: Evaluation mode. See LindbladModel.evaluation_mode
                for more information.

        Returns:
            LindbladModel: Linblad model from parameters.
        """

        if evaluation_mode is None:
            evaluation_mode = hamiltonian.evaluation_mode

        return cls(
            hamiltonian_operators=hamiltonian.get_operators(False),
            hamiltonian_signals=hamiltonian.signals,
            dissipator_operators=dissipator_operators,
            dissipator_signals=dissipator_signals,
            static_hamiltonian=hamiltonian.get_static_operator(False),
            evaluation_mode=evaluation_mode,
        )

    @property
    def rotating_frame(self):
        return self._rotating_frame

    @rotating_frame.setter
    def rotating_frame(self, rotating_frame: Union[Operator, Array, RotatingFrame]):
        new_frame = RotatingFrame(rotating_frame)

        new_static_hamiltonian = GeneratorModel.transfer_static_operator_between_frames(
            self.get_static_hamiltonian(in_frame_basis=True),
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        ham_ops, diss_ops = self.get_operators(in_frame_basis=True)

        new_hamiltonian_operators = self.transfer_operators_between_frames(
            ham_ops,
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )
        new_dissipator_operators = self.transfer_operators_between_frames(
            diss_ops,
            new_frame=new_frame,
            old_frame=self.rotating_frame,
        )

        self._rotating_frame = new_frame

        self._operator_collection = self.construct_operator_collection(
            evaluation_mode=self.evaluation_mode,
            static_hamiltonian=new_static_hamiltonian,
            hamiltonian_operators=new_hamiltonian_operators,
            dissipator_operators=new_dissipator_operators
        )

    def evaluate_hamiltonian(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        """Evaluates Hamiltonian matrix at a given time.

        Args:
            time: The time at which to evaluate the hamiltonian.
            in_frame_basis: Whether to evaluate in the basis in which
                the frame operator is diagonal.
        Returns:
            Array: Hamiltonian matrix."""
        hamiltonian_sig_vals = self._hamiltonian_signals(time)
        ham = self._operator_collection.evaluate_hamiltonian(hamiltonian_sig_vals)
        if self.rotating_frame.frame_diag is not None:
            ham = self.rotating_frame.operator_into_frame(
                time,
                ham,
                operator_in_frame_basis=True,
                return_in_frame_basis=in_frame_basis,
                vectorized_operators=self.vectorized_operators,
            )

        return ham

    def evaluate(self, time: float, in_frame_basis: Optional[bool] = False) -> Array:
        if self._dissipator_signals is not None:
            dissipator_sig_vals = self._dissipator_signals(time)
        else:
            dissipator_sig_vals = None
        if self.vectorized_operators:
            out = self._operator_collection.evaluate(
                self._hamiltonian_signals(time), dissipator_sig_vals
            )
            return self.rotating_frame.vectorized_map_into_frame(
                time, out, operator_in_frame_basis=True, return_in_frame_basis=in_frame_basis
            )
        else:
            raise NotImplementedError(
                "Non-vectorized Lindblad models cannot be represented without a given state."
            )

    def evaluate_rhs(
        self, time: Union[float, int], y: Array, in_frame_basis: Optional[bool] = False
    ) -> Array:
        """Evaluates the Lindblad model at a given time.

        Args:
            time: time at which the model should be evaluated.
            y: Density matrix as an (n,n) Array if not using a
               vectorized evaluation_mode or an (n^2) Array if
               using vectorized evaluation.
            in_frame_basis: whether the density matrix is in the
                            frame already, and if the final result
                            is returned in the rotating frame or not.

        Returns:
            Array: Either the evaluated generator or the state.
        """

        hamiltonian_sig_vals = self._hamiltonian_signals(time)
        if self._dissipator_signals is not None:
            dissipator_sig_vals = self._dissipator_signals(time)
        else:
            dissipator_sig_vals = None

        if self.rotating_frame.frame_diag is not None:

            # Take y out of the frame, but keep in the frame basis
            rhs = self.rotating_frame.operator_out_of_frame(
                time,
                y,
                operator_in_frame_basis=in_frame_basis,
                return_in_frame_basis=True,
                vectorized_operators=self.vectorized_operators,
            )

            rhs = self._operator_collection.evaluate_rhs(
                hamiltonian_sig_vals, dissipator_sig_vals, rhs
            )

            # Put rhs back into the frame, potentially converting its basis.
            rhs = self.rotating_frame.operator_into_frame(
                time,
                rhs,
                operator_in_frame_basis=True,
                return_in_frame_basis=in_frame_basis,
                vectorized_operators=self.vectorized_operators,
            )

        else:
            rhs = self._operator_collection.evaluate_rhs(
                hamiltonian_sig_vals, dissipator_sig_vals, y
            )

        return rhs

    @classmethod
    def construct_operator_collection(cls,
                                      evaluation_mode: str,
                                      static_hamiltonian: Union[None, Array, csr_matrix],
                                      hamiltonian_operators: Union[None, Array, List[csr_matrix]],
                                      dissipator_operators: Union[None, Array, List[csr_matrix]]) -> BaseLindbladOperatorCollection:
        """Sets evaluation mode.

        Args:
            new_mode: String specifying new mode. Available options
                      are 'dense', 'sparse', 'dense_vectorized', and 'sparse_vectorized'.
                      See property doc string for details.

        Raises:
            NotImplementedError: if new_mode is not one of the above
            supported evaluation modes.
        """
        if new_mode == "dense":
            return DenseLindbladCollection(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                dissipator_operators=dissipator_operators,
            )
            self.vectorized_operators = False
        elif new_mode == "sparse":
            return SparseLindbladCollection(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                dissipator_operators=dissipator_operators,
            )
            self.vectorized_operators = False
        elif new_mode == "dense_vectorized":
            return DenseVectorizedLindbladCollection(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                dissipator_operators=dissipator_operators,
            )
        elif new_mode == "sparse_vectorized":
            return SparseVectorizedLindbladCollection(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                dissipator_operators=dissipator_operators,
            )
        else:
            raise NotImplementedError(
                "Evaluation mode '"
                + str(new_mode)
                + "' is not supported. Call help("
                + str(self.__class__.__name__)
                + ".evaluation_mode) for available options."
            )
