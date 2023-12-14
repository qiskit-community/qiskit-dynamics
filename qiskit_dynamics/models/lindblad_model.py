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
Lindblad model module.
"""

from typing import Tuple, Union, List, Optional
from warnings import warn

from qiskit import QiskitError
from qiskit_dynamics.arraylias.alias import ArrayLike
from qiskit_dynamics.signals import Signal, SignalList
from .generator_model import (
    BaseGeneratorModel,
    _static_operator_into_frame_basis,
    _operators_into_frame_basis,
)
from .hamiltonian_model import HamiltonianModel, is_hermitian
from .operator_collections import (
    LindbladCollection,
    ScipySparseLindbladCollection,
    VectorizedLindbladCollection,
    ScipySparseVectorizedLindbladCollection,
)
from .rotating_frame import RotatingFrame

try:
    import jax
except ImportError:
    pass


class LindbladModel(BaseGeneratorModel):
    r"""A model of a quantum system in terms of the Lindblad master equation.

    The Lindblad master equation models the evolution of a density matrix according to:

    .. math::

        \dot{\rho}(t) = -i[H(t), \rho(t)] + \mathcal{D}_0(\rho(t)) + \mathcal{D}(t)(\rho(t)),

    where :math:`\mathcal{D}_0` is the static dissipator portion, and :math:`\mathcal{D}(t)` is the
    time-dependent dissipator portion of the equation, given by

    .. math::

        \mathcal{D}_0(\rho(t)) = \sum_j N_j \rho N_j^\dagger
                                      - \frac{1}{2} \{N_j^\dagger N_j, \rho\},

    and

    .. math::

        \mathcal{D}(t)(\rho(t)) = \sum_j \gamma_j(t) L_j \rho L_j^\dagger
                                  - \frac{1}{2} \{L_j^\dagger L_j, \rho\},

    respectively. In the above:

        - :math:`[\cdot, \cdot]` and :math:`\{\cdot, \cdot\}` respectively denote the matrix
          commutator and anti-commutator,
        - :math:`H(t)` denotes the Hamiltonian,
        - :math:`N_j` denotes the operators appearing in the static dissipator,
        - :math:`L_j` denotes the operators appearing in the time-dpendent portion of the
          dissipator, and
        - :math:`\gamma_j(t)` denotes the signal corresponding to the
          :math:`j^{th}` time-dependent dissipator operator.

    Instantiating an instance of :class:`~qiskit_dynamics.models.LindbladModel` requires specifying
    the above decomposition:

    .. code-block:: python

        lindblad_model = LindbladModel(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            hamiltonian_signals=hamiltonian_signals,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
            dissipator_signals=dissipator_signals
        )

    where the arguments ``hamiltonian_operators``, ``hamiltonian_signals``, and
    ``static_hamiltonian`` are for the Hamiltonian decomposition as in
    :class:`~qiskit_dynamics.models.HamiltonianModel`,
    and the ``static_dissipators`` correspond to the :math:`N_j`, the ``dissipator_operators``
    to the :math:`L_j`, and the ``dissipator_signals`` the :math:`\gamma_j(t)`.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[ArrayLike] = None,
        hamiltonian_operators: Optional[ArrayLike] = None,
        hamiltonian_signals: Optional[Union[List[Signal], SignalList]] = None,
        static_dissipators: Optional[ArrayLike] = None,
        dissipator_operators: Optional[ArrayLike] = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        rotating_frame: Optional[Union[ArrayLike, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        array_library: Optional[str] = None,
        vectorized: bool = False,
        validate: bool = True,
    ):
        """Initialize.

        Args:
            static_hamiltonian: Constant term in Hamiltonian.
            hamiltonian_operators: List of operators in Hamiltonian with time-dependent
                coefficients.
            hamiltonian_signals: Time-dependent coefficients for hamiltonian_operators.
            static_dissipators: List of dissipators with coefficient 1.
            dissipator_operators: List of dissipator operators with time-dependent coefficients.
            dissipator_signals: Time-dependent coefficients for dissipator_operators.
            rotating_frame: Rotating frame in which calcualtions are to be done. If provided, it is
                assumed that all operators were already in the frame basis.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                frame operator is diagonalized.
            array_library: Array library for storing the operators in the model. Supported options
                are ``'numpy'``, ``'jax'``, ``'jax_sparse'``, and ``'scipy_sparse'``. If ``None``,
                the arrays will be handled by general dispatching rules.
            vectorized: Whether or not to setup the Lindblad equation in vectorized mode.
                If ``True``, the operators in the model are stored as :math:`(dim^2,dim^2)` matrices
                that act on vectorized density matrices by left-multiplication. Setting this to
                ``True`` is necessary for ``SuperOp`` simulation.
            validate: If True check input hamiltonian_operators and static_hamiltonian are
                Hermitian.

        Raises:
            QiskitError: If model insufficiently or incorrectly specified.
        """

        if (
            static_hamiltonian is None
            and hamiltonian_operators is None
            and static_dissipators is None
            and dissipator_operators is None
        ):
            raise QiskitError(
                f"{type(self).__name__} requires at least one of static_hamiltonian "
                "hamiltonian_operators, static_dissipators, or dissipator_operators "
                "to be specified at construction."
            )

        if validate:
            if (static_hamiltonian is not None) and (not is_hermitian(static_hamiltonian)):
                raise QiskitError("""LinbladModel static_hamiltonian must be Hermitian.""")
            if (hamiltonian_operators is not None) and any(
                not is_hermitian(op) for op in hamiltonian_operators
            ):
                raise QiskitError("""LindbladModel hamiltonian_operators must be Hermitian.""")

        self._array_library = array_library
        self._vectorized = vectorized
        self._rotating_frame = RotatingFrame(rotating_frame)
        self._in_frame_basis = in_frame_basis

        # jax sparse arrays cannot be used directly at this stage
        setup_library = array_library
        if array_library == "jax_sparse":
            setup_library = "jax"

        # set up internal operators
        if static_hamiltonian is not None:
            static_hamiltonian = -1j * static_hamiltonian
        static_hamiltonian = _static_operator_into_frame_basis(
            static_operator=static_hamiltonian,
            rotating_frame=self._rotating_frame,
            array_library=setup_library,
        )
        if static_hamiltonian is not None:
            static_hamiltonian = 1j * static_hamiltonian

        hamiltonian_operators = _operators_into_frame_basis(
            operators=hamiltonian_operators,
            rotating_frame=self._rotating_frame,
            array_library=setup_library,
        )

        static_dissipators = _operators_into_frame_basis(
            operators=static_dissipators,
            rotating_frame=self._rotating_frame,
            array_library=setup_library,
        )

        dissipator_operators = _operators_into_frame_basis(
            operators=dissipator_operators,
            rotating_frame=self._rotating_frame,
            array_library=setup_library,
        )

        self._operator_collection = _get_lindblad_operator_collection(
            array_library=array_library,
            vectorized=vectorized,
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
        )

        self.signals = (hamiltonian_signals, dissipator_signals)

    @classmethod
    def from_hamiltonian(
        cls,
        hamiltonian: HamiltonianModel,
        static_dissipators: Optional[ArrayLike] = None,
        dissipator_operators: Optional[ArrayLike] = None,
        dissipator_signals: Optional[ArrayLike] = None,
        array_library: Optional[str] = None,
        vectorized: bool = False,
    ):
        """Construct from a :class:`HamiltonianModel`.

        Args:
            hamiltonian: The :class:`HamiltonianModel`.
            static_dissipators: List of dissipators with coefficient 1.
            dissipator_operators: List of dissipators with time-dependent coefficients.
            dissipator_signals: List time-dependent coefficients for dissipator_operators.
            array_library: Array library to use.
            vectorized: Whether or not to vectorize the Lindblad equation.

        Returns:
            LindbladModel: Linblad model from parameters.
        """

        # store whether hamiltonian is in_frame_basis and set to False
        in_frame_basis = hamiltonian.in_frame_basis
        hamiltonian.in_frame_basis = False

        # get the operators
        static_hamiltonian = hamiltonian.static_operator
        hamiltonian_operators = hamiltonian.operators

        # return to previous value
        hamiltonian.in_frame_basis = in_frame_basis

        return cls(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            hamiltonian_signals=hamiltonian.signals,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
            dissipator_signals=dissipator_signals,
            rotating_frame=hamiltonian.rotating_frame,
            in_frame_basis=hamiltonian.in_frame_basis,
            array_library=array_library,
            vectorized=vectorized,
        )

    @property
    def dim(self) -> int:
        """The matrix dimension."""
        if self._operator_collection.static_hamiltonian is not None:
            return self._operator_collection.static_hamiltonian.shape[-1]
        elif self._operator_collection.hamiltonian_operators is not None:
            return self._operator_collection.hamiltonian_operators[0].shape[-1]
        elif self._operator_collection.static_dissipators is not None:
            return self._operator_collection.static_dissipators[0].shape[-1]
        else:
            return self._operator_collection.dissipator_operators[0].shape[-1]

    @property
    def signals(self) -> Tuple[SignalList]:
        """The model's signals as tuple with the 0th entry storing the Hamiltonian signals and the
        1st entry storing the dissipator signals.

        Raises:
            QiskitError: If, when setting this property, the given signals are incompatible with
                operator structure.
        """
        return (self._hamiltonian_signals, self._dissipator_signals)

    @signals.setter
    def signals(self, new_signals: Tuple[List[Signal]]):
        hamiltonian_signals, dissipator_signals = new_signals

        # set Hamiltonian signals
        if hamiltonian_signals is None:
            self._hamiltonian_signals = None
        elif hamiltonian_signals is not None and self.hamiltonian_operators is None:
            raise QiskitError("Hamiltonian signals must be None if hamiltonian_operators is None.")
        else:
            # if signals is a list, instantiate a SignalList
            if isinstance(hamiltonian_signals, list):
                hamiltonian_signals = SignalList(hamiltonian_signals)

            # if it isn't a SignalList by now, raise an error
            if not isinstance(hamiltonian_signals, SignalList):
                raise QiskitError("Hamiltonian signals specified in unaccepted format.")

            # verify signal length is same as operators
            if isinstance(self.hamiltonian_operators, list):
                len_hamiltonian_operators = len(self.hamiltonian_operators)
            else:
                len_hamiltonian_operators = self.hamiltonian_operators.shape[0]
            if len(hamiltonian_signals) != len_hamiltonian_operators:
                raise QiskitError(
                    "Hamiltonian signals need to have the same length as Hamiltonian operators."
                )

            self._hamiltonian_signals = hamiltonian_signals

        # set dissipator signals
        if dissipator_signals is None:
            self._dissipator_signals = None
        elif dissipator_signals is not None and self.dissipator_operators is None:
            raise QiskitError("Dissipator signals must be None if dissipator_operators is None.")
        else:
            # if signals is a list, instantiate a SignalList
            if isinstance(dissipator_signals, list):
                dissipator_signals = SignalList(dissipator_signals)

            # if it isn't a SignalList by now, raise an error
            if not isinstance(dissipator_signals, SignalList):
                raise QiskitError("Dissipator signals specified in unaccepted format.")

            # verify signal length is same as operators
            if isinstance(self.dissipator_operators, list):
                len_dissipator_operators = len(self.dissipator_operators)
            else:
                len_dissipator_operators = self.dissipator_operators.shape[0]
            if len(dissipator_signals) != len_dissipator_operators:
                raise QiskitError(
                    "Dissipator signals need to have the same length as dissipator operators."
                )

            self._dissipator_signals = dissipator_signals

    @property
    def in_frame_basis(self) -> bool:
        """Whether to represent the model in the basis in which the frame operator is
        diagonalized.
        """
        return self._in_frame_basis

    @in_frame_basis.setter
    def in_frame_basis(self, in_frame_basis: bool):
        self._in_frame_basis = in_frame_basis

    @property
    def static_hamiltonian(self) -> ArrayLike:
        """The static Hamiltonian term."""
        if self._operator_collection.static_hamiltonian is None:
            return None

        if self.in_frame_basis:
            return self._operator_collection.static_hamiltonian
        return self.rotating_frame.operator_out_of_frame_basis(
            self._operator_collection.static_hamiltonian
        )

    @property
    def hamiltonian_operators(self) -> ArrayLike:
        """The Hamiltonian operators."""
        if self._operator_collection.hamiltonian_operators is None:
            return None

        if self.in_frame_basis:
            return self._operator_collection.hamiltonian_operators
        return self.rotating_frame.operator_out_of_frame_basis(
            self._operator_collection.hamiltonian_operators
        )

    @property
    def static_dissipators(self) -> ArrayLike:
        """The static dissipator operators."""
        if self._operator_collection.static_dissipators is None:
            return None

        if self.in_frame_basis:
            return self._operator_collection.static_dissipators
        return self.rotating_frame.operator_out_of_frame_basis(
            self._operator_collection.static_dissipators
        )

    @property
    def dissipator_operators(self) -> ArrayLike:
        """The dissipator operators."""
        if self._operator_collection.dissipator_operators is None:
            return None

        if self.in_frame_basis:
            return self._operator_collection.dissipator_operators
        return self.rotating_frame.operator_out_of_frame_basis(
            self._operator_collection.dissipator_operators
        )

    @property
    def array_library(self) -> Union[None, str]:
        """Array library used to store the operators in the model."""
        return self._array_library

    @property
    def vectorized(self) -> bool:
        """Whether or not the Lindblad equation is vectorized."""
        return self._vectorized

    @property
    def rotating_frame(self):
        """The rotating frame."""
        return self._rotating_frame

    def evaluate_hamiltonian(self, time: float) -> ArrayLike:
        """Evaluates Hamiltonian matrix at a given time.

        Args:
            time: The time at which to evaluate the Hamiltonian.

        Returns:
            Array: Hamiltonian matrix.
        """

        hamiltonian_sig_vals = None
        if self._hamiltonian_signals is not None:
            hamiltonian_sig_vals = self._hamiltonian_signals(time)

        ham = self._operator_collection.evaluate_hamiltonian(hamiltonian_sig_vals)
        if self.rotating_frame.frame_diag is not None:
            ham = self.rotating_frame.operator_into_frame(
                time,
                ham,
                operator_in_frame_basis=True,
                return_in_frame_basis=self._in_frame_basis,
                vectorized_operators=self.vectorized,
            )

        return ham

    def evaluate(self, time: float) -> ArrayLike:
        """Evaluate the model in array format as a matrix, independent of state.

        Args:
            time: The time to evaluate the model at.

        Returns:
            Array: The evaluated model as an anti-Hermitian matrix.

        Raises:
            QiskitError: If model cannot be evaluated.
            NotImplementedError: If the model is currently unvectorized.
        """
        hamiltonian_sig_vals = None
        if self._hamiltonian_signals is not None:
            hamiltonian_sig_vals = self._hamiltonian_signals(time)
        elif self._operator_collection.hamiltonian_operators is not None:
            raise QiskitError(
                f"{type(self).__name__} with non-empty hamiltonian operators cannot be evaluated "
                "without hamiltonian signals."
            )

        dissipator_sig_vals = None
        if self._dissipator_signals is not None:
            dissipator_sig_vals = self._dissipator_signals(time)
        elif self._operator_collection.dissipator_operators is not None:
            raise QiskitError(
                f"{type(self).__name__} with non-empty dissipator operators cannot be evaluated "
                "without dissipator signals."
            )

        if self.vectorized:
            out = self._operator_collection.evaluate(hamiltonian_sig_vals, dissipator_sig_vals)
            return self.rotating_frame.vectorized_map_into_frame(
                time, out, operator_in_frame_basis=True, return_in_frame_basis=self._in_frame_basis
            )
        else:
            raise NotImplementedError(
                "Non-vectorized Lindblad models cannot be represented without a given state."
            )

    def evaluate_rhs(self, time: float, y: ArrayLike) -> ArrayLike:
        """Evaluates the Lindblad model at a given time.

        Args:
            time: The time at which the model should be evaluated.
            y: Density matrix as an (n,n) Array if not using a vectorized evaluation_mode, or an
               (n^2) Array if using vectorized evaluation.

        Returns:
            Array: Either the evaluated generator or the state.

        Raises:
            QiskitError: If signals not sufficiently specified.
        """

        hamiltonian_sig_vals = None
        if self._hamiltonian_signals is not None:
            hamiltonian_sig_vals = self._hamiltonian_signals(time)
        elif self._operator_collection.hamiltonian_operators is not None:
            raise QiskitError(
                f"{type(self).__name__} with non-empty hamiltonian operators cannot be evaluated "
                "without hamiltonian signals."
            )

        dissipator_sig_vals = None
        if self._dissipator_signals is not None:
            dissipator_sig_vals = self._dissipator_signals(time)
        elif self._operator_collection.dissipator_operators is not None:
            raise QiskitError(
                f"{type(self).__name__} with non-empty dissipator operators cannot be evaluated "
                "without dissipator signals."
            )

        if self.rotating_frame.frame_diag is not None:
            # Take y out of the frame, but keep in the frame basis
            rhs = self.rotating_frame.operator_out_of_frame(
                time,
                y,
                operator_in_frame_basis=self._in_frame_basis,
                return_in_frame_basis=True,
                vectorized_operators=self.vectorized,
            )

            rhs = self._operator_collection.evaluate_rhs(
                hamiltonian_sig_vals, dissipator_sig_vals, rhs
            )

            # Put rhs back into the frame, potentially converting its basis.
            rhs = self.rotating_frame.operator_into_frame(
                time,
                rhs,
                operator_in_frame_basis=True,
                return_in_frame_basis=self._in_frame_basis,
                vectorized_operators=self.vectorized,
            )

        else:
            rhs = self._operator_collection.evaluate_rhs(
                hamiltonian_sig_vals, dissipator_sig_vals, y
            )

        return rhs


def _get_lindblad_operator_collection(
    array_library: Optional[str],
    vectorized: bool,
    static_hamiltonian: Optional[ArrayLike],
    hamiltonian_operators: Optional[ArrayLike],
    static_dissipators: Optional[ArrayLike],
    dissipator_operators: Optional[ArrayLike],
) -> Union[
    LindbladCollection,
    ScipySparseLindbladCollection,
    VectorizedLindbladCollection,
    ScipySparseVectorizedLindbladCollection,
]:
    """Construct a Lindblad operator collection.

    Args:
        array_library: Array library to use.
        vectorized: Whether or not to vectorize the Lindblad equation.
        static_hamiltonian: Constant part of the Hamiltonian.
        hamiltonian_operators: Operators in Hamiltonian with time-dependent coefficients.
        static_dissipators: Dissipation operators with coefficient 1.
        dissipator_operators: Dissipation operators with variable coefficients.

    Returns:
        Union[
            LindbladCollection,
            ScipySparseLindbladCollection,
            VectorizedLindbladCollection,
            ScipySparseVectorizedLindbladCollection,
        ]: Right-hand side evaluation object.
    """

    operator_kwargs = {
        "static_hamiltonian": static_hamiltonian,
        "hamiltonian_operators": hamiltonian_operators,
        "static_dissipators": static_dissipators,
        "dissipator_operators": dissipator_operators,
    }

    if array_library == "scipy_sparse":
        if vectorized:
            return ScipySparseVectorizedLindbladCollection(**operator_kwargs)

        return ScipySparseLindbladCollection(**operator_kwargs)

    if array_library == "jax_sparse":
        # warn that sparse mode when using JAX is primarily recommended for use on CPU
        if jax.default_backend() != "cpu":
            warn(
                """Using sparse mode with JAX is primarily recommended for use on CPU.""",
                stacklevel=2,
            )

    if vectorized:
        return VectorizedLindbladCollection(**operator_kwargs, array_library=array_library)

    return LindbladCollection(**operator_kwargs, array_library=array_library)
