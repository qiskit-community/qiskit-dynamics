##COPYRIGHT STUFF

"""Generic operator for general linear maps"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional, Callable
from copy import deepcopy
import numpy as np

from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.type_utils import to_array, vec_commutator, vec_dissipator


class BaseOperatorCollection(ABC):
    r"""BaseOperatorCollection is an abstract class
    intended to store a general set of linear mappings {\Lambda_i}
    in order to implement differential equations of the form
    \dot{y} = \Lambda(y,t). Generically, \Lambda will be a sum of
    other linear maps \Lambda_i(y,t), which are in turn some
    combination of left-multiplication \Lambda_i(y,t) = A(t)y(t),
    right-multiplication \Lambda_i(y,t) = y(t)B(t), and both, with
    \Lambda_i(y,t) = A(t)y(t)B(t), but this implementation
    will only engage with these at the level of \Lambda(y,t)

    Drift is a property that represents some time-independent
    component \Lambda_d of the decpmoosition, which will be
    used to facilitate frame transformations."""

    @property
    def drift(self) -> Array:
        """Returns drift part of operator collection."""
        return self._drift

    @drift.setter
    def drift(self, new_drift: Optional[Array] = None):
        """Sets Drift operator, if used."""
        self._drift = new_drift

    @property
    @abstractmethod
    def num_operators(self) -> int:
        """Returns number of operators the collection
        is storing."""
        pass

    @abstractmethod
    def evaluate_generator(self, signal_values: Array) -> Array:
        r"""If the model can be represented simply and
        without reference to the state involved, e.g.
        in the case \dot{y} = G(t)y(t) being represented
        as G(t), returns this independent representation.
        If the model cannot be represented in such a
        manner (c.f. Lindblad model), then errors."""
        pass

    @abstractmethod
    def evaluate_rhs(self, signal_values: Union[List[Array], Array], y: Array) -> Array:
        """Evaluates the model for a given state
        y provided the values of each signal
        component s_j(t). Must be defined for all
        models."""
        pass

    def __call__(
        self, signal_values: Union[List[Array], Array], y: Optional[Array] = None
    ) -> Array:
        """Evaluates the model given the values of the signal
        terms s_j, suppressing the choice between
        evaluate_rhs and evaluate_generator
        from the user. May error if y is not provided and
        model cannot be expressed without choice of state.
        """

        if y is None:
            return self.evaluate_generator(signal_values)

        return self.evaluate_rhs(signal_values, y)

    def copy(self):
        """Return a copy of self."""
        return deepcopy(self)


class DenseOperatorCollection(BaseOperatorCollection):
    r"""Meant to be a calculation object for models that
    only ever need left multiplication–those of the form
    \dot{y} = G(t)y(t), where G(t) = \sum_j s_j(t) G_j + G_d.
    Can evaluate G(t) independently of y.
    """

    @property
    def num_operators(self) -> int:
        return self._operators.shape[-3]

    def evaluate_generator(self, signal_values: Array) -> Array:
        r"""Evaluates the operator G at time t given
        the signal values s_j(t) as G(t) = \sum_j s_j(t)G_j"""
        if self._drift is None:
            return np.tensordot(signal_values, self._operators, axes=1)
        else:
            return np.tensordot(signal_values, self._operators, axes=1) + self._drift

    def evaluate_rhs(self, signal_values: Array, y: Array) -> Array:
        """Evaluates the product G(t)y"""
        return np.dot(self.evaluate_generator(signal_values), y)

    def __init__(self, operators: Array, drift: Optional[Array] = None):
        """Initialize
        Args:
            operators: (k,n,n) Array specifying the terms G_j
            drift: (n,n) Array specifying the extra drift G_d
        """
        self._operators = to_array(operators)
        self.drift = None
        self.drift = drift


class DenseLindbladCollection(BaseOperatorCollection):
    r"""Intended to be the calculation object for the Lindblad equation
    \dot{\rho} = -i[H,\rho] + \sum_j\gamma_j(t)
        (L_j\rho L_j^\dagger - (1/2) * {L_j^\daggerL_j,\rho})
    where [,] and {,} are the operator commutator and anticommutator, respectively.
    In the case that the Hamiltonian is also a function of time, varying as
    H(t) = H_d + \sum_j s_j(t) H_j, where H_d is the drift term,
    this can be further decomposed. We will allow for both our dissipator terms
    and our Hamiltonian terms to have different signal decompositions.
    """

    @property
    def dissipator_operators(self):
        return self._dissipator_operators

    @dissipator_operators.setter
    def dissipator_operators(self, dissipator_operators: Array):
        self._dissipator_operators = dissipator_operators
        if dissipator_operators is not None:
            self._dissipator_operators_conj = np.conjugate(
                np.transpose(dissipator_operators, [0, 2, 1])
            ).copy()
            self._dissipator_products = np.matmul(
                self._dissipator_operators_conj, self._dissipator_operators
            )

    @property
    def num_operators(self):
        return self._hamiltonian_operators.shape[-3], self._dissipator_operators.shape[-3]

    def __init__(
        self,
        hamiltonian_operators: Array,
        drift: Optional[Array],
        dissipator_operators: Optional[Array],
    ):
        r"""Converts an array of Hamiltonian components and signals,
        as well as Lindbladians, into a way of calculating the RHS
        of the Lindblad equation.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as H(t) = \sum_j s(t) H_j by specifying H_j. (k,n,n) array.
            drift: If supplied, treated as a constant term to be added to the
                Hamiltonian of the system.
            dissipator_operators: the terms L_j in Lindblad equation.
                (m,n,n) array.
        """

        self._hamiltonian_operators = hamiltonian_operators
        self.dissipator_operators = dissipator_operators
        self.drift = drift

    def evaluate_generator(self, signal_values: Array) -> Array:
        raise ValueError("Dense Lindblad collections cannot be evaluated without a state.")

    def evaluate_hamiltonian(self, signal_values: Array) -> Array:
        """Gets the Hamiltonian matrix, as calculated by the model,
        and used for the commutator -i[H,y]
        Args:
            signal_values: [Real] values of s_j in H = \sum_j s_j(t) H_j
        Returns:
            Hamiltonian matrix."""
        return np.tensordot(signal_values, self._hamiltonian_operators, axes=1) + self.drift

    def evaluate_rhs(self, signal_values: list[Array], y: Array) -> Array:
        r"""Evaluates Lindblad equation RHS given a pair of signal values
        for the hamiltonian terms and the dissipator terms. Expresses
        the RHS of the Lindblad equation as (A+B)y + y(A-B) + C, where
            A = (-1/2)*\sum_j\gamma(t) L_j^\dagger L_j
            B = -iH
            C = \sum_j \gamma_j(t) L_j y L_j^\dagger
        Args:
            signal_values: length-2 list of Arrays. has the following components
                signal_values[0]: hamiltonian signal values, s_j(t)
                    Must have length self._num_ham_terms
                signal_values[1]: dissipator signal values, \gamma_j(t)
                    Must have length self._num_dis_terms
            y: density matrix as (n,n) Array representing the state at time t
        Returns:
            RHS of Lindblad equation -i[H,y] + \sum_j\gamma_j(t)
            (L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y})
        """

        hamiltonian_matrix = -1j * self.evaluate_hamiltonian(signal_values[0])  # B matrix

        if self._dissipator_operators is not None:
            dissipators_matrix = (-1 / 2) * np.tensordot(  # A matrix
                signal_values[1], self._dissipator_products, axes=1
            )

            left_mult_contribution = np.matmul(hamiltonian_matrix + dissipators_matrix, y)
            right_mult_contribution = np.matmul(y, -hamiltonian_matrix + dissipators_matrix)

            if len(y.shape) == 3:
                # Must do array broadcasting and transposition to ensure vectorization works properly
                y = np.broadcast_to(y, (1, y.shape[0], y.shape[1], y.shape[2])).transpose(
                    [1, 0, 2, 3]
                )

            both_mult_contribution = np.tensordot(
                signal_values[1],
                np.matmul(
                    self._dissipator_operators, np.matmul(y, self._dissipator_operators_conj)
                ),
                axes=(-1, -3),
            )  # C

            return left_mult_contribution + right_mult_contribution + both_mult_contribution

        else:
            return np.dot(hamiltonian_matrix, y) - np.dot(y, hamiltonian_matrix)

class DenseVectorizedLindbladCollection(DenseOperatorCollection):
    """Intended as a calculation object for the Lindblad equation, 
    \dot{\rho} = -i[H,\rho] + \sum_j\gamma_j(t)
            (L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y})
    where all left-, right-, and left-and-right multiplication is 
    handled by vectorization, a process by which \rho, an (n,n)
    matrix, is embedded in a vector space of dimension n^2 using
    the column stacking convention, in which the matrix [a,b;c,d]
    is written as [a,c,b,d]."""

    def __init__(
        self,
        hamiltonian_operators: Array,
        drift: Array,
        dissipator_operators: Optional[Array] = None,
    ):
        r"""Converts an array of Hamiltonian components and signals,
        as well as Lindbladians, into a way of calculating the RHS
        of the Lindblad equation using only left-multiplication.

        Args:
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as H(t) = \sum_j s(t) H_j by specifying H_j. (k,n,n) array.
            drift: Constant term to be added to the Hamiltonian of the system.
            dissipator_operators: the terms L_j in Lindblad equation.
                (m,n,n) array.
        """

        #Convert Hamiltonian to commutator formalism
        self._hamiltonian_operators = hamiltonian_operators
        self._drift_terms = drift
        vec_drift = -1j*vec_commutator(drift)
            
        vec_ham_ops = -1j*vec_commutator(to_array(hamiltonian_operators))
        total_ops = None
        if dissipator_operators is not None:
            vec_diss_ops = vec_dissipator(to_array(dissipator_operators))
            total_ops = np.append(vec_ham_ops,vec_diss_ops,axis=0)
        else:
            total_ops = vec_ham_ops

        super().__init__(total_ops,drift=vec_drift)
    def evaluate_hamiltonian(self, signal_values: Array) -> Array:
        """Gets the Hamiltonian matrix, as calculated by the model,
        and used for the commutator -i[H,y]
        Args:
            signal_values: [Real] values of s_j in H = \sum_j s_j(t) H_j
        Returns:
            Hamiltonian matrix."""
        return (np.tensordot(signal_values, self._hamiltonian_operators, axes=1) + self._drift_terms).flatten(order="F")

    def evaluate_rhs(self, signal_values: Union[list[Array],Array], y: Array) -> Array:
        """Evaluates the RHS of the Lindblad equation using
        vectorized maps. 
        Args: 
            signal_values: either a list [ham_sig_values, dis_sig_values]
                storing the signal values for the Hamiltonian component
                and the dissipator component, or a single array containing
                the total list of signal values. 
            y: Density matrix represented as a vector using column-stacking
                convention.
        Returns: 
            Vectorized RHS of Lindblad equation \dot{\rho} in column-stacking
                convention."""
        if isinstance(signal_values,list):
            if signal_values[1] is not 0:
                signal_values = np.append(signal_values[0],signal_values[1],axis=-1)
            else:
                signal_values = signal_values[0]
        
        return super().evaluate_rhs(signal_values, y)
