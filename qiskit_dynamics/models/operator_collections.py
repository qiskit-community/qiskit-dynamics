##COPYRIGHT STUFF

"""Generic operator for general linear maps"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from copy import deepcopy
import numpy as np

from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.type_utils import to_array 
from qiskit_dynamics.signals import Signal,SignalList

class BaseOperatorCollection(ABC):
    r"""BaseOperatorCollection is an abstract class
    intended to store a general set of linear mappings {\Lambda_i}
    in order to implement differential equations of the form 
    \dot{y} = \Lambda(y,t). Generically, \Lambda will be a sum of 
    other linear maps \Lambda_i(y,t), which are in turn some
    combination of left-multiplication \Lambda_i(y,t) = A(t)y(t), 
    right-multiplication \Lambda_i(y,t) = y(t)B(t), and both, with 
    \Lambda_i(y,t) = A(t)y(t)B(t), but this implementation
    will only engage with these at the level of \Lambda(y,t)"""

    @abstractmethod
    def operators(self):
        """Get operators of the collection"""
        pass

    @abstractmethod
    def num_operators(self):
        """Get number of operators"""
        pass

    @abstractmethod
    def evaluate_without_state(self, signal_values: Array) -> Array:
        """If the model can be represented simply and
        without reference to the state involved, e.g. 
        in the case \dot{y} = G(t)y(t) being represented
        as G(t), returns this independent representation. 
        If the model cannot be represented in such a
        manner (c.f. Lindblad model), then this should 
        raise **some** sort of error. """
        pass

    @abstractmethod
    def evaluate_with_state(self, signal_values: Union[List[Array],Array], y: Array) -> Array:
        """Evaluates the model for a given state 
        y provided the values of each signal
        component s_j(t). Must be defined for all
        models. """
        pass

    def __call__(self, signal_values: Union[List[Array],Array], y: Optional[Array] = None):
        """Evaluates the model given the values of the signal
        terms s_j, suppressing the choice between
        evaluate_with_state and evaluate_without_state
        from the user. May error if y is not provided and
        model cannot be expressed without choice of state.
        """

        if y is None:
            return self.evaluate_without_state(signal_values)

        return self.evaluate_with_state(signal_values, y)

    def copy(self):
        """Return a copy of self."""
        return deepcopy(self)

class DenseOperatorCollection(BaseOperatorCollection): 
    """Meant to be a calculation object for models that 
    only ever need left multiplication–those of the form
    \dot{y} = G(t)y(t), where G(t) = \sum_j s_j(t) G_j. 
    Is able to evaluate G(t) independently of y.
    """

    def operators(self):
        return self._operators

    def num_operators(self):
        return self._num_operators
    
    def evaluate_without_state(self, signal_values: Array) -> Array:
        """Evaluates the operator G at time t given
        the signal values s_j(t) as G(t) = \sum_j s_j(t)G_j"""
        return np.tensordot(signal_values,self._operators)

    def evaluate_with_state(self, signal_values: Array, y: Array) -> Array:
        """Evaluates the product G(t)y"""
        return np.dot(self.evaluate_without_state(signal_values),y)

    def __init__(self, operators: Array):
        """Initialize an LMult model
        Args: 
            operators: (k,n,n) Array specifying the terms G_j
        """
        self._operators = operators
        self._num_operators = operators.shape[0]


class DenseLindbladCollection(BaseOperatorCollection):
    """Intended to be the calculation object for the Lindblad equation
    \dot{\rho} = -i[H,\rho] + \sum_j\gamma_j(t) (L_j\rho L_j^\dagger - (1/2) * {L_j^\daggerL_j,\rho})
    where [,] and {,} are the operator commutator and anticommutator, respectively. 
    def operators(self):
        return self._hamiltonian_operators,self._dissipator_operators

    def num_operators(self):
        return self._num_ham_terms,self._num_dis_terms
        

    def __init__(self,
        hamiltonian_operators: Array,
        dissipator_operators: Optional[Array],
    ):
        """Converts an array of Hamiltonian components and signals, 
        as well as Lindbladians, into a way of calculating the RHS
        of the Lindblad equation.

        Args: 
            hamiltonian_operators: Specifies breakdown of Hamiltonian 
                as H(t) = \sum_j s(t) H_j by specifying H_j. (k,n,n) array.
            dissipator_operators: the terms L_j in Lindblad equation. 
                (m,n,n) array. 
        """
        self._hamiltonian_operators = hamiltonian_operators
        self._num_ham_terms = hamiltonian_operators.shape[0]
        if dissipator_operators == None:
            self._num_dis_terms = 0
            self._dissipator_operators = Array([[]])
        else: 
            self._num_dis_terms = dissipator_operators.shape[0]

            self._dissipator_operators = dissipator_operators
            self._dissipator_operators_conj = np.conjugate(np.transpose(dissipator_operators,[0,2,1])).copy()
            self._dissipator_products = np.matmul(self._dissipator_operators_conj,self._dissipator_operators)
        

    def evaluate_with_state(self, signal_values: List[Array], y: Array):
        """Evaluates Lindblad equation RHS given a pair of signal values 
        for the hamiltonian terms and the dissipator terms. Expresses
        the RHS of the Lindblad equation 
        -i[H,y] + \sum_j\gamma_j(t) (L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y})
        as (A+B)y + y(A-B) + C, where 
            A = (-1/2)*\sum_j\gamma(t) L_j^\dagger L_j
            B = -iH
            C = \sum_j \gamma_j(t) L_j y L_j^\dagger
        Args: 
            signal_values: length-2 list of Arrays. 
            components: 
                signal_values[0]: hamiltonian signal values, s_j(t)
                    Must have length self._num_ham_terms
                signal_values[1]: dissipator signal values, \gamma_j(t)
                    Must have length self._num_dis_terms
            y: density matrix [(n,n) array] representing the state at time t
        """
        dissipators_matrix = (-1/2)*np.tensordot(signal_values[1],self._dissipator_products) #A
        hamiltonian_matrix = -1j*np.tensordot(signal_values[0],self._hamiltonian_operators) #B

        left_mult_contribution = np.dot(hamiltonian_matrix+dissipators_matrix,y)
        right_mult_contribution = np.dot(y,-hamiltonian_matrix+dissipators_matrix)
        both_mult_contribution = np.tensordot(signal_values[1], np.matmul(np.matmul(
            self._dissipator_operators,y),self._dissipator_operators_conj)) #C
        return left_mult_contribution + right_mult_contribution + both_mult_contribution
