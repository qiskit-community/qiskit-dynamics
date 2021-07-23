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

    @property
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

class DenseGeneralOperatorCollection(BaseOperatorCollection):
    """Most general form of dense-matrix stored operator collections. 
    Generically for models of the form \dot{y} = \Lambda(y,t) where
    \Lambda(y,t) = \sum_j s_j(t) \Lambda_j(y). We choose to support 
    linear maps of the form \Lambda_j(y) = A_jyB_j where A_j,B_j 
    are matrices. A special case is also carved out for when B_j = I
    \forall j. In this case, we pass to a LMultDenseOperatorCollection,
    which is intended for use with models of the form \dot{y} = G(t)y. 
    """

    @property
    def num_operators(self):
        return self._num_operators    

    def filter_signals(self,signals: SignalList):
            """To be called by Model objects to sort
            a SignalList of signals s_j into 
            the signal values associated to
            left-only, right-only, and left-and-right 
            multiplication in our linear maps \Lambda_j"""

            filtered_signals = [[],[],[]]
            
            actual_signal_array = np.array(signals.components()) #object array; temporary

            left_signals = SignalList(actual_signal_array[self._left_operator_filter].tolist())
            right_signals = SignalList(actual_signal_array[self._right_operator_filter].toarray())
            both_signals = SignalList(actual_signal_array[self._both_operator_filter].toarray())

            return left_signals,right_signals,both_signals

    def __init__(
        self,
        operators: Array
    ):
        """Initialization for DenseOperatorCollection. 

        Args: 
            operators: 
                Array of either shape (2,k,n,n) or (k,n,n). 
                In the second case, the constructor will assume that
                the operators are for left multiplication. In the first
                case, the (0,:,:,:) entries represent the A_j matrices, 
                and the (1,:,:,:) entries represent the B_j matrices. 
        """

        if operators.shape[-1]!=operators.shape[-2]:
                raise ValueError("Operators must be an array of square matrices")
        
        self._hilbert_space_dimension = operators.shape[-1]
        self._num_operators = operators.shape[-3]
        if len(operators.shape)==4:
            #More general case. Requires y be a matrix
            if operators.shape[0]!=2:
                raise ValueError("For simultaneous left- and right-multiplication support, operators first dimension must have 2 entries.")

            self._multiplication_mode = "general"

            # tracks whether operator (ij) where i = 0,1 and j associated to \Lambda_j is the identity map
            # i.e. is_identity[0,j] = True if A_j = I; likewise for [1,j] and B_j = I.
            is_identity = np.all(np.all(
                np.isclose(operators,np.eye(self._hilbert_space_dimension)),axis=3),axis=2)

        elif len(operators.shape)==3:
            #Assume that all operators are left-multiplying
            self._multiplication_mode = 'left-only'
            # array below is equivalent to finding that all right-multiplying operators are identity
            # row 1 is all True; row 2 is all False
            is_identity = np.array([[j for k in range(operators.shape[0])] for j in range(2)]).astype(bool)
        else:
            raise ValueError("operators must be either rank-3 or rank-4 arrays")
            
        self._left_operator_filter = is_identity[1] #consider both the case A_j != I, B_j = I and A_j=B_j=I as left multiplication
        self._right_operator_filter = np.logical_and(is_identity[0],np.logical_not(is_identity[1]))
        self._both_operator_filter = np.logical_and(np.logical_not(is_identity[0]),np.logical_not(is_identity[1]))

        self._operators = [[],[],[]]
        self._operators[0] = Array(operators[0][self._left_operator_filter])
        self._operators[1] = Array(operators[1][self._right_operator_filter])
        self._operators[2] = Array(np.array([operators[0][self._both_operator_filter],operators[1][self._both_operator_filter]]))

    def evaluate_with_state(self, signal_values: List[Array], y: Array):
        """Left, right, and left & right multiplication are handled differently 
        for efficiency purposes, then added together. This requires that signals
        be broken out into those which apply exclusively to left-multiplication,
        right-multiplication, and left-and-right multiplication. Sorting can be
        done at the DenseOperatorCollection level but *should* be done at the 
        Model level to maximize speed. Not aware of frames at all. 

        Args: 
            signal_values: length-3 list of Arrays of the actual signal values
                s_j(t) [will in general be complex]. 
            y: Array, either vector or matrix. Represents state of system. 
        """

        left_sig_vals,right_sig_vals,both_sig_vals = signal_values
        if self._multiplication_mode=="left-only":
            res = np.dot(np.tensordot(left_sig_vals,self._operators[0],axes=1),y)
        else:
            # allocate memory. Is there a better way to decide the dtype to allow for various
            # levels of numerical precision? Is there a potential bug here where y is 
            # initialized as a real matrix, but we will need it to accept complex values?
            res = np.zeros(tuple([self._hilbert_space_dimension]*len(y.shape)),dtype=y.dtype)
            
            if len(left_sig_vals)>0:
                res += np.dot(np.tensordot(left_sig_vals,self._operators[0],axes=1),y)
            if len(right_sig_vals)>0:
                res += np.dot(y,np.tensordot(right_sig_vals,self._operators[1],axes=1))
            if len(both_sig_vals)>0:
                res += np.tensordot(both_sig_vals,
                    np.matmul(np.matmul(self._operators[2][0],y),
                        self._operators[2][1]))
        return res
        
    def evaluate_without_state(self, signal_values: Array): 
        """Only compatible with left-multiplication systems. 
        As a result, only accepts a single array of signal values. 
        Args: 
            signal_values: Values for s_j(t)
        """
        if self._multiplication_mode!="left-only":
            raise NotImplementedError("Representing linear maps without state is not currently supported.")
        # Note that OperatorCollection is not aware of frames
        return np.tensordot(signal_values,self._operators[0],axes=1)
        
class DenseLindbladCollection(BaseOperatorCollection):
    """Intended to be the calculation object for the Lindblad equation
    \dot{\rho} = -i[H,\rho] + \sum_j\gamma_j(t) (L_j\rho L_j^\dagger - (1/2) * {L_j^\daggerL_j,\rho})
    where [,] and {,} are the operator commutator and anticommutator, respectively. 
    In the case that the Hamiltonian is also a function of time, varying as H(t) = \sum_j s_j(t) H_j, this
    can be further decomposed. We will allow for both our dissipator terms and our Hamiltonian terms to 
    have different signal decompositions. 
    """

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
        for the hamiltonian terms. 
        Args: 
            signal_values: length-2 list of Arrays. 
            components: 
                signal_values[0]: hamiltonian signal values, s_j(t)
                    Must have length self._num_ham_terms
                signal_values[1]: dissipator signal values, \gamma_j(t)
                    Must have length self._num_dis_terms
            y: density matrix [(n,n) array] representing the state at time t
        """
        hamiltonian = np.tensordot(signal_values[0],self._hamiltonian_operators)
        hamiltonian_contribution = -1j*(np.dot(hamiltonian,y)-np.dot(y,hamiltonian))
        dissipator_product_matrix = np.tensordot(signal_values[1],self._dissipator_products)
        # Note that \sum_j \gamma_j(t) (-1/2){L_j^\dagger L_j,\rho} = (-1/2){\sum_j\gamma_j L_j^\dagger L_j,\rho} 
        # by bilinearity of anticommutator
        dissipator_contribution = (-1/2)*(np.dot(dissipator_product_matrix,y)+np.dot(y,dissipator_product_matrix))
        dissipator_contribution+= np.tensordot(signal_values[1], np.matmul(np.matmul(
            self._dissipator_operators,y),self._dissipator_operators_conj))
        return dissipator_contribution + hamiltonian_contribution 



        
        
        



