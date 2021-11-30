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

"""Operator collections as math/calculation objects for Model classes"""

from abc import ABC, abstractmethod
from typing import Union, List, Optional
from copy import copy
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.csr import csr_matrix

from qiskit import QiskitError
from qiskit.quantum_info.operators.operator import Operator
from qiskit_dynamics.array import Array, wrap
from qiskit_dynamics.type_utils import to_array, to_csr, to_BCOO, vec_commutator, vec_dissipator

try:
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    # sparse versions of jax.numpy operations
    jsparse_sum = jsparse.sparsify(jnp.sum)
    jsparse_matmul = jsparse.sparsify(jnp.matmul)
    jsparse_add = jsparse.sparsify(jnp.add)
    jsparse_subtract = jsparse.sparsify(jnp.subtract)

    def jsparse_linear_combo(coeffs, mats):
        """Method for computing a linear combination of sparse arrays."""
        return jsparse_sum(jnp.broadcast_to(coeffs[:, None, None], mats.shape) * mats, axis=0)

    # sparse version of computing A @ X @ B
    jsparse_triple_product = jsparse.sparsify(lambda A, X, B: A @ X @ B)

except ImportError:
    pass


class BaseOperatorCollection(ABC):
    r"""Abstract class representing a two-variable matrix function.

    This class represents a function :math:`c,y \mapsto \Lambda(c, y)`,
    which is assumed to be decomposed as
    :math:`\Lambda(c, y) = (G_d + \sum_jc_jG_j)  y`
    for matrices :math:`G_d` and :math:`G_j`, with
    :math:`G_d` referred to as the static operator.

    Describes an interface for evaluating the map or its action on ``y``,
    given the 1d set of values :math:`c_j`.
    """

    def __init__(
        self,
        static_operator: Optional[any] = None,
        operators: Optional[any] = None,
    ):
        """Initialize.

        Accepted types are determined by concrete subclasses.

        Args:
            operators: (k,n,n) Array specifying the terms :math:`G_j`.
            static_operator: (n,n) Array specifying the extra static_operator :math:`G_d`.
        """
        self.operators = operators
        self.static_operator = static_operator

    @property
    def static_operator(self) -> Array:
        """Returns static part of operator collection."""

    @static_operator.setter
    def static_operator(self, new_static_operator: Optional[Array] = None):
        """Sets static_operator term."""

    @property
    def operators(self) -> Array:
        """Return operators."""

    @operators.setter
    def operators(self, new_operators: Array) -> Array:
        """Return operators."""

    @abstractmethod
    def evaluate(self, signal_values: Array) -> Array:
        r"""Evaluate the map."""

    @abstractmethod
    def evaluate_rhs(self, signal_values: Union[List[Array], Array], y: Array) -> Array:
        r"""Compute the function."""

    def __call__(
        self, signal_values: Union[List[Array], Array], y: Optional[Array] = None
    ) -> Array:
        """Call either ``self.evaluate`` or ``self.evaluate_rhs`` depending on number of
        arguments.
        """

        if y is None:
            return self.evaluate(signal_values)

        return self.evaluate_rhs(signal_values, y)

    def copy(self):
        """Return a copy of self."""
        return copy(self)


class DenseOperatorCollection(BaseOperatorCollection):
    r"""Concrete operator collection representing a function computing left
    multiplication by an affine combination of matrices.

    Concrete instance of ``BaseOperatorCollection`` in which
    :math:`G_d` and :math:`G_j` are dense arrays.
    """

    @property
    def static_operator(self) -> Array:
        """Returns static part of operator collection."""
        return self._static_operator

    @static_operator.setter
    def static_operator(self, new_static_operator: Array):
        """Sets static_operator term."""
        self._static_operator = to_array(new_static_operator)

    @property
    def operators(self) -> Array:
        """Operators in the collection."""
        return self._operators

    @operators.setter
    def operators(self, new_operators: Array):
        self._operators = to_array(new_operators)

    def evaluate(self, signal_values: Union[Array, None]) -> Array:
        r"""Evaluate the affine combination of matrices.

        Returns:
            Evaluated model.
        Raises:
            QiskitError: if both static_operator and operators are None
        """
        if self._static_operator is not None and self._operators is not None:
            return np.tensordot(signal_values, self._operators, axes=1) + self._static_operator
        elif self._static_operator is None and self._operators is not None:
            return np.tensordot(signal_values, self._operators, axes=1)
        elif self._static_operator is not None:
            return self._static_operator
        else:
            raise QiskitError(
                self.__class__.__name__
                + """ with None for both static_operator and
                                operators cannot be evaluated."""
            )

    def evaluate_rhs(self, signal_values: Union[Array, None], y: Array) -> Array:
        """Evaluates the function."""
        return np.dot(self.evaluate(signal_values), y)


class SparseOperatorCollection(BaseOperatorCollection):
    r"""Sparse version of DenseOperatorCollection."""

    def __init__(
        self,
        static_operator: Optional[Union[Array, Operator]] = None,
        operators: Optional[Union[Array, List[Operator]]] = None,
        decimals: Optional[int] = 10,
    ):
        """Initialize.

        Args:
            static_operator: (n,n) Array specifying the static_operator term :math:`G_d`.
            operators: (k,n,n) Array specifying the terms :math:`G_j`.
            decimals: Values will be rounded at ``decimals`` places after decimal.
        """
        self._decimals = decimals
        super().__init__(static_operator=static_operator, operators=operators)

    @property
    def static_operator(self) -> csr_matrix:
        return self._static_operator

    @static_operator.setter
    def static_operator(self, new_static_operator: csr_matrix):
        if new_static_operator is not None:
            self._static_operator = np.round(to_csr(new_static_operator), self._decimals)
        else:
            self._static_operator = None

    @property
    def operators(self) -> List[csr_matrix]:
        if self._operators is None:
            return None

        return list(self._operators)

    @operators.setter
    def operators(self, new_operators: List[csr_matrix]):
        if new_operators is not None:
            new_operators_to_csr = to_csr(list(new_operators))
            new_operators = np.empty(shape=len(new_operators_to_csr), dtype="O")
            for idx, new_op in enumerate(new_operators_to_csr):
                new_operators[idx] = csr_matrix(np.round(new_op, self._decimals))

        self._operators = new_operators

    def evaluate(self, signal_values: Union[Array, None]) -> csr_matrix:
        r"""Sparse version of ``DenseOperatorCollection.evaluate``.

        Args:
            signal_values: Coefficients :math:`c_j`.

        Returns:
            Generator as sparse array.

        Raises:
            QiskitError: If collection cannot be evaluated.
        """
        if self._static_operator is not None and self._operators is not None:
            return (
                np.tensordot(signal_values, self._operators, axes=1).item() + self._static_operator
            )
        elif self._static_operator is None and self._operators is not None:
            return np.tensordot(signal_values, self._operators, axes=1).item()
        elif self.static_operator is not None:
            return self._static_operator
        raise QiskitError(
            self.__class__.__name__
            + """ with None for both static_operator and
                            operators cannot be evaluated."""
        )

    def evaluate_rhs(self, signal_values: Union[Array, None], y: Array) -> Array:
        if len(y.shape) == 2:
            # For 2d array, compute linear combination then multiply
            gen = self.evaluate(signal_values)
            return gen.dot(y)
        elif len(y.shape) == 1:
            # For a 1d array, multiply individual matrices then compute linear combination
            tmparr = np.empty(shape=(1), dtype="O")
            tmparr[0] = y

            if self._static_operator is not None and self._operators is not None:
                return np.dot(signal_values, self._operators * tmparr) + self.static_operator.dot(y)
            elif self._static_operator is None and self._operators is not None:
                return np.dot(signal_values, self._operators * tmparr)
            elif self.static_operator is not None:
                return self.static_operator.dot(y)

            raise QiskitError(
                self.__class__.__name__
                + """  with None for both static_operator and
                                operators cannot be evaluated."""
            )

        raise QiskitError(self.__class__.__name__ + """  cannot evaluate RHS for y.ndim > 3.""")


class JAXSparseOperatorCollection(BaseOperatorCollection):
    """Jax version of SparseOperatorCollection built on jax.experimental.sparse.BCOO."""

    @property
    def static_operator(self) -> "BCOO":
        return self._static_operator

    @static_operator.setter
    def static_operator(self, new_static_operator: Union["BCOO", None]):
        self._static_operator = to_BCOO(new_static_operator)

    @property
    def operators(self) -> Union["BCOO", None]:
        return self._operators

    @operators.setter
    def operators(self, new_operators: Union["BCOO", None]):
        self._operators = to_BCOO(new_operators)

    def evaluate(self, signal_values: Union[Array, None]) -> "BCOO":
        r"""Jax sparse version of ``DenseOperatorCollection.evaluate``.

        Args:
            signal_values: Coefficients :math:`c_j`.

        Returns:
            Generator as sparse jax array.

        Raises:
            QiskitError: If collection cannot be evaluated.
        """
        if signal_values is not None and isinstance(signal_values, Array):
            signal_values = signal_values.data

        if self._static_operator is not None and self._operators is not None:
            return jsparse_linear_combo(signal_values, self._operators) + self._static_operator
        elif self._static_operator is None and self._operators is not None:
            return jsparse_linear_combo(signal_values, self._operators)
        elif self.static_operator is not None:
            return self._static_operator
        raise QiskitError(
            self.__class__.__name__
            + """ with None for both static_operator and
                            operators cannot be evaluated."""
        )

    def evaluate_rhs(self, signal_values: Union[Array, None], y: Array) -> Array:
        if y.ndim < 3:
            if isinstance(y, Array):
                y = y.data
            return Array(jsparse_matmul(self.evaluate(signal_values), y))

        raise QiskitError(self.__class__.__name__ + """  cannot evaluate RHS for y.ndim >= 3.""")


class BaseLindbladOperatorCollection(ABC):
    r"""Abstract class representing a two-variable matrix function for evaluating
    the right hand side of the Lindblad equation.

    In particular, this object represents the function:
        .. math::
            \Lambda(c_1, c_2, \rho) = -i[H_d + \sum_j c_{1,j}H_j,\rho]
                                        + \sum_j(D_j\rho D_j^\dagger
                                          - (1/2) * {D_j^\daggerD_j,\rho})
                                      + \sum_jc_{2,j}(L_j\rho L_j^\dagger
                                        - (1/2) * {L_j^\daggerL_j,\rho})

    where :math:`\[,\]` and :math:`\{,\}` are the operator
    commutator and anticommutator, respectively.

    Describes an interface for evaluating the map or its action on :math:`\rho`,
    given a pair of 1d sets of values :math:`c_1, c_2`.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[any] = None,
        hamiltonian_operators: Optional[any] = None,
        static_dissipators: Optional[any] = None,
        dissipator_operators: Optional[any] = None,
    ):
        r"""Initialize collection. Argument types depend on concrete subclass.

        Args:
            static_hamiltonian: Constant term :math:`H_d` to be added to the Hamiltonian of the
                                system.
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) array.
            static_dissipators: Constant dissipator terms.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) array.
        """
        self.static_hamiltonian = static_hamiltonian
        self.hamiltonian_operators = hamiltonian_operators
        self.static_dissipators = static_dissipators
        self.dissipator_operators = dissipator_operators

    @property
    @abstractmethod
    def static_hamiltonian(self) -> Array:
        """Returns static part of the hamiltonian."""

    @static_hamiltonian.setter
    @abstractmethod
    def static_hamiltonian(self, new_static_operator: Optional[Array] = None):
        """Sets static_operator term."""

    @property
    @abstractmethod
    def hamiltonian_operators(self) -> Array:
        """Returns operators for non-static part of Hamiltonian."""

    @hamiltonian_operators.setter
    @abstractmethod
    def hamiltonian_operators(self, new_hamiltonian_operators: Optional[Array] = None):
        """Set operators for non-static part of Hamiltonian."""

    @property
    @abstractmethod
    def static_dissipators(self) -> Array:
        """Returns operators for static part of dissipator."""

    @static_dissipators.setter
    @abstractmethod
    def static_dissipators(self, new_static_dissipators: Optional[Array] = None):
        """Sets operators for static part of dissipator."""

    @property
    @abstractmethod
    def dissipator_operators(self) -> Array:
        """Returns operators for non-static part of dissipator."""

    @dissipator_operators.setter
    @abstractmethod
    def dissipator_operators(self, new_dissipator_operators: Optional[Array] = None):
        """Sets operators for non-static part of dissipator."""

    @abstractmethod
    def evaluate_hamiltonian(self, ham_sig_vals: Union[None, Array]) -> Union[csr_matrix, Array]:
        """Evaluate the Hamiltonian of the model."""

    @abstractmethod
    def evaluate(
        self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array]
    ) -> Union[csr_matrix, Array]:
        r"""Evaluate the map."""

    @abstractmethod
    def evaluate_rhs(
        self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array], y: Array
    ) -> Array:
        r"""Compute the function."""

    def __call__(
        self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array], y: Optional[Array]
    ) -> Union[csr_matrix, Array]:
        """Evaluate the model, or evaluate the RHS."""
        if y is None:
            return self.evaluate(ham_sig_vals, dis_sig_vals)

        return self.evaluate_rhs(ham_sig_vals, dis_sig_vals, y)

    def copy(self):
        """Return a copy of self."""
        return copy(self)


class DenseLindbladCollection(BaseLindbladOperatorCollection):
    r"""Object for computing the right hand side of the Lindblad equation
    with dense arrays.
    """

    @property
    def static_hamiltonian(self) -> Array:
        return self._static_hamiltonian

    @static_hamiltonian.setter
    def static_hamiltonian(self, new_static_hamiltonian: Optional[Array] = None):
        self._static_hamiltonian = to_array(new_static_hamiltonian)

    @property
    def hamiltonian_operators(self) -> Array:
        return self._hamiltonian_operators

    @hamiltonian_operators.setter
    def hamiltonian_operators(self, new_hamiltonian_operators: Optional[Array] = None):
        self._hamiltonian_operators = to_array(new_hamiltonian_operators)

    @property
    def static_dissipators(self) -> Array:
        return self._static_dissipators

    @static_dissipators.setter
    def static_dissipators(self, new_static_dissipators: Optional[Array] = None):
        self._static_dissipators = to_array(new_static_dissipators)
        if self._static_dissipators is not None:
            self._static_dissipators_adj = np.conjugate(
                np.transpose(self._static_dissipators, [0, 2, 1])
            ).copy()
            self._static_dissipators_product_sum = -0.5 * np.sum(
                np.matmul(self._static_dissipators_adj, self._static_dissipators), axis=0
            )

    @property
    def dissipator_operators(self) -> Array:
        return self._dissipator_operators

    @dissipator_operators.setter
    def dissipator_operators(self, new_dissipator_operators: Optional[Array] = None):
        self._dissipator_operators = to_array(new_dissipator_operators)
        if self._dissipator_operators is not None:
            self._dissipator_operators_adj = np.conjugate(
                np.transpose(self._dissipator_operators, [0, 2, 1])
            ).copy()
            self._dissipator_products = np.matmul(
                self._dissipator_operators_adj, self._dissipator_operators
            )

    def evaluate(self, ham_sig_vals: Array, dis_sig_vals: Array) -> Array:
        raise ValueError("Non-vectorized Lindblad collections cannot be evaluated without a state.")

    def evaluate_hamiltonian(self, ham_sig_vals: Union[None, Array]) -> Array:
        r"""Compute the Hamiltonian.

        Args:
            ham_sig_vals: [Real] values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.
        Returns:
            Hamiltonian matrix.
        Raises:
            QiskitError: If collection not sufficiently specified.
        """
        if self._static_hamiltonian is not None and self._hamiltonian_operators is not None:
            return (
                np.tensordot(ham_sig_vals, self._hamiltonian_operators, axes=1)
                + self._static_hamiltonian
            )
        elif self._static_hamiltonian is None and self._hamiltonian_operators is not None:
            return np.tensordot(ham_sig_vals, self._hamiltonian_operators, axes=1)
        elif self._static_hamiltonian is not None:
            return self._static_hamiltonian
        else:
            raise QiskitError(
                self.__class__.__name__
                + """ with None for both static_hamiltonian and
                                hamiltonian_operators cannot evaluate Hamiltonian."""
            )

    def evaluate_rhs(
        self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array], y: Array
    ) -> Array:
        r"""Evaluates Lindblad equation RHS given a pair of signal values
        for the hamiltonian terms and the dissipator terms. Expresses
        the RHS of the Lindblad equation as :math:`(A+B)y + y(A-B) + C`, where
            .. math::
            A = (-1/2)*\sum_jD_j^\dagger D_j + (-1/2)*\sum_j\gamma_j(t) L_j^\dagger L_j,

            B = -iH,

            C = \sum_j \gamma_j(t) L_j y L_j^\dagger.

        Args:
            ham_sig_vals: hamiltonian coefficient values, :math:`s_j(t)`.
            dis_sig_vals: dissipator signal values, :math:`\gamma_j(t)`.
            y: density matrix as (n,n) Array representing the state at time :math:`t`.
        Returns:
            RHS of Lindblad equation
            .. math::
                -i[H,y] + \sum_j\gamma_j(t)(L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y}).
        Raises:
            QiskitError: If operator collection is underspecified.
        """

        hamiltonian_matrix = None
        if self._static_hamiltonian is not None or self._hamiltonian_operators is not None:
            hamiltonian_matrix = -1j * self.evaluate_hamiltonian(ham_sig_vals)  # B matrix

        # if dissipators present (includes both hamiltonian is None and is not None)
        if self._dissipator_operators is not None or self._static_dissipators is not None:

            # A matrix
            if self._static_dissipators is None:
                dissipators_matrix = np.tensordot(
                    -0.5 * dis_sig_vals, self._dissipator_products, axes=1
                )
            elif self._dissipator_operators is None:
                dissipators_matrix = self._static_dissipators_product_sum
            else:
                dissipators_matrix = self._static_dissipators_product_sum + np.tensordot(
                    -0.5 * dis_sig_vals, self._dissipator_products, axes=1
                )

            if hamiltonian_matrix is not None:
                left_mult_contribution = np.matmul(hamiltonian_matrix + dissipators_matrix, y)
                right_mult_contribution = np.matmul(y, dissipators_matrix - hamiltonian_matrix)
            else:
                left_mult_contribution = np.matmul(dissipators_matrix, y)
                right_mult_contribution = np.matmul(y, dissipators_matrix)

            if len(y.shape) == 3:
                # Must do array broadcasting and transposition to ensure vectorization works
                y = np.broadcast_to(y, (1, y.shape[0], y.shape[1], y.shape[2])).transpose(
                    [1, 0, 2, 3]
                )

            if self._static_dissipators is None:
                both_mult_contribution = np.tensordot(
                    dis_sig_vals,
                    np.matmul(
                        self._dissipator_operators, np.matmul(y, self._dissipator_operators_adj)
                    ),
                    axes=(-1, -3),
                )
            elif self._dissipator_operators is None:
                both_mult_contribution = np.sum(
                    np.matmul(self._static_dissipators, np.matmul(y, self._static_dissipators_adj)),
                    axis=-3,
                )
            else:
                both_mult_contribution = np.sum(
                    np.matmul(self._static_dissipators, np.matmul(y, self._static_dissipators_adj)),
                    axis=-3,
                ) + np.tensordot(
                    dis_sig_vals,
                    np.matmul(
                        self._dissipator_operators, np.matmul(y, self._dissipator_operators_adj)
                    ),
                    axes=(-1, -3),
                )

            return left_mult_contribution + right_mult_contribution + both_mult_contribution
        # if just hamiltonian
        elif hamiltonian_matrix is not None:
            return np.dot(hamiltonian_matrix, y) - np.dot(y, hamiltonian_matrix)
        else:
            raise QiskitError(
                """DenseLindbladCollection with None for static_hamiltonian,
                                 hamiltonian_operators, static_dissipators, and
                                 dissipator_operators, cannot evaluate rhs."""
            )


class SparseLindbladCollection(DenseLindbladCollection):
    """Sparse version of DenseLindbladCollection."""

    def __init__(
        self,
        static_hamiltonian: Optional[Union[csr_matrix, Operator]] = None,
        hamiltonian_operators: Optional[Union[List[csr_matrix], List[Operator]]] = None,
        static_dissipators: Optional[Union[List[csr_matrix], List[Operator]]] = None,
        dissipator_operators: Optional[Union[List[csr_matrix], List[Operator]]] = None,
        decimals: Optional[int] = 10,
    ):
        r"""Initializes sparse lindblad collection.

        Args:
            static_hamiltonian: Constant term :math:`H_d` to be added to the Hamiltonian of the
                                system.
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) array.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) array.
            decimals: operator values will be rounded to ``decimals`` places after the
                decimal place to avoid excess storage of near-zero values
                in sparse format.
        """
        self._decimals = decimals
        super().__init__(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
        )

    @property
    def static_hamiltonian(self) -> csr_matrix:
        return self._static_hamiltonian

    @static_hamiltonian.setter
    def static_hamiltonian(self, new_static_hamiltonian: Optional[csr_matrix] = None):
        if new_static_hamiltonian is not None:
            new_static_hamiltonian = np.round(
                to_csr(new_static_hamiltonian), decimals=self._decimals
            )
        self._static_hamiltonian = new_static_hamiltonian

    @property
    def hamiltonian_operators(self) -> np.ndarray:
        if self._hamiltonian_operators is None:
            return None

        return list(self._hamiltonian_operators)

    @hamiltonian_operators.setter
    def hamiltonian_operators(self, new_hamiltonian_operators: Optional[List[csr_matrix]] = None):
        if new_hamiltonian_operators is not None:
            new_hamiltonian_operators = to_csr(new_hamiltonian_operators)
            new_hamiltonian_operators = [
                np.round(op, decimals=self._decimals) for op in new_hamiltonian_operators
            ]
            new_hamiltonian_operators = np.array(new_hamiltonian_operators, dtype="O")

        self._hamiltonian_operators = new_hamiltonian_operators

    @property
    def static_dissipators(self) -> Union[None, csr_matrix]:
        if self._static_dissipators is None:
            return None

        return list(self._static_dissipators)

    @static_dissipators.setter
    def static_dissipators(self, new_static_dissipators: Optional[List[csr_matrix]] = None):
        """Set up the dissipators themselves, as well as their adjoints, and the product of
        adjoint with operator.
        """
        self._static_dissipators = None
        if new_static_dissipators is not None:
            # setup new dissipators
            new_static_dissipators = to_csr(new_static_dissipators)
            new_static_dissipators = [
                np.round(op, decimals=self._decimals) for op in new_static_dissipators
            ]

            # setup adjoints
            static_dissipators_adj = [op.conj().transpose() for op in new_static_dissipators]

            # wrap in object arrays
            new_static_dissipators = np.array(new_static_dissipators, dtype="O")
            static_dissipators_adj = np.array(static_dissipators_adj, dtype="O")

            # pre-compute products
            static_dissipators_product_sum = -0.5 * np.sum(
                static_dissipators_adj * new_static_dissipators, axis=0
            )

            self._static_dissipators = new_static_dissipators
            self._static_dissipators_adj = static_dissipators_adj
            self._static_dissipators_product_sum = static_dissipators_product_sum

    @property
    def dissipator_operators(self) -> Union[None, List[csr_matrix]]:
        if self._dissipator_operators is None:
            return None

        return list(self._dissipator_operators)

    @dissipator_operators.setter
    def dissipator_operators(self, new_dissipator_operators: Optional[List[csr_matrix]] = None):
        """Set up the dissipators themselves, as well as their adjoints, and the product of
        adjoint with operator.
        """
        self._dissipator_operators = None
        if new_dissipator_operators is not None:
            # setup new dissipators
            new_dissipator_operators = to_csr(new_dissipator_operators)
            new_dissipator_operators = [
                np.round(op, decimals=self._decimals) for op in new_dissipator_operators
            ]

            # setup adjoints
            dissipator_operators_adj = [op.conj().transpose() for op in new_dissipator_operators]

            # wrap in object arrays
            new_dissipator_operators = np.array(new_dissipator_operators, dtype="O")
            dissipator_operators_adj = np.array(dissipator_operators_adj, dtype="O")

            # pre-compute projducts
            dissipator_products = dissipator_operators_adj * new_dissipator_operators

            self._dissipator_operators = new_dissipator_operators
            self._dissipator_operators_adj = dissipator_operators_adj
            self._dissipator_products = dissipator_products

    def evaluate_hamiltonian(self, ham_sig_vals: Union[None, Array]) -> csr_matrix:
        r"""Compute the Hamiltonian.

        Args:
            ham_sig_vals: [Real] values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.
        Returns:
            Hamiltonian matrix.
        Raises:
            QiskitError: If collection not sufficiently specified.
        """
        if self._static_hamiltonian is not None and self._hamiltonian_operators is not None:
            return (
                np.sum(ham_sig_vals * self._hamiltonian_operators, axis=-1)
                + self.static_hamiltonian
            )
        elif self._static_hamiltonian is None and self._hamiltonian_operators is not None:
            return np.sum(ham_sig_vals * self._hamiltonian_operators, axis=-1)
        elif self._static_hamiltonian is not None:
            return self._static_hamiltonian
        else:
            raise QiskitError(
                self.__class__.__name__
                + """ with None for both static_hamiltonian and
                                hamiltonian_operators cannot evaluate Hamiltonian."""
            )

    def evaluate_rhs(
        self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array], y: Array
    ) -> Array:
        r"""Evaluates the RHS of the LindbladModel for a given list of signal values.

        Args:
            ham_sig_vals: stores Hamiltonian signal values :math:`s_j(t)`.
            dis_sig_vals: stores dissipator signal values :math:`\gamma_j(t)`.
                Pass None if no dissipator operators involved.
            y: density matrix of system. (k,n,n) Array.
        Returns:
            RHS of Lindbladian.
        Raises:
            QiskitError: If RHS cannot be evaluated due to insufficient collection data.

        Calculation details:
            * for csr_matrices is equivalent to matrix multiplicaiton.
            We use numpy array broadcasting rules, combined with the above
            fact, to achieve speeds that are substantially faster than a for loop.
            First, in the case of a single (n,n) density matrix, we package the entire
            array as a single-element array whose entry is the array. In the case of
            multiple density matrices a (k,n,n) Array, we package everything as a
            (k,1) Array whose [j,0] entry is the [j,:,:] density matrix.

            In calculating the left- and right-mult contributions, we package
            H+L and H-L as (1) object arrays whose single entry stores the relevant
            sparse matrix. We can then multiply our packaged density matrix and
            [H\pm L]. Using numpy broadcasting rules, [H\pm L] will be broadcast
            to a (k,1) Array for elementwise multiplication with our packaged density
            matrices. After this, elementwise multiplication is applied. This in turn
            references each object's __mul__ function, which–for our csr_matrix components
            means matrix multiplication.

            In calculating the left-right-multiplication part, we use our (m)-shape
            object arrays holding the dissipator operators to perform multiplication.
            We can take an elementwise product with our packaged density matrix, at which
            point our dissipator operators are broadcast as (m) -> (1,m) -> (k,m) shaped,
            and our packaged density matrix as (k,1) -> (k,m). Elementwise multiplication
            is then applied, which is interpreted as matrix multiplication. This yields
            an array where entry [i,j] is an object storing the results of s_jL_j\rho_i L_j^\dagger.
            We can then sum over j and unpackage our object array to get our desired result.
        """
        hamiltonian_matrix = None
        if self._static_hamiltonian is not None or self._hamiltonian_operators is not None:
            hamiltonian_matrix = -1j * self.evaluate_hamiltonian(ham_sig_vals)  # B matrix

        # package (n,n) Arrays as (1)
        # Arrays of dtype object, or (k,n,n) Arrays as (k,1) Arrays of dtype object
        y = package_density_matrices(y)

        # if dissipators present (includes both hamiltonian is None and is not None)
        if self._dissipator_operators is not None or self._static_dissipators is not None:

            # A matrix
            if self._static_dissipators is None:
                dissipators_matrix = np.sum(
                    -0.5 * dis_sig_vals * self._dissipator_products, axis=-1
                )
            elif self._dissipator_operators is None:
                dissipators_matrix = self._static_dissipators_product_sum
            else:
                dissipators_matrix = self._static_dissipators_product_sum + np.sum(
                    -0.5 * dis_sig_vals * self._dissipator_products, axis=-1
                )

            if hamiltonian_matrix is not None:
                left_mult_contribution = np.squeeze([hamiltonian_matrix + dissipators_matrix] * y)
                right_mult_contribution = np.squeeze(y * [dissipators_matrix - hamiltonian_matrix])
            else:
                left_mult_contribution = np.squeeze([dissipators_matrix] * y)
                right_mult_contribution = np.squeeze(y * [dissipators_matrix])

            # both_mult_contribution[i] = \gamma_i L_i\rho L_i^\dagger performed in array language
            if self._static_dissipators is None:
                both_mult_contribution = np.sum(
                    (dis_sig_vals * self._dissipator_operators)
                    * y
                    * self._dissipator_operators_adj,
                    axis=-1,
                )
            elif self._dissipator_operators is None:
                both_mult_contribution = np.sum(
                    self._static_dissipators * y * self._static_dissipators_adj, axis=-1
                )
            else:
                both_mult_contribution = (
                    np.sum(
                        (dis_sig_vals * self._dissipator_operators)
                        * y
                        * self._dissipator_operators_adj,
                        axis=-1,
                    )
                    + np.sum(self._static_dissipators * y * self._static_dissipators_adj, axis=-1)
                )

            out = left_mult_contribution + right_mult_contribution + both_mult_contribution

        elif hamiltonian_matrix is not None:
            out = (([hamiltonian_matrix] * y) - (y * [hamiltonian_matrix]))[0]
        else:
            raise QiskitError(
                "SparseLindbladCollection with None for static_hamiltonian, "
                "hamiltonian_operators, and dissipator_operators, cannot evaluate rhs."
            )
        if len(y.shape) == 2:
            # Very slow; avoid if not necessary (or if better implementation found). Needs to
            # map a (k) Array of dtype object with j^{th} entry a (n,n) Array -> (k,n,n) Array.
            out = unpackage_density_matrices(out.reshape(y.shape[0], 1))

        return out


class JAXSparseLindbladCollection(BaseLindbladOperatorCollection):
    r"""Object for computing the right hand side of the Lindblad equation
    with using jax.experimental.sparse.BCOO arrays.
    """

    @property
    def static_hamiltonian(self) -> "BCOO":
        return self._static_hamiltonian

    @static_hamiltonian.setter
    def static_hamiltonian(self, new_static_hamiltonian: Union["BCOO", None]):
        self._static_hamiltonian = to_BCOO(new_static_hamiltonian)

    @property
    def hamiltonian_operators(self) -> Union["BCOO", None]:
        return self._hamiltonian_operators

    @hamiltonian_operators.setter
    def hamiltonian_operators(self, new_hamiltonian_operators: Union["BCOO", None]):
        self._hamiltonian_operators = to_BCOO(new_hamiltonian_operators)

    @property
    def static_dissipators(self) -> Union["BCOO", None]:
        return self._static_dissipators

    @static_dissipators.setter
    def static_dissipators(self, new_static_dissipators: Union["BCOO", None]):
        """Operators constructed using dense operations."""
        self._static_dissipators = to_array(new_static_dissipators)
        if self._static_dissipators is not None:
            self._static_dissipators_adj = np.conjugate(
                np.transpose(self._static_dissipators, [0, 2, 1])
            ).copy()
            self._static_dissipators_product_sum = -0.5 * np.sum(
                np.matmul(self._static_dissipators_adj, self._static_dissipators), axis=0
            )
            self._static_dissipators = jsparse.BCOO.fromdense(
                self._static_dissipators.data, n_batch=1
            )
            self._static_dissipators_adj = jsparse.BCOO.fromdense(
                self._static_dissipators_adj.data, n_batch=1
            )
            self._static_dissipators_product_sum = jsparse.BCOO.fromdense(
                self._static_dissipators_product_sum.data
            )

    @property
    def dissipator_operators(self) -> Union["BCOO", None]:
        return self._dissipator_operators

    @dissipator_operators.setter
    def dissipator_operators(self, new_dissipator_operators: Union["BCOO", None]):
        """Operators constructed using dense operations."""
        self._dissipator_operators = to_array(new_dissipator_operators)
        if self._dissipator_operators is not None:
            self._dissipator_operators_adj = np.conjugate(
                np.transpose(self._dissipator_operators, [0, 2, 1])
            ).copy()
            self._dissipator_products = np.matmul(
                self._dissipator_operators_adj, self._dissipator_operators
            )
            self._dissipator_operators = jsparse.BCOO.fromdense(
                self._dissipator_operators.data, n_batch=1
            )
            self._dissipator_operators_adj = jsparse.BCOO.fromdense(
                self._dissipator_operators_adj.data, n_batch=1
            )
            self._dissipator_products = -0.5 * jsparse.BCOO.fromdense(
                self._dissipator_products.data, n_batch=1
            )

    def evaluate(self, ham_sig_vals: Array, dis_sig_vals: Array) -> Array:
        raise ValueError("Non-vectorized Lindblad collections cannot be evaluated without a state.")

    def evaluate_hamiltonian(self, ham_sig_vals: Union["BCOO", None]) -> "BCOO":
        r"""Compute the Hamiltonian.

        Args:
            ham_sig_vals: [Real] values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.
        Returns:
            Hamiltonian matrix.
        Raises:
            QiskitError: If collection not sufficiently specified.
        """
        if isinstance(ham_sig_vals, Array):
            ham_sig_vals = ham_sig_vals.data

        if self._static_hamiltonian is not None and self._hamiltonian_operators is not None:
            return (
                jsparse_linear_combo(ham_sig_vals, self._hamiltonian_operators)
                + self._static_hamiltonian
            )
        elif self._static_hamiltonian is None and self._hamiltonian_operators is not None:
            return jsparse_linear_combo(ham_sig_vals, self._hamiltonian_operators)
        elif self._static_hamiltonian is not None:
            return self._static_hamiltonian
        else:
            raise QiskitError(
                self.__class__.__name__
                + """ with None for both static_hamiltonian and
                                hamiltonian_operators cannot evaluate Hamiltonian."""
            )

    @wrap
    def evaluate_rhs(
        self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array], y: Array
    ) -> Array:
        r"""Evaluates Lindblad equation RHS given a pair of signal values
        for the hamiltonian terms and the dissipator terms. Expresses
        the RHS of the Lindblad equation as :math:`(A+B)y + y(A-B) + C`, where
            .. math::
            A = (-1/2)*\sum_jD_j^\dagger D_j + (-1/2)*\sum_j\gamma_j(t) L_j^\dagger L_j,

            B = -iH,

            C = \sum_j \gamma_j(t) L_j y L_j^\dagger.

        Args:
            ham_sig_vals: hamiltonian coefficient values, :math:`s_j(t)`.
            dis_sig_vals: dissipator signal values, :math:`\gamma_j(t)`.
            y: density matrix as (n,n) Array representing the state at time :math:`t`.
        Returns:
            RHS of Lindblad equation
            .. math::
                -i[H,y] + \sum_j\gamma_j(t)(L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y}).
        Raises:
            QiskitError: If operator collection is underspecified.
        """

        hamiltonian_matrix = None
        if self._static_hamiltonian is not None or self._hamiltonian_operators is not None:
            hamiltonian_matrix = -1j * self.evaluate_hamiltonian(ham_sig_vals)  # B matrix

        # if dissipators present (includes both hamiltonian is None and is not None)
        if self._dissipator_operators is not None or self._static_dissipators is not None:

            # A matrix
            if self._static_dissipators is None:
                dissipators_matrix = jsparse_linear_combo(dis_sig_vals, self._dissipator_products)
            elif self._dissipator_operators is None:
                dissipators_matrix = self._static_dissipators_product_sum
            else:
                dissipators_matrix = self._static_dissipators_product_sum + jsparse_linear_combo(
                    dis_sig_vals, self._dissipator_products
                )

            if hamiltonian_matrix is not None:
                left_mult_contribution = jsparse_matmul(hamiltonian_matrix + dissipators_matrix, y)
                right_mult_contribution = jsparse_matmul(
                    y, dissipators_matrix + (-1 * hamiltonian_matrix)
                )
            else:
                left_mult_contribution = jsparse_matmul(dissipators_matrix, y)
                right_mult_contribution = jsparse_matmul(y, dissipators_matrix)

            if len(y.shape) == 3:
                # Must do array broadcasting and transposition to ensure vectorization works
                y = jnp.broadcast_to(y, (1, y.shape[0], y.shape[1], y.shape[2])).transpose(
                    [1, 0, 2, 3]
                )
            if self._static_dissipators is None:
                both_mult_contribution = jnp.tensordot(
                    dis_sig_vals,
                    jsparse_triple_product(
                        self._dissipator_operators, y, self._dissipator_operators_adj
                    ),
                    axes=(-1, -3),
                )
            elif self._dissipator_operators is None:
                both_mult_contribution = jnp.sum(
                    jsparse_triple_product(
                        self._static_dissipators, y, self._static_dissipators_adj
                    ),
                    axis=-3,
                )
            else:
                both_mult_contribution = jnp.sum(
                    jsparse_triple_product(
                        self._static_dissipators, y, self._static_dissipators_adj
                    ),
                    axis=-3,
                ) + jnp.tensordot(
                    dis_sig_vals,
                    jsparse_triple_product(
                        self._dissipator_operators, y, self._dissipator_operators_adj
                    ),
                    axes=(-1, -3),
                )

            out = left_mult_contribution + right_mult_contribution + both_mult_contribution

            return out
        # if just hamiltonian
        elif hamiltonian_matrix is not None:
            return jsparse_matmul(hamiltonian_matrix, y) - jsparse_matmul(y, hamiltonian_matrix)
        else:
            raise QiskitError(
                """JAXSparseLindbladCollection with None for static_hamiltonian,
                                 hamiltonian_operators, static_dissipators, and
                                 dissipator_operators, cannot evaluate rhs."""
            )


class BaseVectorizedLindbladCollection(BaseLindbladOperatorCollection, BaseOperatorCollection):
    """Base class for Vectorized Lindblad collections.

    The vectorized Lindblad equation represents the Lindblad master equation in the structure
    of a linear matrix differential equation in standard form. Hence, this class inherits
    from both ``BaseLindbladOperatorCollection`` and ``BaseOperatorCollection``.

    This class manages the general property handling of converting operators in a Lindblad
    collection to the correct type, constructing vectorized versions, and combining for use in a
    BaseOperatorCollection. Requires implementation of:

        - ``convert_to_internal_type``: Convert operators to the required internal type,
          e.g. csr or Array.
        - ``evaluation_class``: Class property that returns the subclass of BaseOperatorCollection
          to be used when evaluating the model, e.g. DenseOperatorCollection or
          SparseOperatorCollection.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[Array] = None,
        hamiltonian_operators: Optional[Array] = None,
        static_dissipators: Optional[Array] = None,
        dissipator_operators: Optional[Array] = None,
    ):
        r"""Initialize collection.

        Args:
            static_hamiltonian: Constant term :math:`H_d` to be added to the Hamiltonian of the
                                system.
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) array.
            static_dissipators: Dissipator terms with coefficient 1.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) array.
        """
        self._static_hamiltonian = None
        self._hamiltonian_operators = None
        self._static_dissipators = None
        self._dissipator_operators = None
        self._static_operator = None
        self._operators = None
        super().__init__(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
        )

    @abstractmethod
    def convert_to_internal_type(self, obj: any) -> any:
        """Convert either a single operator or a list of operators to an internal representation."""

    @property
    @abstractmethod
    def evaluation_class(self) -> BaseOperatorCollection:
        """Class used for evaluating the vectorized model or RHS."""

    @property
    def static_hamiltonian(self) -> Union[Array, csr_matrix]:
        """Returns static part of operator collection."""
        return self._static_hamiltonian

    @static_hamiltonian.setter
    def static_hamiltonian(self, new_static_hamiltonian: Optional[Union[Array, csr_matrix]] = None):
        """Sets static_operator term."""
        self._static_hamiltonian = self.convert_to_internal_type(new_static_hamiltonian)
        if self._static_hamiltonian is not None:
            self._vec_static_hamiltonian = vec_commutator(self._static_hamiltonian)

        self.concatenate_static_operators()

    @property
    def hamiltonian_operators(self) -> Array:
        return self._hamiltonian_operators

    @hamiltonian_operators.setter
    def hamiltonian_operators(
        self, new_hamiltonian_operators: Optional[Union[Array, csr_matrix]] = None
    ):
        self._hamiltonian_operators = self.convert_to_internal_type(new_hamiltonian_operators)
        if self._hamiltonian_operators is not None:
            self._vec_hamiltonian_operators = vec_commutator(self._hamiltonian_operators)

        self.concatenate_operators()

    @property
    def static_dissipators(self) -> Union[Array, List[csr_matrix]]:
        return self._static_dissipators

    @static_dissipators.setter
    def static_dissipators(
        self, new_static_dissipators: Optional[Union[Array, List[csr_matrix]]] = None
    ):
        self._static_dissipators = self.convert_to_internal_type(new_static_dissipators)
        if self._static_dissipators is not None:
            self._vec_static_dissipators_sum = np.sum(
                vec_dissipator(self._static_dissipators), axis=0
            )

        self.concatenate_static_operators()

    @property
    def dissipator_operators(self) -> Union[Array, List[csr_matrix]]:
        return self._dissipator_operators

    @dissipator_operators.setter
    def dissipator_operators(
        self, new_dissipator_operators: Optional[Union[Array, List[csr_matrix]]] = None
    ):
        self._dissipator_operators = self.convert_to_internal_type(new_dissipator_operators)
        if self._dissipator_operators is not None:
            self._vec_dissipator_operators = vec_dissipator(self._dissipator_operators)

        self.concatenate_operators()

    def concatenate_static_operators(self):
        """Concatenate static hamiltonian and static dissipators."""
        if self._static_hamiltonian is not None and self._static_dissipators is not None:
            self._static_operator = self._vec_static_hamiltonian + self._vec_static_dissipators_sum
        elif self._static_hamiltonian is None and self._static_dissipators is not None:
            self._static_operator = self._vec_static_dissipators_sum
        elif self._static_hamiltonian is not None and self._static_dissipators is None:
            self._static_operator = self._vec_static_hamiltonian
        else:
            self._static_operator = None

    def concatenate_operators(self):
        """Concatenate hamiltonian operators and dissipator operators."""
        if self._hamiltonian_operators is not None and self._dissipator_operators is not None:
            self._operators = np.append(
                self._vec_hamiltonian_operators, self._vec_dissipator_operators, axis=0
            )
        elif self._hamiltonian_operators is not None and self._dissipator_operators is None:
            self._operators = self._vec_hamiltonian_operators
        elif self._hamiltonian_operators is None and self._dissipator_operators is not None:
            self._operators = self._vec_dissipator_operators
        else:
            self._operators = None

    def concatenate_signals(
        self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array]
    ) -> Array:
        """Concatenate hamiltonian and linblad signals."""
        if self._hamiltonian_operators is not None and self._dissipator_operators is not None:
            return np.append(ham_sig_vals, dis_sig_vals, axis=-1)
        if self._hamiltonian_operators is not None and self._dissipator_operators is None:
            return ham_sig_vals
        if self._hamiltonian_operators is None and self._dissipator_operators is not None:
            return dis_sig_vals

        return None

    def evaluate(self, ham_sig_vals: Union[None, Array], dis_sig_vals: Union[None, Array]) -> Array:
        """Evaluate the model."""
        signal_values = self.concatenate_signals(ham_sig_vals, dis_sig_vals)
        return self.evaluation_class.evaluate(self, signal_values)

    def evaluate_rhs(self, ham_sig_vals: Array, dis_sig_vals: Array, y: Array) -> Array:
        r"""Evaluates the RHS of the Lindblad equation using
        vectorized maps.

        Args:
            ham_sig_vals: hamiltonian signal coefficients.
            dis_sig_vals: dissipator signal coefficients.
                If none involved, pass None.
            y: Density matrix represented as a vector using column-stacking
                convention.
        Returns:
            Vectorized RHS of Lindblad equation :math:`\dot{\rho}` in column-stacking
                convention.
        """
        return self.evaluate(ham_sig_vals, dis_sig_vals) @ y


class DenseVectorizedLindbladCollection(
    BaseVectorizedLindbladCollection, DenseLindbladCollection, DenseOperatorCollection
):
    r"""Vectorized version of DenseLindbladCollection.

    Utilizes BaseVectorizedLindbladCollection for property handling, DenseLindbladCollection
    for evaluate_hamiltonian, and DenseOperatorCollection for operator property handling.
    """

    def convert_to_internal_type(self, obj: any) -> Array:
        return to_array(obj)

    @property
    def evaluation_class(self):
        return DenseOperatorCollection


class SparseVectorizedLindbladCollection(
    BaseVectorizedLindbladCollection, SparseLindbladCollection, SparseOperatorCollection
):
    r"""Vectorized version of SparseLindbladCollection.

    Utilizes BaseVectorizedLindbladCollection for property handling, SparseLindbladCollection
    for evaluate_hamiltonian, and SparseOperatorCollection for static_operator and operator
    property handling.
    """

    def convert_to_internal_type(self, obj: any) -> Union[csr_matrix, List[csr_matrix]]:
        if obj is None:
            return None

        obj = to_csr(obj)
        if issparse(obj):
            return np.round(obj, decimals=self._decimals)
        else:
            return [np.round(sub_obj, decimals=self._decimals) for sub_obj in obj]

    @property
    def evaluation_class(self):
        return SparseOperatorCollection


class JAXSparseVectorizedLindbladCollection(
    BaseVectorizedLindbladCollection, JAXSparseLindbladCollection, JAXSparseOperatorCollection
):
    r"""Vectorized version of JAXSparseLindbladCollection.

    Utilizes BaseVectorizedLindbladCollection for property handling, JAXSparseLindbladCollection
    for evaluate_hamiltonian, and JAXSparseOperatorCollection for static_operator and operator
    property handling.
    """

    def convert_to_internal_type(self, obj: any) -> "BCOO":
        return to_BCOO(obj)

    @property
    def evaluation_class(self):
        return JAXSparseOperatorCollection

    def concatenate_static_operators(self):
        """Override base class to convert to BCOO again at the end. The vectorization operations
        are not implemented for BCOO type, so they automatically get converted to Arrays,
        and hence need to be converted back.
        """
        super().concatenate_static_operators()
        self._static_operator = self.convert_to_internal_type(self._static_operator)

    def concatenate_operators(self):
        """Override base class to convert to BCOO again at the end. The vectorization operations
        are not implemented for BCOO type, so they automatically get converted to Arrays,
        and hence need to be converted back.
        """
        super().concatenate_operators()
        self._operators = self.convert_to_internal_type(self._operators)

    @wrap
    def evaluate_rhs(self, ham_sig_vals: Array, dis_sig_vals: Array, y: Array) -> Array:
        return jsparse_matmul(self.evaluate(ham_sig_vals, dis_sig_vals), y)


def package_density_matrices(y: Array) -> Array:
    """Sends a (k,n,n) Array y of density matrices to a
    (k,1) Array of dtype object, where entry [j,0] is
    y[j]. Formally avoids For loops through vectorization.
    Args:
        y: (k,n,n) Array.
    Returns:
        Array with dtype object."""
    # As written here, only works for (n,n) Arrays
    obj_arr = np.empty(shape=(1), dtype="O")
    obj_arr[0] = y
    return obj_arr


# Using vectorization with signature, works on (k,n,n) Arrays -> (k,1) Array
package_density_matrices = np.vectorize(package_density_matrices, signature="(n,n)->(1)")


def unpackage_density_matrices(y: Array) -> Array:
    """Inverse function of package_density_matrices,
    Much slower than packaging. Avoid using unless
    absolutely needed (as in case of passing multiple
    density matrices to SparseLindbladCollection.evaluate_rhs)."""
    return y[0]


unpackage_density_matrices = np.vectorize(unpackage_density_matrices, signature="(1)->(n,n)")
