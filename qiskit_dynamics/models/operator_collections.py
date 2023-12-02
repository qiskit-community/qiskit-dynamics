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

"""Operator collections as math/calculation objects for model classes."""

from typing import Any, Union, List, Optional
import numpy as np
from scipy.sparse import csr_matrix, issparse

from qiskit import QiskitError
from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS as numpy_alias
from qiskit_dynamics.arraylias.alias import ArrayLike, _numpy_multi_dispatch
from qiskit_dynamics.type_utils import to_csr, vec_commutator, vec_dissipator


def _linear_combo(coeffs, mats):
    return _numpy_multi_dispatch(coeffs, mats, path="linear_combo")


def _matmul(A, B, **kwargs):
    return _numpy_multi_dispatch(A, B, path="matmul", **kwargs)


def _to_csr_object_array(ops, decimals):
    """Turn a list of matrices into a numpy object array of scipy sparse matrix instances."""
    if ops is None:
        return None
    return np.array(
        [numpy_alias(like="scipy_sparse").asarray(np.round(op, decimals)) for op in ops]
    )


class OperatorCollection:
    r"""Class for evaluating a linear combination of operators acting on a state.

    This class represents a function :math:`c,y \mapsto \Lambda(c, y)`, which is assumed to be
    decomposed as :math:`\Lambda(c, y) = (G_d + \sum_jc_jG_j)  y` for matrices :math:`G_d` and
    :math:`G_j`, with :math:`G_d` referred to as the static operator.

    This works for ``array_library in ["numpy", "jax", "jax_sparse"]``.
    """

    def __init__(
        self,
        static_operator: Optional[ArrayLike] = None,
        operators: Optional[ArrayLike] = None,
        array_library: Optional[str] = None,
    ):
        """Initialize.

        Args:
            operators: ``(k,n,n)`` array specifying the terms :math:`G_j`.
            static_operator: ``(n,n)`` array specifying the extra static_operator :math:`G_d`.
            array_library: Underlying library to use for array operations.

        Raises:
            QiskitError: If "scipy_sparse" is passed as array_library.
        """
        if array_library == "scipy_sparse":
            raise QiskitError("scipy_sparse is not a valid array_library for OperatorCollection.")

        if static_operator is not None:
            self._static_operator = numpy_alias(like=array_library).asarray(static_operator)
        else:
            self._static_operator = None

        if operators is not None:
            self._operators = numpy_alias(like=array_library).asarray(operators)
        else:
            self._operators = None

    @property
    def dim(self) -> int:
        """The matrix dimension."""
        if self.static_operator is not None:
            return self.static_operator.shape[-1]
        else:
            return self.operators[0].shape[-1]

    @property
    def static_operator(self) -> Union[ArrayLike, None]:
        """The static part of the operator collection."""
        return self._static_operator

    @property
    def operators(self) -> Union[ArrayLike, None]:
        """The operators of this collection."""
        return self._operators

    def evaluate(self, coefficients: Union[ArrayLike, None]) -> ArrayLike:
        r"""Evaluate the operator :math:`\Lambda(c, \cdot) = (G_d + \sum_jc_jG_j)`.

        Args:
            coefficients: The coefficient values :math:`c`.

        Returns:
            An ``ArrayLike`` that acts on states ``y`` via multiplication.

        Raises:
            QiskitError: If both static_operator and operators are ``None``.
        """
        if self._static_operator is not None and self._operators is not None:
            return _linear_combo(coefficients, self._operators) + self._static_operator
        elif self._static_operator is None and self._operators is not None:
            return _linear_combo(coefficients, self._operators)
        elif self._static_operator is not None:
            return self._static_operator
        raise QiskitError(
            "OperatorCollection with None for both static_operator and operators cannot be "
            "evaluated."
        )

    def evaluate_rhs(self, coefficients: Union[ArrayLike, None], y: ArrayLike) -> ArrayLike:
        r"""Evaluate the function and return :math:`\Lambda(c, y) = (G_d + \sum_jc_jG_j)  y`.

        Args:
            coefficients: The coefficients :math:`c`.
            y: The system state.

        Returns:
            The evaluated function.
        """
        return _matmul(self.evaluate(coefficients), y)

    def __call__(
        self, coefficients: Union[ArrayLike, None], y: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """Call :meth:`~evaluate` or :meth:`~evaluate_rhs` depending on the presense of ``y``.

        Args:
            coefficients: The coefficients :math:`c` to use for evaluation.
            y: Optionally, the system state.

        Returns:
            The output of :meth:`~evaluate` or :meth:`self.evaluate_rhs`.
        """
        return self.evaluate(coefficients) if y is None else self.evaluate_rhs(coefficients, y)


class ScipySparseOperatorCollection:
    r"""Scipy sparse version of :class:`OperatorCollection`."""

    def __init__(
        self,
        static_operator: Optional[ArrayLike] = None,
        operators: Optional[ArrayLike] = None,
        decimals: Optional[int] = 10,
    ):
        """Initialize.

        Args:
            static_operator: (n,n) Array specifying the static_operator term :math:`G_d`.
            operators: (k,n,n) Array specifying the terms :math:`G_j`.
            decimals: Values will be rounded at ``decimals`` places after decimal.
        """
        if static_operator is not None:
            self._static_operator = numpy_alias(like="scipy_sparse").asarray(
                np.round(static_operator, decimals)
            )
        else:
            self._static_operator = None

        self._operators = _to_csr_object_array(operators, decimals)

    @property
    def static_operator(self) -> Union[None, csr_matrix]:
        """The static part of the operator collection."""
        return self._static_operator

    @property
    def operators(self) -> Union[None, List[csr_matrix]]:
        """The operators of this collection."""
        if self._operators is None:
            return None

        return list(self._operators)

    def evaluate(self, coefficients: Union[ArrayLike, None]) -> csr_matrix:
        r"""Evaluate the operator :math:`\Lambda(c, \cdot) = (G_d + \sum_jc_jG_j)`.

        Args:
            coefficients: The signals values :math:`c` to use on the operators.

        Returns:
            An :class:`~Array` that acts on states ``y`` via multiplication.

        Raises:
            QiskitError: If collection cannot be evaluated.
        """
        if self._static_operator is not None and self._operators is not None:
            return (
                np.tensordot(coefficients, self._operators, axes=1).item() + self._static_operator
            )
        elif self._static_operator is None and self._operators is not None:
            return np.tensordot(coefficients, self._operators, axes=1).item()
        elif self.static_operator is not None:
            return self._static_operator
        raise QiskitError(
            f"{type(self).__name__} with None for both static_operator and operators cannot be "
            "evaluated."
        )

    def evaluate_rhs(self, coefficients: Union[ArrayLike, None], y: ArrayLike) -> ArrayLike:
        r"""Evaluate the function and return :math:`\Lambda(c, y) = (G_d + \sum_jc_jG_j)  y`.

        Args:
            coefficients: The signals :math:`c` to use on the operators.
            y: The system state.

        Returns:
            The evaluated function.
        Raises:
            QiskitError: If the function cannot be evaluated.
        """
        if len(y.shape) == 2:
            # For 2d array, compute linear combination then multiply
            gen = self.evaluate(coefficients)
            return gen.dot(y)
        elif len(y.shape) == 1:
            # For a 1d array, multiply individual matrices then compute linear combination
            tmparr = np.empty(shape=(1), dtype="O")
            tmparr[0] = y

            if self._static_operator is not None and self._operators is not None:
                return np.dot(coefficients, self._operators * tmparr) + self.static_operator.dot(y)
            elif self._static_operator is None and self._operators is not None:
                return np.dot(coefficients, self._operators * tmparr)
            elif self.static_operator is not None:
                return self.static_operator.dot(y)

            raise QiskitError(
                self.__class__.__name__
                + """  with None for both static_operator and
                                operators cannot be evaluated."""
            )

        raise QiskitError(self.__class__.__name__ + """  cannot evaluate RHS for y.ndim > 3.""")

    def __call__(
        self, coefficients: Union[ArrayLike, None], y: Optional[ArrayLike] = None
    ) -> ArrayLike:
        """Call :meth:`~evaluate` or :meth:`~evaluate_rhs` depending on the presense of ``y``.

        Args:
            coefficients: The coefficients :math:`c` to use for evaluation.
            y: Optionally, the system state.

        Returns:
            The output of :meth:`~evaluate` or :meth:`self.evaluate_rhs`.
        """
        return self.evaluate(coefficients) if y is None else self.evaluate_rhs(coefficients, y)


class LindbladCollection:
    r"""Class representing a two-variable matrix function for evaluating the right hand
    side of the Lindblad equation.

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

    This class works for ``array_library in ["numpy", "jax", "jax_sparse"]``.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[ArrayLike] = None,
        hamiltonian_operators: Optional[ArrayLike] = None,
        static_dissipators: Optional[ArrayLike] = None,
        dissipator_operators: Optional[ArrayLike] = None,
        array_library: Optional[str] = None,
    ):
        r"""Initialize collection. Argument types depend on concrete subclass.

        Args:
            static_hamiltonian: Constant term :math:`H_d` to be added to the Hamiltonian of the
                                system.
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) array.
            static_dissipators: Constant dissipator terms.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) array.
            array_library: Array library to use for storing arrays in the collection.

        Raises:
            QiskitError: If "scipy_sparse" is passed as the array_library.
        """

        if array_library == "scipy_sparse":
            raise QiskitError("scipy_sparse is not a valid array_library for OperatorCollection.")

        if static_hamiltonian is not None:
            self._static_hamiltonian = numpy_alias(like=array_library).asarray(static_hamiltonian)
        else:
            self._static_hamiltonian = None

        if hamiltonian_operators is not None:
            self._hamiltonian_operators = numpy_alias(like=array_library).asarray(
                hamiltonian_operators
            )
        else:
            self._hamiltonian_operators = None

        if static_dissipators is not None:
            if array_library == "jax_sparse":
                self._static_dissipators = numpy_alias(like="jax").asarray(static_dissipators)
            else:
                self._static_dissipators = numpy_alias(like=array_library).asarray(
                    static_dissipators
                )

            self._static_dissipators_adj = unp.conjugate(
                unp.transpose(self._static_dissipators, [0, 2, 1])
            )
            self._static_dissipators_product_sum = -0.5 * unp.sum(
                unp.matmul(self._static_dissipators_adj, self._static_dissipators), axis=0
            )

            if array_library == "jax_sparse":
                from jax.experimental.sparse import BCOO

                self._static_dissipators = BCOO.fromdense(self._static_dissipators, n_batch=1)
                self._static_dissipators_adj = BCOO.fromdense(
                    self._static_dissipators_adj, n_batch=1
                )
                self._static_dissipators_product_sum = BCOO.fromdense(
                    self._static_dissipators_product_sum
                )
        else:
            self._static_dissipators = None

        if dissipator_operators is not None:
            if array_library == "jax_sparse":
                self._dissipator_operators = numpy_alias(like="jax").asarray(dissipator_operators)
            else:
                self._dissipator_operators = numpy_alias(like=array_library).asarray(
                    dissipator_operators
                )

            self._dissipator_operators_adj = unp.conjugate(
                unp.transpose(self._dissipator_operators, [0, 2, 1])
            )
            self._dissipator_products = -0.5 * unp.matmul(
                self._dissipator_operators_adj, self._dissipator_operators
            )

            if array_library == "jax_sparse":
                from jax.experimental.sparse import BCOO

                self._dissipator_operators = BCOO.fromdense(self._dissipator_operators, n_batch=1)
                self._dissipator_operators_adj = BCOO.fromdense(
                    self._dissipator_operators_adj, n_batch=1
                )
                self._dissipator_products = BCOO.fromdense(self._dissipator_products, n_batch=1)
        else:
            self._dissipator_operators = None

    @property
    def static_hamiltonian(self) -> ArrayLike:
        """The static part of the Hamiltonian."""
        return self._static_hamiltonian

    @property
    def hamiltonian_operators(self) -> ArrayLike:
        """The operators for the non-static part of Hamiltonian."""
        return self._hamiltonian_operators

    @property
    def static_dissipators(self) -> ArrayLike:
        """The operators for the static part of dissipator."""
        return self._static_dissipators

    @property
    def dissipator_operators(self) -> ArrayLike:
        """The operators for the non-static part of dissipator."""
        return self._dissipator_operators

    def evaluate_hamiltonian(self, ham_coefficients: Optional[ArrayLike]) -> ArrayLike:
        r"""Evaluate the Hamiltonian of the model.

        Args:
            ham_coefficients: The values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.

        Returns:
            The Hamiltonian.

        Raises:
            QiskitError: If collection not sufficiently specified.
        """
        if self._static_hamiltonian is not None and self._hamiltonian_operators is not None:
            return (
                _linear_combo(ham_coefficients, self._hamiltonian_operators)
                + self._static_hamiltonian
            )
        elif self._static_hamiltonian is None and self._hamiltonian_operators is not None:
            return _linear_combo(ham_coefficients, self._hamiltonian_operators)
        elif self._static_hamiltonian is not None:
            return self._static_hamiltonian
        else:
            raise QiskitError(
                self.__class__.__name__
                + """ with None for both static_hamiltonian and
                                hamiltonian_operators cannot evaluate Hamiltonian."""
            )

    def evaluate(
        self, ham_coefficients: Optional[ArrayLike], dis_coefficients: Optional[ArrayLike]
    ) -> ArrayLike:
        r"""Evaluate the function and return :math:`\Lambda(c_1, c_2, \cdot)`.

        Args:
            ham_coefficients: The signals :math:`c_1` to use on the Hamiltonians.
            dis_coefficients: The signals :math:`c_2` to use on the dissipators.

        Returns:
            The evaluated function.

        Raises:
            ValueError: Always.
        """
        raise ValueError("Non-vectorized Lindblad collections cannot be evaluated without a state.")

    def evaluate_rhs(
        self,
        ham_coefficients: Optional[ArrayLike],
        dis_coefficients: Optional[ArrayLike],
        y: ArrayLike,
    ) -> ArrayLike:
        r"""Evaluates Lindblad equation RHS given a pair of signal values for the hamiltonian terms
        and the dissipator terms.

        Expresses the RHS of the Lindblad equation as :math:`(A+B)y + y(A-B) + C`, where

        .. math::
            A = (-1/2)*\sum_jD_j^\dagger D_j + (-1/2)*\sum_j\gamma_j(t) L_j^\dagger L_j,

            B = -iH,

            C = \sum_j \gamma_j(t) L_j y L_j^\dagger.

        Args:
            ham_coefficients: Hamiltonian coefficient values, :math:`s_j(t)`.
            dis_coefficients: Dissipator signal values, :math:`\gamma_j(t)`.
            y: Density matrix as ``(n,n)`` array representing the state at time :math:`t`.

        Returns:
            RHS of the Lindblad equation
            .. math::
                -i[H,y] + \sum_j\gamma_j(t)(L_j y L_j^\dagger - (1/2) * {L_j^\daggerL_j,y}).

        Raises:
            QiskitError: If operator collection is underspecified.
        """

        hamiltonian_matrix = None
        if self._static_hamiltonian is not None or self._hamiltonian_operators is not None:
            hamiltonian_matrix = -1j * self.evaluate_hamiltonian(ham_coefficients)  # B matrix

        # if dissipators present (includes both hamiltonian is None and is not None)
        if self._dissipator_operators is not None or self._static_dissipators is not None:
            # A matrix
            if self._static_dissipators is None:
                dissipators_matrix = _linear_combo(dis_coefficients, self._dissipator_products)
            elif self._dissipator_operators is None:
                dissipators_matrix = self._static_dissipators_product_sum
            else:
                dissipators_matrix = self._static_dissipators_product_sum + _linear_combo(
                    dis_coefficients, self._dissipator_products
                )

            if hamiltonian_matrix is not None:
                left_mult_contribution = _matmul(hamiltonian_matrix + dissipators_matrix, y)
                right_mult_contribution = _matmul(y, dissipators_matrix - hamiltonian_matrix)
            else:
                left_mult_contribution = _matmul(dissipators_matrix, y)
                right_mult_contribution = _matmul(y, dissipators_matrix)

            if len(y.shape) == 3:
                # Must do array broadcasting and transposition to ensure vectorization works
                y = unp.broadcast_to(y, (1, y.shape[0], y.shape[1], y.shape[2])).transpose(
                    [1, 0, 2, 3]
                )

            if self._static_dissipators is None:
                both_mult_contribution = _numpy_multi_dispatch(
                    dis_coefficients,
                    _matmul(self._dissipator_operators, _matmul(y, self._dissipator_operators_adj)),
                    path="tensordot",
                    axes=(-1, -3),
                )
            elif self._dissipator_operators is None:
                both_mult_contribution = unp.sum(
                    _matmul(self._static_dissipators, _matmul(y, self._static_dissipators_adj)),
                    axis=-3,
                )
            else:
                both_mult_contribution = unp.sum(
                    _matmul(self._static_dissipators, _matmul(y, self._static_dissipators_adj)),
                    axis=-3,
                ) + _numpy_multi_dispatch(
                    dis_coefficients,
                    _matmul(self._dissipator_operators, _matmul(y, self._dissipator_operators_adj)),
                    path="tensordot",
                    axes=(-1, -3),
                )

            return left_mult_contribution + right_mult_contribution + both_mult_contribution
        # if just hamiltonian
        elif hamiltonian_matrix is not None:
            return _matmul(hamiltonian_matrix, y) - _matmul(y, hamiltonian_matrix)
        else:
            raise QiskitError(
                """LindbladCollection with None for static_hamiltonian,
                                 hamiltonian_operators, static_dissipators, and
                                 dissipator_operators, cannot evaluate rhs."""
            )

    def __call__(
        self,
        ham_coefficients: Optional[ArrayLike],
        dis_coefficients: Optional[ArrayLike],
        y: Optional[ArrayLike],
    ) -> ArrayLike:
        """Call :meth:`~evaluate` or :meth:`~evaluate_rhs` depending on the presense of ``y``.

        Args:
            ham_coefficients: The signals :math:`c_1` to use on the Hamiltonians.
            dis_coefficients: The signals :math:`c_2` to use on the dissipators.
            y: Optionally, the system state.

        Returns:
            The evaluated function.
        """
        if y is None:
            return self.evaluate(ham_coefficients, dis_coefficients)

        return self.evaluate_rhs(ham_coefficients, dis_coefficients, y)


class ScipySparseLindbladCollection:
    """Scipy sparse version of LindbladCollection."""

    def __init__(
        self,
        static_hamiltonian: Optional[ArrayLike] = None,
        hamiltonian_operators: Optional[ArrayLike] = None,
        static_dissipators: Optional[ArrayLike] = None,
        dissipator_operators: Optional[ArrayLike] = None,
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
        if static_hamiltonian is not None:
            self._static_hamiltonian = numpy_alias(like="scipy_sparse").asarray(
                np.round(static_hamiltonian, decimals)
            )
        else:
            self._static_hamiltonian = None

        self._hamiltonian_operators = _to_csr_object_array(hamiltonian_operators, decimals)
        self._static_dissipators = _to_csr_object_array(static_dissipators, decimals)
        self._dissipator_operators = _to_csr_object_array(dissipator_operators, decimals)

        if static_dissipators is not None:
            # setup adjoints
            self._static_dissipators_adj = np.array(
                [op.conj().transpose() for op in self._static_dissipators]
            )

            # pre-compute product and sum
            self._static_dissipators_product_sum = -0.5 * np.sum(
                self._static_dissipators_adj * self._static_dissipators, axis=0
            )

        if self._dissipator_operators is not None:
            # setup adjoints
            self._dissipator_operators_adj = np.array(
                [op.conj().transpose() for op in self._dissipator_operators]
            )

            # pre-compute products
            self._dissipator_products = self._dissipator_operators_adj * self._dissipator_operators

    @property
    def static_hamiltonian(self) -> Union[None, csr_matrix]:
        """The static part of the operator collection."""
        return self._static_hamiltonian

    @property
    def hamiltonian_operators(self) -> Union[None, list]:
        """The operators for the non-static part of Hamiltonian."""
        if self._hamiltonian_operators is None:
            return None

        return list(self._hamiltonian_operators)

    @property
    def static_dissipators(self) -> Union[None, list]:
        """The operators for the static part of dissipator."""
        if self._static_dissipators is None:
            return None

        return list(self._static_dissipators)

    @property
    def dissipator_operators(self) -> Union[None, list]:
        """The operators for the non-static part of dissipator."""
        if self._dissipator_operators is None:
            return None

        return list(self._dissipator_operators)

    def evaluate_hamiltonian(self, ham_coefficients: Optional[ArrayLike]) -> csr_matrix:
        r"""Compute the Hamiltonian.

        Args:
            ham_coefficients: [Real] values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.
        Returns:
            Hamiltonian matrix.
        Raises:
            QiskitError: If collection not sufficiently specified.
        """
        if self._static_hamiltonian is not None and self._hamiltonian_operators is not None:
            return (
                np.sum(ham_coefficients * self._hamiltonian_operators, axis=-1)
                + self.static_hamiltonian
            )
        elif self._static_hamiltonian is None and self._hamiltonian_operators is not None:
            return np.sum(ham_coefficients * self._hamiltonian_operators, axis=-1)
        elif self._static_hamiltonian is not None:
            return self._static_hamiltonian
        else:
            raise QiskitError(
                self.__class__.__name__
                + """ with None for both static_hamiltonian and
                                hamiltonian_operators cannot evaluate Hamiltonian."""
            )

    def evaluate(
        self, ham_coefficients: Optional[ArrayLike], dis_coefficients: Optional[ArrayLike]
    ) -> ArrayLike:
        r"""Evaluate the function and return :math:`\Lambda(c_1, c_2, \cdot)`.

        Args:
            ham_coefficients: The signals :math:`c_1` to use on the Hamiltonians.
            dis_coefficients: The signals :math:`c_2` to use on the dissipators.

        Returns:
            The evaluated function.

        Raises:
            ValueError: Always.
        """
        raise ValueError("Non-vectorized Lindblad collections cannot be evaluated without a state.")

    def evaluate_rhs(
        self,
        ham_coefficients: Optional[ArrayLike],
        dis_coefficients: Optional[ArrayLike],
        y: ArrayLike,
    ) -> ArrayLike:
        r"""Evaluate the RHS of the Lindblad model for a given list of signal values.

        Args:
            ham_coefficients: Stores Hamiltonian signal values :math:`s_j(t)`.
            dis_coefficients: Stores dissipator signal values :math:`\gamma_j(t)`. Pass ``None`` if
                no dissipator operators are involved.
            y: Density matrix of the system, a ``(k,n,n)`` Array.

        Returns:
            RHS of the Lindbladian.

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
            hamiltonian_matrix = -1j * self.evaluate_hamiltonian(ham_coefficients)  # B matrix

        # package (n,n) Arrays as (1)
        # Arrays of dtype object, or (k,n,n) Arrays as (k,1) Arrays of dtype object
        y = package_density_matrices(y)

        # if dissipators present (includes both hamiltonian is None and is not None)
        if self._dissipator_operators is not None or self._static_dissipators is not None:
            # A matrix
            if self._static_dissipators is None:
                dissipators_matrix = np.sum(
                    -0.5 * dis_coefficients * self._dissipator_products, axis=-1
                )
            elif self._dissipator_operators is None:
                dissipators_matrix = self._static_dissipators_product_sum
            else:
                dissipators_matrix = self._static_dissipators_product_sum + np.sum(
                    -0.5 * dis_coefficients * self._dissipator_products, axis=-1
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
                    (dis_coefficients * self._dissipator_operators)
                    * y
                    * self._dissipator_operators_adj,
                    axis=-1,
                )
            elif self._dissipator_operators is None:
                both_mult_contribution = np.sum(
                    self._static_dissipators * y * self._static_dissipators_adj, axis=-1
                )
            else:
                both_mult_contribution = np.sum(
                    (dis_coefficients * self._dissipator_operators)
                    * y
                    * self._dissipator_operators_adj,
                    axis=-1,
                ) + np.sum(self._static_dissipators * y * self._static_dissipators_adj, axis=-1)

            out = left_mult_contribution + right_mult_contribution + both_mult_contribution

        elif hamiltonian_matrix is not None:
            out = (([hamiltonian_matrix] * y) - (y * [hamiltonian_matrix]))[0]
        else:
            raise QiskitError(
                "ScipySparseLindbladCollection with None for static_hamiltonian, "
                "hamiltonian_operators, and dissipator_operators, cannot evaluate rhs."
            )
        if len(y.shape) == 2:
            # Very slow; avoid if not necessary (or if better implementation found). Needs to
            # map a (k) Array of dtype object with j^{th} entry a (n,n) Array -> (k,n,n) Array.
            out = unpackage_density_matrices(out.reshape(y.shape[0], 1))

        return out

    def __call__(
        self,
        ham_coefficients: Union[None, ArrayLike],
        dis_coefficients: Union[None, ArrayLike],
        y: Optional[ArrayLike],
    ) -> ArrayLike:
        """Call :meth:`~evaluate` or :meth:`~evaluate_rhs` depending on the presense of ``y``.

        Args:
            ham_coefficients: The signals :math:`c_1` to use on the Hamiltonians.
            dis_coefficients: The signals :math:`c_2` to use on the dissipators.
            y: Optionally, the system state.

        Returns:
            The evaluated function.
        """
        if y is None:
            return self.evaluate(ham_coefficients, dis_coefficients)

        return self.evaluate_rhs(ham_coefficients, dis_coefficients, y)


class VectorizedLindbladCollection:
    """Vectorized Lindblad collection class.

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

    This class works for ``array_library in ["numpy", "jax", "jax_sparse"]``.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[ArrayLike] = None,
        hamiltonian_operators: Optional[ArrayLike] = None,
        static_dissipators: Optional[ArrayLike] = None,
        dissipator_operators: Optional[ArrayLike] = None,
        array_library: Optional[str] = None,
    ):
        r"""Initialize collection.

        Args:
            static_hamiltonian: Constant term :math:`H_d` to be added to the Hamiltonian of the
                                system.
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) array.
            static_dissipators: Dissipator terms with coefficient 1.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) array.

        Raises:
            QiskitError: If "scipy_sparse" is passed as array_library.
        """

        self._array_library = array_library

        if array_library == "scipy_sparse":
            raise QiskitError("scipy_sparse is not a valid array_library for OperatorCollection.")

        if static_hamiltonian is not None:
            self._static_hamiltonian = self._convert_to_array_type(static_hamiltonian)
            self._vec_static_hamiltonian = vec_commutator(self._static_hamiltonian)
        else:
            self._static_hamiltonian = None

        if hamiltonian_operators is not None:
            self._hamiltonian_operators = self._convert_to_array_type(hamiltonian_operators)
            self._vec_hamiltonian_operators = vec_commutator(self._hamiltonian_operators)
        else:
            self._hamiltonian_operators = None

        if static_dissipators is not None:
            self._static_dissipators = self._convert_to_array_type(static_dissipators)
            self._vec_static_dissipators_sum = unp.sum(
                vec_dissipator(self._static_dissipators), axis=0
            )
        else:
            self._static_dissipators = None

        if dissipator_operators is not None:
            self._dissipator_operators = self._convert_to_array_type(dissipator_operators)
            self._vec_dissipator_operators = vec_dissipator(self._dissipator_operators)
        else:
            self._dissipator_operators = None

        # concatenate static operators
        if self._static_hamiltonian is not None and self._static_dissipators is not None:
            static_operator = self._vec_static_hamiltonian + self._vec_static_dissipators_sum
        elif self._static_hamiltonian is None and self._static_dissipators is not None:
            static_operator = self._vec_static_dissipators_sum
        elif self._static_hamiltonian is not None and self._static_dissipators is None:
            static_operator = self._vec_static_hamiltonian
        else:
            static_operator = None

        # concatenate non-static operators
        if self._hamiltonian_operators is not None and self._dissipator_operators is not None:
            operators = unp.append(
                self._vec_hamiltonian_operators, self._vec_dissipator_operators, axis=0
            )
        elif self._hamiltonian_operators is not None and self._dissipator_operators is None:
            operators = self._vec_hamiltonian_operators
        elif self._hamiltonian_operators is None and self._dissipator_operators is not None:
            operators = self._vec_dissipator_operators
        else:
            operators = None

        # build internally used operator collection
        self._operator_collection = self._construct_operator_collection(
            static_operator=static_operator, operators=operators
        )

    @property
    def static_hamiltonian(self) -> Union[ArrayLike, None]:
        """The static part of the operator collection."""
        return self._static_hamiltonian

    @property
    def hamiltonian_operators(self) -> Union[ArrayLike, None]:
        """The operators for the non-static part of Hamiltonian."""
        return self._hamiltonian_operators

    @property
    def static_dissipators(self) -> Union[ArrayLike, None]:
        """The operators for the static part of dissipator."""
        return self._static_dissipators

    @property
    def dissipator_operators(self) -> Union[ArrayLike, None]:
        """The operators for the non-static part of dissipator."""
        return self._dissipator_operators

    def evaluate_hamiltonian(self, ham_coefficients: Union[None, ArrayLike]) -> ArrayLike:
        r"""Evaluate the Hamiltonian of the model.

        Args:
            ham_coefficients: The values of :math:`s_j` in :math:`H = \sum_j s_j(t) H_j + H_d`.

        Returns:
            The Hamiltonian.

        Raises:
            QiskitError: If collection not sufficiently specified.
        """
        if self._static_hamiltonian is not None and self._hamiltonian_operators is not None:
            return (
                _linear_combo(ham_coefficients, self._hamiltonian_operators)
                + self._static_hamiltonian
            )
        elif self._static_hamiltonian is None and self._hamiltonian_operators is not None:
            return _linear_combo(ham_coefficients, self._hamiltonian_operators)
        elif self._static_hamiltonian is not None:
            return self._static_hamiltonian
        else:
            raise QiskitError(
                self.__class__.__name__
                + """ with None for both static_hamiltonian and
                                hamiltonian_operators cannot evaluate Hamiltonian."""
            )

    def evaluate(
        self, ham_coefficients: Optional[ArrayLike], dis_coefficients: Optional[ArrayLike]
    ) -> ArrayLike:
        r"""Compute and return :math:`\Lambda(c_1, c_2, \cdot)`.

        Args:
            ham_coefficients: The signals :math:`c_1` to use on the Hamiltonians.
            dis_coefficients: The signals :math:`c_2` to use on the dissipators.

        Returns:
            The evaluated function.
        """
        coeffs = self._concatenate_coefficients(ham_coefficients, dis_coefficients)
        return self._operator_collection.evaluate(coeffs)

    def evaluate_rhs(
        self,
        ham_coefficients: Optional[ArrayLike],
        dis_coefficients: Optional[ArrayLike],
        y: ArrayLike,
    ) -> ArrayLike:
        r"""Evaluates the RHS of the Lindblad equation using vectorized maps.

        Args:
            ham_coefficients: Hamiltonian signal coefficients.
            dis_coefficients: Dissipator signal coefficients. If none involved, pass ``None``.
            y: Density matrix represented as a vector using column-stacking convention.

        Returns:
            Vectorized RHS of Lindblad equation :math:`\dot{\rho}` in column-stacking convention.
        """
        coeffs = self._concatenate_coefficients(ham_coefficients, dis_coefficients)
        return self._operator_collection.evaluate_rhs(coeffs, y)

    def __call__(
        self,
        ham_coefficients: Optional[ArrayLike],
        dis_coefficients: Optional[ArrayLike],
        y: Optional[ArrayLike],
    ) -> ArrayLike:
        """Call :meth:`~evaluate` or :meth:`~evaluate_rhs` depending on the presense of ``y``.

        Args:
            ham_coefficients: The signals :math:`c_1` to use on the Hamiltonians.
            dis_coefficients: The signals :math:`c_2` to use on the dissipators.
            y: Optionally, the system state.

        Returns:
            The evaluated function.
        """
        if y is None:
            return self.evaluate(ham_coefficients, dis_coefficients)

        return self.evaluate_rhs(ham_coefficients, dis_coefficients, y)

    def _convert_to_array_type(self, obj: Any) -> ArrayLike:
        return numpy_alias(like=self._array_library).asarray(obj)

    def _construct_operator_collection(self, *args, **kwargs):
        """The class used for evaluating the vectorized model or RHS."""
        return OperatorCollection(*args, **kwargs, array_library=self._array_library)

    def _concatenate_coefficients(self, ham_coefficients, dis_coefficients):
        if self._hamiltonian_operators is not None and self._dissipator_operators is not None:
            return _numpy_multi_dispatch(ham_coefficients, dis_coefficients, path="append", axis=-1)
        if self._hamiltonian_operators is not None and self._dissipator_operators is None:
            return ham_coefficients
        if self._hamiltonian_operators is None and self._dissipator_operators is not None:
            return dis_coefficients

        return None


class ScipySparseVectorizedLindbladCollection(VectorizedLindbladCollection):
    r"""Scipy sparse version of VectorizedLindbladCollection."""

    def __init__(
        self,
        static_hamiltonian: Optional[ArrayLike] = None,
        hamiltonian_operators: Optional[ArrayLike] = None,
        static_dissipators: Optional[ArrayLike] = None,
        dissipator_operators: Optional[ArrayLike] = None,
        decimals: Optional[int] = 10,
    ):
        r"""Initialize collection.

        Args:
            static_hamiltonian: Constant term :math:`H_d` to be added to the Hamiltonian of the
                                system.
            hamiltonian_operators: Specifies breakdown of Hamiltonian
                as :math:`H(t) = \sum_j s(t) H_j+H_d` by specifying H_j. (k,n,n) array.
            static_dissipators: Dissipator terms with coefficient 1.
            dissipator_operators: the terms :math:`L_j` in Lindblad equation. (m,n,n) array.
            decimals: Decimals to round the sparse operators to 0.
        """
        self._decimals = decimals
        super().__init__(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
        )

    def _convert_to_array_type(self, obj: any) -> Union[csr_matrix, List[csr_matrix]]:
        if obj is None:
            return None

        obj = to_csr(obj)
        if issparse(obj):
            return np.round(obj, decimals=self._decimals)
        else:
            return [np.round(sub_obj, decimals=self._decimals) for sub_obj in obj]

    def _construct_operator_collection(self, *args, **kwargs):
        """The class used for evaluating the vectorized model or RHS."""
        return ScipySparseOperatorCollection(*args, **kwargs)


def package_density_matrices(y: ArrayLike) -> ArrayLike:
    """Sends an array ``y`` of density matrices to a ``(1,)`` array of dtype object, where entry
    ``[0]`` is ``y``. Formally avoids for-loops through vectorization.

    Args:
        y: An array.
    Returns:
        Array with dtype object.
    """
    # As written here, only works for (n,n) Arrays
    obj_arr = np.empty(shape=(1,), dtype="O")
    obj_arr[0] = y
    return obj_arr


# Using vectorization with signature, works on (k,n,n) Arrays -> (k,1) Array
package_density_matrices = np.vectorize(package_density_matrices, signature="(n,n)->(1)")


def unpackage_density_matrices(y: ArrayLike) -> ArrayLike:
    """Inverse function of :func:`package_density_matrices`.

    Since this function is much slower than packaging, avoid it unless absolutely needed (as in case
    of passing multiple density matrices to :meth:`SparseLindbladCollection.evaluate_rhs`).

    Args:
        y: An array to extract the first element from.
    Returns:
        A ``(k,n,n)`` array.
    """
    return y[0]


unpackage_density_matrices = np.vectorize(unpackage_density_matrices, signature="(1)->(n,n)")
