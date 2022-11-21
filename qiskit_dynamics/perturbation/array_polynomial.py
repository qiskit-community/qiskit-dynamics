# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Array polynomial.
"""

from typing import List, Optional, Callable, Tuple, Union
from copy import copy
from itertools import product

from multiset import Multiset

import numpy as np
from numpy.typing import DTypeLike

from qiskit import QiskitError

from qiskit_dynamics.array import Array
from qiskit_dynamics.perturbation.multiset_utils import (
    _validate_non_negative_ints,
    _get_all_submultisets,
    _sorted_multisets,
    _submultisets_and_complements,
    _multiset_to_sorted_list,
)
from qiskit_dynamics.perturbation.custom_binary_op import _CustomBinaryOp

try:
    import jax.numpy as jnp
except ImportError:
    pass


class ArrayPolynomial:
    r"""A polynomial with array-valued coefficients.

    This class represents a multi-variable function of the form:

    .. math::
        f(c_1, \dots, c_r) = A_\emptyset + \sum_{I \in S} c_I A_I,

    where in the above:

        - :math:`S` is a finite set of multisets
          indicating non-zero monomial terms,
        - For a given multiset of non-negative integers :math:`I=(i_1, \dots, i_k)`,
          :math:`c_I = c_{i_1} \times \dots \times c_{i_k}`, and
        - The :math:`A_I` are arrays of the same shape, indexed by the first dimension.

    See the :ref:`multiset and power series notation section <multiset power series>`
    of the perturbation review for an explanation of the multiset notation.

    An :class:`.ArrayPolynomial` is instantiated with the arguments:

        - ``constant_term`` specifying the array :math:`A_\emptyset`.
        - ``array_coefficients`` specifying a list of the arrays :math:`A_I`, or as a single array
          whose first index lists the :math:`A_I`,
        - ``monomial_labels`` specifying the set :math:`S` as a list of
          ``Multiset`` instances ordered in
          correspondence with ``array_coefficients``.

    For example, the :class:`.ArrayPolynomial` corresponding to the mathematical polynomial

    .. math::

        f(c_0, c_1) = A_\emptyset
            + c_{(0)} A_{(0)} + c_{(0, 1)}A_{(0, 1)} + c_{(1, 1)}A_{(1, 1)}

    for arrays :math:`A_\emptyset, A_{(0)}, A_{(0, 1)}, A_{(1, 1)}` stored in variables
    ``A_c``, ``A0``, ``A01``, and ``A11`` can be instantiated with

    .. code-block:: python

        ap = ArrayPolynomial(
            constant_term = A_c
            array_coefficients=[A0, A01, A11],
            monomial_labels=[Multiset({0: 1}), Multiset({0: 1, 1: 1}), Multiset({1: 2})]
        )

    Once instantiated, the polynomial can be evaluated on an array of variable values, e.g.

    .. code-block:: python

        c = np.array([c0, c1])
        ap(c) # polynomial evaluated on variables

    :class:`.ArrayPolynomial` supports some array properties, e.g. ``ap.shape`` and ``ap.ndim``
    return the shape and number of dimensions of the output of the polynomial. Some array
    methods are also supported, such as ``transpose`` and ``trace``, and their output produces
    a new :class:`.ArrayPolynomial` which evaluates to the array one would obtain by first
    evaluating the original, then calling the array method. E.g.

    .. code-block:: python

        ap2 = ap1.transpose()
        ap2(c) == ap1(c).transpose()

    Finally, :class:`.ArrayPolynomial` supports algebraic operations, e.g.

    .. code-block:: python

        ap3 = ap1 @ ap2
        ap3(c) == ap1(c) @ ap2(c)

    It also has specialized algebraic methods that perform algebraic operations while
    "ignoring" terms. E.g., for two instances ``ap1`` and ``ap2``, the call

    .. code-block:: python

        ap1.matmul(ap2, monomial_filter=lambda x: len(x) <= 3)

    is similar to ``ap1 @ ap2``, but will result in an :class:`.ArrayPolynomial` in which all
    terms of degree larger than ``3`` will not be included in the results.
    """
    __array_priority__ = 20

    def __init__(
        self,
        constant_term: Optional[Array] = None,
        array_coefficients: Optional[Array] = None,
        monomial_labels: Optional[List[Multiset]] = None,
    ):
        """Construct a multivariable matrix polynomial.

        Args:
            constant_term: An array representing the constant term of the polynomial.
            array_coefficients: A 3d array representing a list of array coefficients.
            monomial_labels: A list of multisets with non-negative integer entries of the same
                             length as ``array_coefficients`` indicating the monomial coefficient
                             for each corresponding ``array_coefficients``.
        Raises:
            QiskitError: If insufficient information is supplied to define an ArrayPolynomial,
                         or if monomial labels contain anything other than non-negative integers.
        """

        if array_coefficients is None and constant_term is None:
            raise QiskitError(
                "At least one of array_coefficients and constant_term must be specified."
            )

        if monomial_labels is not None:
            self._monomial_labels = [Multiset(m) for m in monomial_labels]
            for m in self._monomial_labels:
                _validate_non_negative_ints(m)
        else:
            self._monomial_labels = []

        # If operating in jax mode, wrap in Arrays
        if Array.default_backend() == "jax":

            if array_coefficients is not None:
                self._array_coefficients = Array(array_coefficients)
            else:
                self._array_coefficients = None

            if constant_term is not None:
                self._constant_term = Array(constant_term)
            else:
                self._constant_term = None

            self._compute_monomials = _get_monomial_compute_function_jax(self._monomial_labels)
        else:
            if constant_term is not None:
                self._constant_term = np.array(constant_term)
            else:
                self._constant_term = None

            if array_coefficients is not None:
                self._array_coefficients = np.array(array_coefficients)
            else:
                self._array_coefficients = None

            self._compute_monomials = _get_monomial_compute_function(self._monomial_labels)

    @property
    def monomial_labels(self) -> Union[List, None]:
        """The monomial labels corresponding to non-constant terms."""
        return self._monomial_labels

    @property
    def array_coefficients(self) -> Union[Array, None]:
        """The array coefficients for non-constant terms."""
        return self._array_coefficients

    @property
    def constant_term(self) -> Union[Array, None]:
        """The constant term."""
        return self._constant_term

    def compute_monomials(self, c: Array) -> Union[Array, None]:
        """Vectorized computation of all scalar monomial terms in the polynomial as specified by
        ``self.monomial_labels``.

        Args:
            c: Array of variables.
        Returns:
            Array of all monomial terms ordered according to ``self.monomial_labels``.
        """
        return self._compute_monomials(c)

    @property
    def shape(self) -> Tuple[int]:
        """Shape of the arrays in the polynomial."""
        if self._constant_term is not None:
            return self._constant_term.shape
        else:
            return self._array_coefficients.shape[1:]

    @property
    def ndim(self) -> int:
        """Number of dimensions of the coefficients of the polynomial."""
        if self._constant_term is not None:
            return self._constant_term.ndim
        else:
            return self._array_coefficients.ndim - 1

    def conj(self) -> "ArrayPolynomial":
        """Return an ArrayPolynomial that is the conjugate of self."""

        constant_term = None
        coefficients = None

        if self._constant_term is not None:
            constant_term = np.conj(self._constant_term)

        if self._array_coefficients is not None:
            coefficients = np.conj(self._array_coefficients)

        return ArrayPolynomial(
            array_coefficients=coefficients,
            monomial_labels=copy(self._monomial_labels),
            constant_term=constant_term,
        )

    def transpose(self, axes: Optional[Tuple[int]] = None) -> "ArrayPolynomial":
        """Return the ArrayPolynomial when transposing all coefficients."""

        constant_term = None
        coefficients = None

        if self._constant_term is not None:
            constant_term = np.transpose(self._constant_term, axes)

        if self._array_coefficients is not None:
            if axes is None:
                axes = tuple(range(1, self.ndim + 1)[::-1])
            else:
                axes = tuple(ax + 1 for ax in axes)
            axes = (0,) + axes
            coefficients = np.transpose(self._array_coefficients, axes)

        return ArrayPolynomial(
            array_coefficients=coefficients,
            monomial_labels=copy(self._monomial_labels),
            constant_term=constant_term,
        )

    def trace(
        self,
        offset: Optional[int] = 0,
        axis1: Optional[int] = 0,
        axis2: Optional[int] = 1,
        dtype: Optional[DTypeLike] = None,
    ) -> "ArrayPolynomial":
        """Take the trace of the coefficients."""

        if self.ndim < 2:
            raise QiskitError("ArrayPolynomial.trace() requires ArrayPolynomial.ndim at least 2.")

        constant_term = None
        coefficients = None

        if self._constant_term is not None:
            constant_term = np.trace(
                self._constant_term, offset=offset, axis1=axis1, axis2=axis2, dtype=dtype
            )

        if self._array_coefficients is not None:
            coefficients = np.trace(
                self._array_coefficients,
                offset=offset,
                axis1=axis1 + 1,
                axis2=axis2 + 1,
                dtype=dtype,
            )

        return ArrayPolynomial(
            array_coefficients=coefficients,
            monomial_labels=copy(self._monomial_labels),
            constant_term=constant_term,
        )

    def sum(self, axis: Optional[Union[int, Tuple[int]]] = None, dtype: Optional[DTypeLike] = None):
        """Perform a sum on the coefficients."""

        constant_term = None
        coefficients = None

        # constant term can be handled normally
        if self.constant_term is not None:
            constant_term = self.constant_term.sum(axis=axis, dtype=dtype)

        # axis must be shifted for array coefficients
        if self.array_coefficients is not None:

            if self.ndim == 0 and axis is None:
                coefficients = np.array(self.array_coefficients, dtype=dtype)
            else:
                if axis is None:
                    axis = tuple(k for k in range(1, self.ndim + 1))
                elif isinstance(axis, int):
                    axis = axis + 1
                elif isinstance(axis, tuple):
                    axis = tuple(k + 1 for k in axis)

                coefficients = self.array_coefficients.sum(axis=axis, dtype=dtype)

        return ArrayPolynomial(
            array_coefficients=coefficients,
            monomial_labels=copy(self._monomial_labels),
            constant_term=constant_term,
        )

    @property
    def real(self) -> "ArrayPolynomial":
        """Return the real part of self."""

        constant_term = None
        array_coefficients = None

        if self.constant_term is not None:
            constant_term = self.constant_term.real

        if self.array_coefficients is not None:
            array_coefficients = self.array_coefficients.real

        return ArrayPolynomial(
            array_coefficients=array_coefficients,
            monomial_labels=copy(self._monomial_labels),
            constant_term=constant_term,
        )

    def add(
        self,
        other: Union["ArrayPolynomial", int, float, complex, Array],
        monomial_filter: Optional[Callable] = None,
    ) -> "ArrayPolynomial":
        """Add two polynomials with bounds on which terms to keep.

        Optionally, a function ``monomial_filter`` can be provided to limit which monomials
        appear in the output. It must accept as input a ``Multiset`` and return a ``bool``,
        and a term with label given by ``multiset`` will be included only if
        ``monomial_filter(multiset) == True``, and will not be computed if
        ``monomial_filter(multiset) == False``.

        Args:
            other: Other to add to self.
            monomial_filter: Function determining which terms to compute and keep.
        Returns:
            ArrayPolynomial achieved by adding both self and other.
        Raises:
            QiskitError: if other cannot be cast as an ArrayPolynomial.
        """

        if isinstance(other, (int, float, complex, np.ndarray, Array)):
            other = ArrayPolynomial(constant_term=other)

        if isinstance(other, ArrayPolynomial):
            return _array_polynomial_addition(self, other, monomial_filter=monomial_filter)

        raise QiskitError(
            "Only types castable as an ArrayPolynomial can be added to an ArrayPolynomial."
        )

    def matmul(
        self,
        other: Union["ArrayPolynomial", int, float, complex, np.ndarray, Array],
        monomial_filter: Optional[Callable] = None,
    ) -> "ArrayPolynomial":
        """Matmul self @ other with bounds on which terms to keep.

        Optionally, a function ``monomial_filter`` can be provided to limit which monomials
        appear in the output. It must accept as input a ``Multiset`` and return a ``bool``,
        and a term with label given by ``multiset`` will be included only if
        ``monomial_filter(multiset) == True``, and will not be computed if
        ``monomial_filter(multiset) == False``.

        Args:
            other: Other to add to self.
            monomial_filter: Function determining which terms to compute and keep.
        Returns:
            ArrayPolynomial achieved by matmul of self and other.
        Raises:
            QiskitError: if other cannot be cast as an ArrayPolynomial.
        """
        if isinstance(other, (int, float, complex, np.ndarray, Array)):
            other = ArrayPolynomial(constant_term=other)

        if isinstance(other, ArrayPolynomial):
            return _array_polynomial_distributive_binary_op(
                self, other, lambda A, B: A @ B, monomial_filter=monomial_filter
            )

        raise QiskitError(f"Type {type(other)} not supported by ArrayPolynomial.matmul.")

    def mul(
        self,
        other: Union["ArrayPolynomial", int, float, complex, np.ndarray, Array],
        monomial_filter: Optional[Callable] = None,
    ) -> "ArrayPolynomial":
        """Entrywise multiplication of two ArrayPolynomials with bounds on which terms to keep.

        Optionally, a function ``monomial_filter`` can be provided to limit which monomials
        appear in the output. It must accept as input a ``Multiset`` and return a ``bool``,
        and a term with label given by ``multiset`` will be included only if
        ``monomial_filter(multiset) == True``, and will not be computed if
        ``monomial_filter(multiset) == False``.

        Args:
            other: Other to add to self.
            monomial_filter: Function determining which terms to compute and keep.
        Returns:
            ArrayPolynomial achieved by matmul of self and other.
        Raises:
            QiskitError: if other cannot be cast as an ArrayPolynomial.
        """

        if isinstance(other, (int, float, complex, np.ndarray, Array)):
            other = ArrayPolynomial(constant_term=other)

        if isinstance(other, ArrayPolynomial):
            return _array_polynomial_distributive_binary_op(
                self, other, lambda A, B: A * B, monomial_filter=monomial_filter
            )

        raise QiskitError(f"Type {type(other)} not supported by ArrayPolynomial.mul.")

    def __add__(
        self, other: Union["ArrayPolynomial", int, float, complex, Array]
    ) -> "ArrayPolynomial":
        """Dunder method for addition of two ArrayPolynomials."""
        return self.add(other)

    def __radd__(
        self, other: Union["ArrayPolynomial", int, float, complex, Array]
    ) -> "ArrayPolynomial":
        """Dunder method for right-addition of two ArrayPolynomials."""
        return self.add(other)

    def __neg__(self) -> "ArrayPolynomial":
        constant_term = None
        if self.constant_term is not None:
            # pylint: disable=invalid-unary-operand-type
            constant_term = -self.constant_term

        array_coefficients = None
        if self.array_coefficients is not None:
            # pylint: disable=invalid-unary-operand-type
            array_coefficients = -self.array_coefficients

        return ArrayPolynomial(
            constant_term=constant_term,
            monomial_labels=self.monomial_labels,
            array_coefficients=array_coefficients,
        )

    def __sub__(
        self, other: Union["ArrayPolynomial", int, float, complex, Array]
    ) -> "ArrayPolynomial":
        return self + (-other)

    def __rsub__(
        self, other: Union["ArrayPolynomial", int, float, complex, Array]
    ) -> "ArrayPolynomial":
        return other + (-self)

    def __mul__(
        self, other: Union["ArrayPolynomial", int, float, complex, Array]
    ) -> "ArrayPolynomial":
        """Dunder method for entry-wise multiplication."""
        return self.mul(other)

    def __rmul__(
        self, other: Union["ArrayPolynomial", int, float, complex, Array]
    ) -> "ArrayPolynomial":
        """Dunder method for right-multiplication."""
        return self.mul(other)

    def __matmul__(self, other: Union["ArrayPolynomial", Array]) -> "ArrayPolynomial":
        """Dunder method for matmul."""
        return self.matmul(other)

    def __rmatmul__(self, other: Union["ArrayPolynomial", Array]) -> "ArrayPolynomial":
        """Dunder method for rmatmul."""
        if isinstance(other, (int, float, complex, np.ndarray, Array)):
            other = ArrayPolynomial(constant_term=other)

        if isinstance(other, ArrayPolynomial):
            return other.matmul(self)

        raise QiskitError(f"Type {type(other)} not supported by ArrayPolynomial.__rmatmul__.")

    def __getitem__(self, idx) -> "ArrayPolynomial":
        """Index the ArrayPolynomial similarly to an array."""
        constant_term = None
        array_coefficients = None

        if self.constant_term is not None:
            constant_term = self.constant_term[idx]

        if self.array_coefficients is not None:
            array_coefficients = self.array_coefficients[(slice(None),) + idx]

        return ArrayPolynomial(
            array_coefficients=array_coefficients,
            monomial_labels=copy(self._monomial_labels),
            constant_term=constant_term,
        )

    def __len__(self) -> int:
        """Number of terms in the polynomial."""
        num_terms = 0
        if self.array_coefficients is not None:
            num_terms = num_terms + len(self.array_coefficients)

        if self.constant_term is not None:
            num_terms = num_terms + 1

        return num_terms

    def __call__(self, c: Optional[Array] = None) -> Array:
        """Evaluate the polynomial.

        Args:
            c: Array of variables.
        Returns:
            Value of the polynomial at c.
        """

        if self._array_coefficients is not None and self._constant_term is not None:
            monomials = self._compute_monomials(c)
            return self._constant_term + np.tensordot(
                self._array_coefficients, monomials, axes=(0, 0)
            )
        elif self._array_coefficients is not None:
            monomials = self._compute_monomials(c)
            return np.tensordot(self._array_coefficients, monomials, axes=(0, 0))
        else:
            return self._constant_term


def _get_monomial_compute_function(multisets: List[Multiset]) -> Callable:
    """Construct a vectorized function for computing multivariable monomial terms indicated by
    multisets.

    The returned function takes in the individual variables as an array, and returns an array
    of computed monomial terms in the order indicated by multisets.

    The returned function is vectorized in the sense that the supplied first order terms can be
    arrays.

    The algorithm computes monomial terms of increasing order, recursively utilizing lower
    order terms.

    Args:
        multisets: list of multisets.

    Returns:
        Callable: Vectorized function for computing monomials.
    """
    if multisets is None or len(multisets) == 0:
        return lambda c: None

    complete_multiset_list = _get_all_submultisets(multisets)

    (
        first_order_terms,
        first_order_range,
        left_indices,
        right_indices,
        update_ranges,
    ) = _get_recursive_monomial_rule(complete_multiset_list)

    multiset_len = len(complete_multiset_list)

    location_list = np.array(
        [complete_multiset_list.index(multiset) for multiset in multisets], dtype=int
    )

    def monomial_function(c):
        shape = [multiset_len] + list(c.shape[1:])
        mono_vec = np.empty(shape=shape, dtype=complex)
        mono_vec[first_order_range[0] : first_order_range[1]] = c[first_order_terms]

        for left_index, right_index, update_range in zip(
            left_indices, right_indices, update_ranges
        ):
            mono_vec[update_range[0] : update_range[1]] = (
                mono_vec[left_index] * mono_vec[right_index]
            )

        return mono_vec[location_list]

    return monomial_function


# pylint: disable=invalid-name
def _get_monomial_compute_function_jax(multisets: List) -> Callable:
    """JAX version of _get_monomial_compute_function."""

    if multisets is None or len(multisets) == 0:
        return lambda c: None

    complete_multiset_list = _get_all_submultisets(multisets)

    first_order_terms, _, left_indices, right_indices, _ = _get_recursive_monomial_rule(
        complete_multiset_list
    )
    location_list = np.array(
        [complete_multiset_list.index(multiset) for multiset in multisets], dtype=int
    )

    # initial function sets up required first order terms
    def monomial_function_init(c):
        return c[first_order_terms]

    monomial_function = monomial_function_init

    # function for generating next update and compositions to avoid looping reference issues
    def get_next_update_func(left_index, right_index):
        def update_next(mono_vec):
            return jnp.append(mono_vec, mono_vec[left_index] * mono_vec[right_index], axis=0)

        return update_next

    def compose_functions(f, g):
        def new_func(x):
            return g(f(x))

        return new_func

    # recursively compose updates with monomial_function
    for left_index, right_index in zip(left_indices, right_indices):
        next_update = get_next_update_func(left_index, right_index)
        monomial_function = compose_functions(monomial_function, next_update)

    # return only the requested terms
    def trimmed_output_function(c):
        return monomial_function(c)[location_list]

    return trimmed_output_function


def _get_recursive_monomial_rule(complete_multisets: List) -> Tuple:
    """Helper function for _get_monomial_compute_function and _get_monomial_compute_function_jax;
    computes a representation of the algorithm for computing monomials that is used by both
    functions.

    complete_multisets is assumed to be closed under taking submultisets and in canonical order.

    Args:
        complete_multisets: Description of monomial terms.

    Returns:
        Tuple: Collection of lists organizing computation for both
        _get_monomial_compute_function and _get_monomial_compute_function_jax.
    """

    # first, construct representation of recursive rule explicitly in terms of multisets
    first_order_terms = []
    left_terms = []
    right_terms = []
    current_left = -1
    current_right_list = []
    current_len = 2

    # convert multisets to list representation
    complete_multisets = [_multiset_to_sorted_list(multiset) for multiset in complete_multisets]

    for multiset in complete_multisets:
        if len(multiset) == 1:
            first_order_terms.append(multiset[0])
        else:
            if multiset[0] != current_left or len(multiset) != current_len:
                current_len = len(multiset)
                if current_left != -1:
                    left_terms.append(current_left)
                    right_terms.append(current_right_list)

                current_left = multiset[0]
                current_right_list = [multiset[1:]]
            else:
                current_right_list.append(multiset[1:])

    # if current_left is still -1, then only first order terms exist
    if current_left == -1:
        return np.array(first_order_terms), [0, len(first_order_terms)], [], [], []

    # proceed assuming at least one term above first order exists
    # append the last one
    left_terms.append(current_left)
    right_terms.append(current_right_list)

    # convert representation of rule in terms of multisets into one in terms of
    # array indices

    # set up arrays
    first_order_terms = np.array(first_order_terms, dtype=int)

    left_indices = []
    right_indices = []
    for left_term, right_term in zip(left_terms, right_terms):
        left_indices.append(complete_multisets.index([left_term]))

        right_index_list = []
        for term in right_term:
            right_index_list.append(complete_multisets.index(term))

        right_indices.append(np.array(right_index_list, dtype=int))

    # set up index updating ranges
    first_order_range = [0, len(first_order_terms)]
    update_ranges = []
    current_idx = first_order_range[1]
    for right_index in right_indices:
        next_idx = current_idx + len(right_index)
        update_ranges.append([current_idx, next_idx])
        current_idx = next_idx

    return (
        np.array(first_order_terms),
        first_order_range,
        np.array(left_indices),
        right_indices,
        update_ranges,
    )


def _array_polynomial_distributive_binary_op(
    ap1: ArrayPolynomial,
    ap2: ArrayPolynomial,
    binary_op: Callable,
    monomial_filter: Optional[Callable] = None,
) -> ArrayPolynomial:
    """Apply a distributive binary op on two array polynomials."""

    # determine list of Multisets required for monomial labels, including filtering
    all_multisets = []

    # if no filter is provided, set to always return True
    if monomial_filter is None:
        monomial_filter = lambda x: True

    if ap1.constant_term is not None:
        for multiset in ap2.monomial_labels:
            if monomial_filter(multiset) and multiset not in all_multisets:
                all_multisets.append(multiset)
    if ap2.constant_term is not None:
        for multiset in ap1.monomial_labels:
            if monomial_filter(multiset) and multiset not in all_multisets:
                all_multisets.append(multiset)

    for I, J in product(ap1.monomial_labels, ap2.monomial_labels):
        IuJ = I + J
        if monomial_filter(IuJ) and IuJ not in all_multisets:
            all_multisets.append(IuJ)

    all_multisets = _sorted_multisets(all_multisets)

    # setup constant term
    new_constant_term = None
    if (
        ap1.constant_term is not None
        and ap2.constant_term is not None
        and monomial_filter(Multiset({}))
    ):
        new_constant_term = binary_op(ap1.constant_term, ap2.constant_term)

    # return constant case
    if not all_multisets:
        return ArrayPolynomial(constant_term=new_constant_term)

    # iteratively construct custom multiplication rule,
    # temporarily treating the constant terms as index -1
    operation_rule = []
    for multiset in all_multisets:
        rule_indices = []

        if multiset in ap1.monomial_labels:
            idx = ap1.monomial_labels.index(multiset)
            rule_indices.append([idx, -1])

        if multiset in ap2.monomial_labels:
            idx = ap2.monomial_labels.index(multiset)
            rule_indices.append([-1, idx])

        if len(multiset) > 1:
            for I, J in zip(*_submultisets_and_complements(multiset)):
                if I in ap1.monomial_labels and J in ap2.monomial_labels:
                    rule_indices.append(
                        [ap1.monomial_labels.index(I), ap2.monomial_labels.index(J)]
                    )

        # if non-empty,
        if rule_indices:
            operation_rule.append((np.ones(len(rule_indices)), np.array(rule_indices)))

    lmats = None
    if ap1.constant_term is not None:
        lmats = np.expand_dims(ap1.constant_term, 0)
    else:
        lmats = np.expand_dims(np.zeros_like(Array(ap1.array_coefficients[0])), 0)

    if ap1.array_coefficients is not None:
        lmats = np.append(lmats, ap1.array_coefficients, axis=0)

    rmats = None
    if ap2.constant_term is not None:
        rmats = np.expand_dims(ap2.constant_term, 0)
    else:
        rmats = np.expand_dims(np.zeros_like(Array(ap2.array_coefficients[0])), 0)

    if ap2.array_coefficients is not None:
        rmats = np.append(rmats, ap2.array_coefficients, axis=0)

    custom_binary_op = _CustomBinaryOp(
        operation_rule=operation_rule,
        binary_op=binary_op,
        index_offset=1,
    )
    new_array_coefficients = custom_binary_op(lmats, rmats)

    return ArrayPolynomial(
        array_coefficients=new_array_coefficients,
        monomial_labels=all_multisets,
        constant_term=new_constant_term,
    )


def _array_polynomial_addition(
    ap1: ArrayPolynomial,
    ap2: ArrayPolynomial,
    monomial_filter: Optional[Callable] = None,
) -> ArrayPolynomial:
    """Add two ArrayPolynomials."""

    for a, b in zip(ap1.shape[::-1], ap2.shape[::-1]):
        if not (a == 1 or b == 1 or a == b):
            raise QiskitError(
                "ArrayPolynomial addition requires shapes be broadcastable to eachother."
            )

    if monomial_filter is None:
        monomial_filter = lambda x: True

    # construct constant term
    new_constant_term = None

    # if constant term is to be included, determine what it is
    if monomial_filter(Multiset({})):
        if ap1.constant_term is not None and ap2.constant_term is not None:
            new_constant_term = ap1.constant_term + ap2.constant_term
        elif ap1.constant_term is not None:
            new_constant_term = ap1.constant_term
        elif ap2.constant_term is not None:
            new_constant_term = ap2.constant_term

    # exit early if both polynomials are constant
    if ap1.array_coefficients is None and ap2.array_coefficients is None:
        return ArrayPolynomial(constant_term=new_constant_term)

    # construct list of admissable multisets and sort into canonical order
    new_multisets = []
    for multiset in ap1.monomial_labels + ap2.monomial_labels:
        if monomial_filter(multiset) and multiset not in new_multisets:
            new_multisets.append(multiset)
    new_multisets = _sorted_multisets(new_multisets)

    # construct index order mapping for each polynomial
    idx1 = []
    idx2 = []
    for multiset in new_multisets:
        if multiset in ap1.monomial_labels:
            idx1.append(ap1.monomial_labels.index(multiset))
        else:
            idx1.append(-1)

        if multiset in ap2.monomial_labels:
            idx2.append(ap2.monomial_labels.index(multiset))
        else:
            idx2.append(-1)

    # if either is empty, pad with single -1, then convert to array
    idx1 = idx1 or [-1]
    idx2 = idx2 or [-1]
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # append zero to the coefficient arrays
    array_coefficients1 = np.zeros((1,) + ap1.shape, dtype=complex)
    array_coefficients2 = np.zeros((1,) + ap1.shape, dtype=complex)
    if ap1.array_coefficients is not None:
        array_coefficients1 = np.append(ap1.array_coefficients, array_coefficients1, axis=0)
    if ap2.array_coefficients is not None:
        array_coefficients2 = np.append(ap2.array_coefficients, array_coefficients2, axis=0)

    new_coefficients = array_coefficients1[idx1] + array_coefficients2[idx2]

    return ArrayPolynomial(
        array_coefficients=new_coefficients,
        monomial_labels=new_multisets,
        constant_term=new_constant_term,
    )
