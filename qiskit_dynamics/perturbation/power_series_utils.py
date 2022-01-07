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
# pylint: disable=invalid-name

r"""
Module for for power series/polynomial computation and manipulation.

Note that within this file, 'multiset' always refers to a multiset of
integers, which is always represented as a list of integers.
"""

from typing import List, Optional, Callable, Tuple
from copy import deepcopy
from itertools import combinations

import numpy as np

from qiskit_dynamics.array import Array

try:
    import jax.numpy as jnp
except ImportError:
    pass


class MatrixPolynomial:
    r"""A polynomial with matrix-valued coefficients.

    This class represents a multi-variable function of the form:

    .. math::
        f(c_1, \dots, c_r) = M_0 + \sum_{I \in S} c_I M_I,

    where in the above:

        - :math:`S` is a finite set of index multisets indicating non-zero monomial terms,
        - For a given index multiset :math:`I=(i_1, \dots, i_k)`,
          :math:`c_I = c_{i_1} \times \dots \times c_{i_k}`, and
        - The :math:`M_I` are matrices specified as arrays.

    At instantiation, the user specifies :math:`S` as a list of index multisets,
    :math:`M_I` as list of matrices (specified as a 3d array) whose first index has
    corresponding ordering with the list specifying :math:`S`, and can optionally
    specify a constant term :math:`M_0`.
    """

    def __init__(
        self,
        matrix_coefficients: Array,
        monomial_multisets: List[List[int]],
        constant_term: Optional[Array] = None,
    ):
        """Construct a multivariable matrix polynomial.

        Args:
            matrix_coefficients: A 3d array representing a list of matrix coefficients.
            monomial_multisets: A list of multisets of the same length as ``matrix_coefficients``
                                indicating the monomial coefficient for each corresponding
                                ``matrix_coefficients``.
            constant_term: A 2d array representing the constant term of the polynomial.
        """

        self._monomial_multisets = monomial_multisets

        # If operating in jax mode, wrap in Arrays
        if Array(matrix_coefficients).backend == "jax":
            self._matrix_coefficients = Array(matrix_coefficients)

            if constant_term is not None:
                constant_term = Array(constant_term)
            else:
                constant_term = np.zeros_like(Array(matrix_coefficients[0]))

            self._constant_term = constant_term
            self._compute_monomials = get_monomial_compute_function_jax(monomial_multisets)
        else:
            if constant_term is None:
                constant_term = np.zeros_like(matrix_coefficients[0])

            self._matrix_coefficients = matrix_coefficients
            self._constant_term = constant_term
            self._compute_monomials = get_monomial_compute_function(monomial_multisets)

    @property
    def monomial_multisets(self) -> List:
        """The monomial multisets corresponding to non-constant terms."""
        return self._monomial_multisets

    @property
    def matrix_coefficients(self) -> Array:
        """The matrix coefficients for non-constant terms."""
        return self._matrix_coefficients

    @property
    def constant_term(self) -> Array:
        """The constant term."""
        return self._constant_term

    def compute_monomials(self, c: Array) -> Array:
        """Vectorized computation of all scalar monomial terms in the polynomial as specified by
        ``self.monomial_multisets``.

        Args:
            c: Array of variables.
        Returns:
            Array of all monomial terms ordered according to ``self.monomial_multisets``.
        """
        return self._compute_monomials(c)

    def conj(self) -> "MatrixPolynomial":
        """Returns polynomial attained by conjugating all coefficients."""
        return MatrixPolynomial(
            matrix_coefficients=np.conj(self._matrix_coefficients),
            monomial_multisets=self._monomial_multisets,
        )

    def transpose(self) -> "MatrixPolynomial":
        """Returns polynomial attained by transposing all coefficients."""
        return MatrixPolynomial(
            matrix_coefficients=self._matrix_coefficients.transpose((0, 2, 1)),
            monomial_multisets=self._monomial_multisets,
        )

    def adjoint(self) -> "MatrixPolynomial":
        """Returns polynomial attained by taking the adjoint of all coefficients."""
        return MatrixPolynomial(
            matrix_coefficients=np.conj(self._matrix_coefficients.transpose((0, 2, 1))),
            monomial_multisets=self._monomial_multisets,
        )

    def __call__(self, c: Array) -> Array:
        """Evaluate the polynomial.

        Args:
            c: Array of variables.
        Returns:
            Value of the polynomial at c.
        """
        monomials = self._compute_monomials(c)
        return self._constant_term + np.tensordot(self._matrix_coefficients, monomials, axes=(0, 0))


def get_monomial_compute_function(multiset_list: List) -> Callable:
    """Construct a vectorized function for computing multivariable monomial terms indicated by
    multiset_list. I.e. Each multiset is a list of indices, and a given index list [i1, ..., ik]
    indicates the monomial term c[i1] * ... * c[ik]. This function assumes:

        - Every multiset in multiset_list is sorted.

    The returned function takes in the individual variables as an array, and returns an array
    of computed monomial terms in the order indicated by multiset_list.

    The returned function is vectorized in the sense that the supplied first order terms can be
    arrays.

    The algorithm computes monomial terms of increasing order, recursively utilizing lower
    order terms.

    Args:
        multiset_list: list of multisets.

    Returns:
        Callable: Vectorized function for computing monomials.
    """
    complete_multiset_list = get_complete_index_multisets(multiset_list)

    (
        first_order_terms,
        first_order_range,
        left_indices,
        right_indices,
        update_ranges,
    ) = get_recursive_monomial_rule(complete_multiset_list)

    multiset_len = len(complete_multiset_list)

    location_list = np.array(
        [complete_multiset_list.index(multiset) for multiset in multiset_list], dtype=int
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


def get_monomial_compute_function_jax(multiset_list: List) -> Callable:
    """JAX version of get_monomial_compute_function."""

    complete_multiset_list = get_complete_index_multisets(multiset_list)

    first_order_terms, _, left_indices, right_indices, _ = get_recursive_monomial_rule(
        complete_multiset_list
    )
    location_list = np.array(
        [complete_multiset_list.index(multiset) for multiset in multiset_list], dtype=int
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


def get_recursive_monomial_rule(multiset_list: List) -> Tuple:
    """Helper function for get_monomial_compute_function and get_monomial_compute_function_jax;
    computes a representation of the algorithm for computing monomials that is used by both
    functions.

    Args:
        multiset_list: Description of monomial terms.

    Returns:
        Tuple: Collection of lists organizing computation for both
        get_monomial_compute_function and get_monomial_compute_function_jax.
    """

    # first, construct representation of recursive rule explicitly in terms of multisets
    first_order_terms = []
    left_terms = []
    right_terms = []
    current_left = -1
    current_right_list = []
    current_len = 2

    for multiset in multiset_list:
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
        return first_order_terms, [0, len(first_order_terms)], [], [], []

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
        left_indices.append(multiset_list.index([left_term]))

        right_index_list = []
        for term in right_term:
            right_index_list.append(multiset_list.index(term))

        right_indices.append(np.array(right_index_list, dtype=int))

    # set up index updating ranges
    first_order_range = [0, len(first_order_terms)]
    update_ranges = []
    current_idx = first_order_range[1]
    for right_index in right_indices:
        next_idx = current_idx + len(right_index)
        update_ranges.append([current_idx, next_idx])
        current_idx = next_idx

    return first_order_terms, first_order_range, left_indices, right_indices, update_ranges


def get_complete_index_multisets(index_multisets: List) -> List:
    """Given a list of index multisets, return its "completion", in the sense of
    returning a list of all index multisets achievable by taking subsets of the multisets
    in the original list.

    This list is sorted in the canonical way: for index multisets I, J, I < J iff:
        - len(I) < len(J) or
        - len(I) == len(J) and I < J when viewed as strings

    Args:
        index_multisets: List of index multisets (not necessarilly correctly formatted).

    Returns:
        List: Complete list of index multisets generated by the argument.
    """

    if index_multisets == []:
        return []

    # clean list to unique list of properly formatted terms
    index_multisets = clean_index_multisets(index_multisets)

    max_order = max(map(len, index_multisets))

    # partition according to order
    term_dict = {k: [] for k in range(1, max_order + 1)}
    for term in index_multisets:
        order = len(term)
        if term not in term_dict[order]:
            term_dict[order].append(list(term))

    # loop through orders in reverse order, adding subterms to lower levels if necessary
    for order in range(max_order, 1, -1):

        for sym_term in term_dict[order]:
            depending_terms = submultisets_and_complements(sym_term, 2)[1]

            for dep_term in depending_terms:
                if dep_term not in term_dict[order - 1]:
                    term_dict[order - 1].append(dep_term)

    complete_terms = []
    for term_list in term_dict.values():
        complete_terms += term_list

    # sort in terms of increasing length and lexicographic order
    complete_terms.sort(key=str)
    complete_terms.sort(key=len)

    return complete_terms


def clean_index_multisets(index_multisets: List) -> List:
    """Given a list of index multisets, put them into canonical non-decreasing
    order, and eliminate any duplicates.

    Args:
        index_multisets: List of index multisets.

    Returns:
        List
    """

    ordered_unique_terms = []
    for term in index_multisets:
        # deep copy to not modify original list, which may be user supplied
        sorted_copy = list(deepcopy(term))
        sorted_copy.sort()
        if sorted_copy not in ordered_unique_terms:
            ordered_unique_terms.append(sorted_copy)

    return ordered_unique_terms


def submultisets_and_complements(
    index_multiset: List[int], subset_bound: Optional[int] = None
) -> List:
    """Given a multiset of indices specified as a list if ints, compute a pair of lists
    containing all submultisets and their corresponding complements.

    Note: does not include the empty set in the subsets, and the default behaviour is to not
    include the full set either.

    Args:
        index_multiset: Index multiset specified as a list of ints.
        subset_bound: Strict upper bound on subset size to consider.
                      Defaults to len(index_multiset).

    Returns:
        List, List: submultisets and corresponding complements
    """

    if subset_bound is None:
        subset_bound = len(index_multiset)

    submultisets = []
    complements = []

    for k in range(1, subset_bound):
        location_subsets = combinations(range(len(index_multiset)), k)
        for location_subset in location_subsets:
            subset = []
            complement = []

            for loc, multiset_entry in enumerate(index_multiset):
                if loc in location_subset:
                    subset.append(multiset_entry)
                else:
                    complement.append(multiset_entry)

            if subset not in submultisets:
                submultisets.append(subset)
                complements.append(complement)

    return submultisets, complements


def is_submultiset(A: List[int], B: List[int]) -> bool:
    """Check if A is a submultiset of B, where A and B are specified as lists of ints."""

    # get the unique elements of A
    A_set = set(A)

    for elem in A_set:
        if A.count(elem) > B.count(elem):
            return False

    return True


def multiset_complement(A: List[int], B: List[int]) -> bool:
    r"""Compute the multiset complement B\A, where A and B are specified as lists of ints."""

    # get the unique elements of B
    B_set = list(set(B))
    B_set.sort()

    complement = []
    for elem in B_set:
        diff = B.count(elem) - A.count(elem)
        if diff > 0:
            complement += [elem] * diff

    return complement


def submultiset_filter(multiset_candidates: List, multiset_list: List) -> List:
    """Filter the list of multiset_candidates based on whether they are a
    submultiset of an element in multiset_list.
    """

    filtered_multisets = []
    for candidate in multiset_candidates:
        for multiset in multiset_list:
            if is_submultiset(candidate, multiset):
                filtered_multisets.append(candidate)
                break

    return filtered_multisets
