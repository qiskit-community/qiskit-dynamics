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

from typing import List, Optional, Callable, Tuple
from copy import deepcopy
from itertools import combinations

import numpy as np

from qiskit_dynamics.array import Array

from qiskit_dynamics.perturbation import Multiset
from qiskit_dynamics.perturbation.multiset import get_all_submultisets

try:
    import jax.numpy as jnp
except ImportError:
    pass


class ArrayPolynomial:
    r"""A polynomial with array-valued coefficients.

    This class represents a multi-variable function of the form:

    .. math::
        f(c_1, \dots, c_r) = M_0 + \sum_{I \in S} c_I M_I,

    where in the above:

        - :math:`S` is a finite set of index multisets indicating non-zero monomial terms,
        - For a given index multiset :math:`I=(i_1, \dots, i_k)`,
          :math:`c_I = c_{i_1} \times \dots \times c_{i_k}`, and
        - The :math:`M_I` are arrays of the same shape, indexed by the first dimension.

    At instantiation, the user specifies :math:`S` as a list of index multisets,
    :math:`M_I` as list of arrays (specified as a >1d array) whose first index has
    corresponding ordering with the list specifying :math:`S`, and can optionally
    specify a constant term :math:`M_0`.
    """

    def __init__(
        self,
        array_coefficients: Array,
        monomial_multisets: List[Multiset],
        constant_term: Optional[Array] = None,
    ):
        """Construct a multivariable matrix polynomial.

        Args:
            array_coefficients: A 3d array representing a list of matrix coefficients.
            monomial_multisets: A list of multisets of the same length as ``matrix_coefficients``
                                indicating the monomial coefficient for each corresponding
                                ``matrix_coefficients``.
            constant_term: An array representing the constant term of the polynomial.
        """

        self._monomial_multisets = monomial_multisets

        # If operating in jax mode, wrap in Arrays
        if Array.default_backend() == "jax":
            self._array_coefficients = Array(array_coefficients)

            if constant_term is not None:
                constant_term = Array(constant_term)
            else:
                constant_term = np.zeros_like(Array(array_coefficients[0]))

            self._constant_term = constant_term
            self._compute_monomials = get_monomial_compute_function_jax(monomial_multisets)
        else:
            if constant_term is None:
                constant_term = np.zeros_like(array_coefficients[0])

            self._array_coefficients = array_coefficients
            self._constant_term = constant_term
            self._compute_monomials = get_monomial_compute_function(monomial_multisets)

    @property
    def monomial_multisets(self) -> List:
        """The monomial multisets corresponding to non-constant terms."""
        return self._monomial_multisets

    @property
    def array_coefficients(self) -> Array:
        """The array coefficients for non-constant terms."""
        return self._array_coefficients

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

    def __call__(self, c: Array) -> Array:
        """Evaluate the polynomial.

        Args:
            c: Array of variables.
        Returns:
            Value of the polynomial at c.
        """
        monomials = self._compute_monomials(c)
        return self._constant_term + np.tensordot(self._array_coefficients, monomials, axes=(0, 0))


def get_monomial_compute_function(multisets: List[Multiset]) -> Callable:
    """Construct a vectorized function for computing multivariable monomial terms indicated by
    multisets.

    The returned function takes in the individual variables as an array, and returns an array
    of computed monomial terms in the order indicated by multisets.

    The returned function is vectorized in the sense that the supplied first order terms can be
    arrays.

    The algorithm computes monomial terms of increasing order, recursively utilizing lower
    order terms.

    Args:
        complete_multisets: list of multisets.

    Returns:
        Callable: Vectorized function for computing monomials.
    """
    complete_multiset_list = get_all_submultisets(multisets)

    (
        first_order_terms,
        first_order_range,
        left_indices,
        right_indices,
        update_ranges,
    ) = get_recursive_monomial_rule(complete_multiset_list)

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


def get_monomial_compute_function_jax(multisets: List) -> Callable:
    """JAX version of get_monomial_compute_function."""

    complete_multiset_list = get_all_submultisets(multisets)

    first_order_terms, _, left_indices, right_indices, _ = get_recursive_monomial_rule(
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


def get_recursive_monomial_rule(complete_multisets: List) -> Tuple:
    """Helper function for get_monomial_compute_function and get_monomial_compute_function_jax;
    computes a representation of the algorithm for computing monomials that is used by both
    functions.

    complete_multisets is assumed to be closed under taking submultisets and in canonical order.

    Args:
        complete_multisets: Description of monomial terms.

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

    # convert multisets to list representation
    multisets_as_lists = []
    for multiset in complete_multisets:
        new_list = []
        for key, value in multiset.counts_dict.items():
            new_list += [key] * value

        multisets_as_lists.append(new_list)

    complete_multisets = multisets_as_lists

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
