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
Module for representing custom matrix products.

The functions in this file support customized dot rules between
lists of arrays :math:`A = (A_i)` and :math:`B = (B_i)` of the form:

.. math::
    (A \times B)_i = \sum_{jk} a_{ijk}A_jB_k,

where :math:`a_{ijk}` is an array of complex scalars. In the above, the
product :math:`A_jB_k` is normally thought of as a matrix product,
e.g. if :math:`A` and :math:`B` are 3d arrays, the functionality here
works for any arrays with at least 3 dimensions for which the last
2 dimensions are square.

Here, we represent the array :math:`a_{ijk}` in a sparse representation,
passed as the ``multiplication_rule`` argument to the function
:meth:`compile_custom_dot_rule`. The results of
:meth:`compile_custom_dot_rule` can then be used with :meth:`custom_dot`,
or :meth:`custom_dot_jax` to implement the product.

The sparse format for specifying :math:`a_{ijk}` is as a list where each
entry specifies the 2d sub-array :math:`a_{i}` as a 2-tuple
with entries:
    - The non-zero entries of :math:`a_i` given as a list.
    - A 2d-array with each entry being the pair of indices ``[j,k]``.
:meth:`compile_custom_dot_rule` translates the above specification into:
    - A specification of the unique pairs of required products
      :math:`A_jB_k` in terms of two arrays giving the left
      indices and the right indices.
    - A 2-tuple of 2d arrays specifying the linear combination of
      unique products to compute the :math:`i^{th}` entry of the
      full product. The :math:`i^{th}` entry of each array is:
        - The list of coefficients in the linear combination.
        - The index of the unique product for each coefficient.
      These arrays are padded with the value ``-1`` to make each
      linear combo the same length.
Additionally, the index ``-1`` when specifying a product is interpreted
as the identity matrix.
    - E.g. when specifying a multiplication rule, the pair ``[-1, 2]`` represents
      simply ``B[2]`` and ``[0, -1]`` represents ``A[0]``.
This enables implicit identities when specifying the custom product.
"""

from typing import Optional, List, Tuple

import numpy as np

try:
    import jax.numpy as jnp
    from jax import vmap
except ImportError:
    pass


def compile_custom_dot_rule(
    multiplication_rule: List,
    index_offset: Optional[int] = 0,
    unique_mult_len: Optional[int] = None,
    linear_combo_len: Optional[int] = None,
) -> Tuple[np.array, np.array]:
    """Compile the list of unique products and linear combinations required
    to implement a custom dot. See module doc string for formatting details.

    Args:
        multiplication_rule: Custom multiplication rule in the sparse format in the
                             file doc string.
        index_offset: Integer specifying a shift in the 2nd and 3rd indices in the
                      sparse representation of :math:`a_{ijk}`.
        unique_mult_len: Integer specifying a minimum length to represent the
                         unique multiplications list. The unique multiplication list
                         is padded with entries ``[-2, -2]`` to meet the minimum length.
        linear_combo_len: Minimum length for linear combo specification. Coefficients are
                          padded with zeros, and the unique multiplication indices are
                          padded with ``-1``.

    Returns:
        Tuple[np.array, np.array]: multiplication rule compiled into a list of
                                   unique products and a list of linear combinations of
                                   unique products for implementing the custom dot rule.
    """

    # force numpy usage for algebra specification
    new_rule = []
    for coeffs, index_pairs in multiplication_rule:
        new_rule.append((np.array(coeffs), np.array(index_pairs, dtype=int) + index_offset))

    multiplication_rule = tuple(new_rule)

    # construct list of unique multiplications and linear combo rule
    unique_mult_list = []
    linear_combo_rule = []
    for coeffs, index_pairs in multiplication_rule:

        sub_linear_combo = []
        for index_pair in index_pairs:
            # convert to list to avoid array comparison issues
            index_pair = list(index_pair)
            if index_pair not in unique_mult_list:
                unique_mult_list.append(index_pair)

            sub_linear_combo.append(unique_mult_list.index(index_pair))

        linear_combo_rule.append((coeffs, np.array(sub_linear_combo, dtype=int)))

    unique_mult_pairs = np.array(unique_mult_list, dtype=int)

    if unique_mult_len is not None and unique_mult_len > len(unique_mult_pairs):
        padding = -2 * np.ones((unique_mult_len - len(unique_mult_pairs), 2), dtype=int)
        unique_mult_pairs = np.append(unique_mult_pairs, padding, axis=0)

    # pad linear combo rule with -1 for shorter rules
    max_len = linear_combo_len or 0
    for coeffs, _ in linear_combo_rule:
        max_len = max(max_len, len(coeffs))

    padded_linear_combo_rule = []
    for coeffs, indices in linear_combo_rule:
        if coeffs.shape[0] < max_len:
            pad_len = max_len - coeffs.shape[0]
            coeffs = np.append(coeffs, np.zeros(pad_len))
            indices = np.append(indices, -1 * np.ones(pad_len, dtype=int))

        padded_linear_combo_rule.append((coeffs, indices))

    coeff_array = np.vstack([a[0] for a in padded_linear_combo_rule])
    index_array = np.vstack([a[1] for a in padded_linear_combo_rule])

    linear_combo_rule = (coeff_array, index_array)

    return unique_mult_pairs, linear_combo_rule


def unique_products(A: np.array, B: np.array, unique_mult_pairs: np.array) -> np.array:
    """Compute ``A[j] @ B[k]`` for index pairs ``[j, k]`` in ``unique_mult_pairs``."""
    M0 = np.zeros_like(A[0])
    unique_mults = np.empty((len(unique_mult_pairs),) + A.shape[1:], dtype=complex)
    for idx, mult_pair in enumerate(unique_mult_pairs):
        if mult_pair[0] == -2:
            unique_mults[idx] = M0
        elif mult_pair[0] == -1:
            unique_mults[idx] = B[mult_pair[1]]
        elif mult_pair[1] == -1:
            unique_mults[idx] = A[mult_pair[0]]
        else:
            unique_mults[idx] = A[mult_pair[0]] @ B[mult_pair[1]]

    return unique_mults


def linear_combos(unique_mults: np.array, linear_combo_rule: Tuple[np.array, np.array]) -> np.array:
    r"""Compute linear combinations of the entries in the array ``unique_mults``
    according to ``linear_combo_rule``. The :math:`j^{th}` entry of the output is given by
    :math:`sum_k c[j, k] unique_mults[idx[j, k]]`, where ``linear_combo_rule`` is ``(c, idx)``.
    """
    M0 = np.zeros_like(unique_mults[0])
    C = np.empty((len(linear_combo_rule[0]),) + unique_mults[0].shape, dtype=complex)
    for idx, (coeffs, prod_indices) in enumerate(zip(linear_combo_rule[0], linear_combo_rule[1])):
        mat = M0
        for coeff, prod_idx in zip(coeffs, prod_indices):
            if idx != -1:
                mat = mat + coeff * unique_mults[prod_idx]
        C[idx] = mat

    return C


def custom_dot(
    A: np.array, B: np.array, compiled_custom_product: Tuple[np.array, np.array]
) -> np.array:
    """Multiply arrays of dimension at least 3 according to the ``compiled_custom_product``
    rule, output by :meth:`compile_custom_dot_rule`."""
    unique_mult_pairs, linear_combo_rule = compiled_custom_product
    unique_mults = unique_products(A, B, unique_mult_pairs)
    output = linear_combos(unique_mults, linear_combo_rule)
    return np.array(output)


def unique_products_jax(A: np.array, B: np.array, unique_mult_pairs: np.array) -> np.array:
    """Jax version of a single loop step of :meth:`linear_combos`."""
    # vectorized append identity and 0 to A and B
    big_ident = jnp.broadcast_to(jnp.eye(A.shape[-1], dtype=complex), (1,) + A.shape[1:])
    big_zeros = jnp.broadcast_to(jnp.zeros(A.shape[-1], dtype=complex), (1,) + A.shape[1:])
    X = jnp.append(big_zeros, big_ident, axis=0)

    A = jnp.append(A, X, axis=0)
    B = jnp.append(B, X, axis=0)

    return A[unique_mult_pairs[:, 0]] @ B[unique_mult_pairs[:, 1]]


def single_linear_combo_jax(
    unique_mults: np.array, single_combo_rule: Tuple[np.array, np.array]
) -> np.array:
    """Jax version of :meth:`unique_products`."""
    coeffs, indices = single_combo_rule
    return jnp.tensordot(coeffs, unique_mults[indices], axes=1)


try:
    linear_combos_jax = vmap(single_linear_combo_jax, in_axes=(None, (0, 0)))
except NameError:
    pass


def custom_dot_jax(
    A: np.array, B: np.array, compiled_custom_product: Tuple[np.array, np.array]
) -> np.array:
    """Jax version of ``custom_dot``."""

    unique_mult_pairs, linear_combo_rule = compiled_custom_product
    unique_mults = unique_products_jax(A, B, unique_mult_pairs)
    output = linear_combos_jax(unique_mults, linear_combo_rule)
    return output
