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

"""
Module for custom binary operations between arrays.
"""

from typing import Optional, List, Tuple, Callable

import numpy as np

from qiskit_dynamics.array import Array

try:
    import jax.numpy as jnp
    from jax import vmap
except ImportError:
    pass


class _CustomBinaryOp:
    r"""A binary operation between arrays of dimension >1d built from taking linear combinations
    of a base binary operation acting on sub-arrays.

    This class constructs customized binary operations between
    lists of arrays :math:`A = (A_i)` and :math:`B = (B_i)` of the form:

    .. math::
        (A \times B)_i = \sum_{jk} a_{ijk} f(A_j, B_k),

    where :math:`a_{ijk}` is an array of complex scalars, and :math:`f` is a binary operation
    (a common example being matrix multiplication).

    At instantiation the binary operation :math:`f`, as well
    as the array :math:`a_{ijk}`, are specified. The array :math:`a_{ijk}` is given in a
    sparse format, as a list where each
    entry specifies the 2d sub-array :math:`a_{i}` as a 2-tuple
    with entries:

        - The non-zero entries of :math:`a_i` given as a list.
        - A 2d-array with each entry being the pair of indices ``[j,k]``.

    Internally, this specification is translated into one more suited for efficient
    evaluation:
        - A specification of the unique pairs of required evaluations
          :math:`f(A_j, B_k)` in terms of two arrays giving the left
          indices and the right indices.
        - A 2-tuple of 2d arrays specifying the linear combination of
          unique evaluations of :math:`f` to compute the :math:`i^{th}` entry of the
          binary operations. The :math:`i^{th}` entry of each array is:
            - The list of coefficients in the linear combination.
            - The index of the unique product for each coefficient.
          These arrays are padded with the value ``-1`` to make each
          linear combo the same length (relevant for JAX evaluation).
    """

    def __init__(
        self,
        operation_rule: List,
        binary_op: Callable,
        index_offset: Optional[int] = 0,
        operation_rule_compiled: Optional[bool] = False,
        backend: Optional[str] = None,
    ):
        """Initialize the binary operation.

        Args:
            operation_rule: Rule for the binary op as described in the doc string.
            binary_op: The binary operation.
            index_offset: Shift to be added to the indices in operation_rule.
            operation_rule_compiled: True if the operation_rule already corresponds to a
                                     rule that has been compiled to the internal representation.
            backend: Whether to use JAX or other looping logic.
        """

        # store binary op and compile rule to internal format for evaluation
        self._binary_op = binary_op

        if not operation_rule_compiled:
            operation_rule = _compile_custom_operation_rule(operation_rule, index_offset)
        self._unique_evaluation_pairs, self._linear_combo_rule = operation_rule
        self._backend = backend or Array.default_backend()

        # establish which version of functions to use
        if self._backend == "jax":
            self.__compute_unique_evaluations = lambda A, B: _compute_unique_evaluations_jax(
                A, B, self._unique_evaluation_pairs, vmap(self._binary_op)
            )
            self.__compute_linear_combos = lambda C: _compute_linear_combos_jax(
                C, self._linear_combo_rule
            )
        else:
            self.__compute_unique_evaluations = lambda A, B: _compute_unique_evaluations(
                A, B, self._unique_evaluation_pairs, self._binary_op
            )
            self.__compute_linear_combos = lambda C: _compute_linear_combos(
                C, self._linear_combo_rule
            )

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Evaluate the custom binary operation on arrays A, B."""
        if self._backend == "jax":
            A = Array(A).data
            B = Array(B).data

        unique_evaluations = self.__compute_unique_evaluations(A, B)
        return self.__compute_linear_combos(unique_evaluations)


class _CustomMatmul(_CustomBinaryOp):
    """Custom matmul multiplication."""

    def __init__(
        self,
        operation_rule: List,
        index_offset: Optional[int] = 0,
        operation_rule_compiled: Optional[bool] = False,
        backend: Optional[str] = None,
    ):
        """Initialize."""

        binary_op = lambda A, B: A @ B
        super().__init__(
            operation_rule=operation_rule,
            binary_op=binary_op,
            index_offset=index_offset,
            operation_rule_compiled=operation_rule_compiled,
            backend=backend,
        )


class _CustomMul(_CustomBinaryOp):
    """Custom mul multiplication."""

    def __init__(
        self,
        operation_rule: List,
        index_offset: Optional[int] = 0,
        operation_rule_compiled: Optional[bool] = False,
        backend: Optional[str] = None,
    ):
        """Initialize."""

        binary_op = lambda A, B: A * B
        super().__init__(
            operation_rule=operation_rule,
            binary_op=binary_op,
            index_offset=index_offset,
            operation_rule_compiled=operation_rule_compiled,
            backend=backend,
        )


def _compile_custom_operation_rule(
    operation_rule: List,
    index_offset: Optional[int] = 0,
    unique_evaluation_len: Optional[int] = None,
    linear_combo_len: Optional[int] = None,
) -> Tuple[np.array, np.array]:
    """Compile the list of unique evaluations and linear combinations required
    to implement a given operation_rule.

    See _CustomBinaryOp doc string for formatting details.

    Args:
        operation_rule: Custom operation rule in the sparse format in the
                        _CustomBinaryOp doc string.
        index_offset: Integer specifying a shift to apply in the 2nd and 3rd indices in the
                      sparse representation of :math:`a_{ijk}`.
        unique_evaluation_len: Integer specifying a minimum length to represent the
                               unique multiplications list. The unique multiplication list
                               is padded with entries ``[-1, -1]`` to meet the minimum length.
        linear_combo_len: Minimum length for linear combo specification. Coefficients are
                          padded with zeros, and the unique multiplication indices are
                          padded with ``-1``.

    Returns:
        Tuple[np.array, np.array]: Multiplication rule compiled into a list of
                                   unique products and a list of linear combinations of
                                   unique products for implementing the custom dot rule.
    """

    # force numpy usage for algebra specification
    new_rule = []
    for coeffs, index_pairs in operation_rule:
        new_rule.append((np.array(coeffs), np.array(index_pairs, dtype=int) + index_offset))

    operation_rule = tuple(new_rule)

    # construct list of unique multiplications and linear combo rule
    unique_evaluation_list = []
    linear_combo_rule = []
    for coeffs, index_pairs in operation_rule:

        sub_linear_combo = []
        for index_pair in index_pairs:
            # convert to list to avoid array comparison issues
            index_pair = list(index_pair)
            if index_pair not in unique_evaluation_list:
                unique_evaluation_list.append(index_pair)

            sub_linear_combo.append(unique_evaluation_list.index(index_pair))

        linear_combo_rule.append((coeffs, np.array(sub_linear_combo, dtype=int)))

    unique_evaluation_pairs = np.array(unique_evaluation_list, dtype=int)

    if unique_evaluation_len is not None and unique_evaluation_len > len(unique_evaluation_pairs):
        padding = -1 * np.ones((unique_evaluation_len - len(unique_evaluation_pairs), 2), dtype=int)
        unique_evaluation_pairs = np.append(unique_evaluation_pairs, padding, axis=0)

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

    return unique_evaluation_pairs, linear_combo_rule


def _compute_unique_evaluations(
    A: np.array, B: np.array, unique_evaluation_pairs: np.array, binary_op: Callable
) -> np.array:
    """Compute ``binary_op(A[j], B[k])`` for index pairs ``[j, k]`` in
    ``unique_evaluation_pairs``.
    """

    # evaluate first pair (assumes not all evaluation pairs are paddings of [-1, -1])
    eval_pair = unique_evaluation_pairs[0]
    unique_evaluation = binary_op(A[eval_pair[0]], B[eval_pair[1]])

    M0 = np.zeros(unique_evaluation.shape, dtype=complex)
    unique_evaluations = np.empty(
        (len(unique_evaluation_pairs),) + unique_evaluation.shape, dtype=complex
    )
    unique_evaluations[0] = unique_evaluation

    for idx, eval_pair in enumerate(unique_evaluation_pairs[1:]):
        if eval_pair[0] == -1:
            unique_evaluations[idx + 1] = M0
        else:
            unique_evaluations[idx + 1] = binary_op(A[eval_pair[0]], B[eval_pair[1]])

    return unique_evaluations


def _compute_linear_combos(
    unique_evaluations: np.array, linear_combo_rule: Tuple[np.array, np.array]
) -> np.array:
    r"""Compute linear combinations of the entries in the array ``unique_mults``
    according to ``linear_combo_rule``. The :math:`j^{th}` entry of the output is given by
    :math:`sum_k c[j, k] unique_mults[idx[j, k]]`, where ``linear_combo_rule`` is ``(c, idx)``.
    """
    M0 = np.zeros_like(unique_evaluations[0])
    C = np.empty((len(linear_combo_rule[0]),) + unique_evaluations[0].shape, dtype=complex)
    for idx, (coeffs, eval_indices) in enumerate(zip(linear_combo_rule[0], linear_combo_rule[1])):
        mat = M0
        for coeff, eval_idx in zip(coeffs, eval_indices):
            if idx != -1:
                mat = mat + coeff * unique_evaluations[eval_idx]
        C[idx] = mat

    return C


def _compute_unique_evaluations_jax(
    A: np.array,
    B: np.array,
    unique_evaluation_pairs: np.array,
    binary_op: Callable,
) -> np.array:
    """JAX version of a single loop step of :meth:`linear_combos`. Note that in this function
    binary_op is assumed to be vectorized."""
    A = jnp.append(A, jnp.zeros((1,) + A[0].shape, dtype=complex), axis=0)
    B = jnp.append(B, jnp.zeros((1,) + B[0].shape, dtype=complex), axis=0)

    return binary_op(A[unique_evaluation_pairs[:, 0]], B[unique_evaluation_pairs[:, 1]])


def _compute_single_linear_combo_jax(
    unique_evaluations: np.array, single_combo_rule: Tuple[np.array, np.array]
) -> np.array:
    """JAX version of :meth:`unique_products`."""
    coeffs, indices = single_combo_rule
    return jnp.tensordot(coeffs, unique_evaluations[indices], axes=1)


try:
    _compute_linear_combos_jax = vmap(_compute_single_linear_combo_jax, in_axes=(None, (0, 0)))
except NameError:
    pass
