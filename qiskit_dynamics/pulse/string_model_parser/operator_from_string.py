# -*- coding: utf-8 -*-

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

"""Generate operators from string."""

import numpy as np

from qiskit import QiskitError


def operator_from_string(op_label: str, subsystem_index: int, subsystem_dims: dict) -> np.ndarray:
    r"""Generates a dense operator acting on a single subsystem, tensoring
    identities for remaining subsystems.

    The single system operator is specified via a string in ``op_label``,
    the list of subsystems and their corresponding dimensions are specified in the
    dictionary ``subsystem_dims``, with system label being the keys specified as ``int``s,
    and system dimensions the values also specified as ``int``s, and ``subsystem_index``
    indicates which subsystem the operator specified by ``op_label`` acts on.

    Accepted ``op_labels`` are:
        - `'X'`: If the target subsystem is two dimensional, the
          Pauli :math:`X` operator, and if greater than two dimensional, returns
          :math:`a + a^\dagger`, where :math:`a` and :math:`a^\dagger` are the
          annihiliation and creation operators, respectively.
        - `'Y'`: If the target subsystem is two dimensional, the
          Pauli :math:`Y` operator, and if greater than two dimensional, returns
          :math:`-i(a - a^\dagger)`, where :math:`a` and :math:`a^\dagger` are the
          annihiliation and creation operators, respectively.
        - `'Z'`: If the target subsystem is two dimensional, the
          Pauli :math:`Z` operator, and if greater than two dimensional, returns
          :math:`I - 2 * N`, where :math:`N` is the number operator.
        - `'a'`, `'A'`, or `'Sm'`: If two dimensional, the sigma minus operator, and if greater,
          generalizes to the operator.
        - `'C'`, or `'Sp'`: If two dimensional, sigma plus operator, and if greater,
          generalizes to the creation operator.
        - `'N'`, or `'O'`: The number operator.
        - `'I'`: The identity operator.

    Note that the ordering of tensor factors is reversed.

    Args:
        op_label: The string labelling the single system operator.
        subsystem_index: Index of the subsystem to apply the operator.
        subsystem_dims: dictionary of subsystem labels and dimensions.

    Returns:
        np.ndarray corresponding to the specified operator.
    """

    # construct single system operator
    out = single_operator_from_string(op_label, subsystem_dims[subsystem_index])

    # sort subsystem labels and dimensions
    sorted_subsystem_keys, sorted_subsystem_dims = zip(
        *sorted(zip(subsystem_dims.keys(), subsystem_dims.values()))
    )

    # get subsystem location in ordered list
    subsystem_location = sorted_subsystem_keys.index(subsystem_index)

    # tensor identity on right if subsystem_index is not first subsystem
    if subsystem_location != 0:
        total_dim = np.prod(sorted_subsystem_dims[:subsystem_location])
        out = np.kron(out, ident(total_dim))

    # tensor identity on left if subsystem_index not the last subsystem
    if subsystem_location + 1 != len(sorted_subsystem_keys):
        total_dim = np.prod(sorted_subsystem_dims[(subsystem_location + 1) :])
        out = np.kron(ident(total_dim), out)

    return out


# functions for generating individual operators
def a(dim: int) -> np.ndarray:
    """Annihilation operator."""
    return np.diag(np.sqrt(np.arange(1, dim, dtype=complex)), 1)


def adag(dim: int) -> np.ndarray:
    """Creation operator."""
    return a(dim).conj().transpose()


def N(dim: int) -> np.ndarray:
    """Number operator."""
    return np.diag(np.arange(dim, dtype=complex))


def X(dim: int) -> np.ndarray:
    """Generalized X operator, written in terms of raising and lowering operators."""
    return a(dim) + adag(dim)


def Y(dim: int) -> np.ndarray:
    """Generalized Y operator, written in terms of raising and lowering operators."""
    return -1j * (a(dim) - adag(dim))


def Z(dim: int) -> np.ndarray:
    """Generalized Z operator, written as id - 2 * N."""
    return ident(dim) - 2 * N(dim)


def ident(dim: int) -> np.ndarray:
    """Identity operator."""
    return np.eye(dim, dtype=complex)


# operator names
__operdict = {
    "X": X,
    "Y": Y,
    "Z": Z,
    "a": a,
    "A": a,
    "Sm": a,
    "Sp": adag,
    "C": adag,
    "N": N,
    "O": N,
    "I": ident,
}


def single_operator_from_string(op_label: str, dim: int) -> np.ndarray:
    """Generate a single operator from a string.

    Helper function for operator_from_string, see its documentation for
    label interpretation.

    Args:
        op_label: String representing operator.
        dim: Dimension of operator.

    Returns:
        np.ndarray

    Raises:
        QiskitError: If op_label doesn't correspond to a known operator.
    """

    op_func = __operdict.get(op_label, None)
    if op_func is None:
        raise QiskitError("String {} does not correspond to a known operator.".format(op_label))

    return op_func(dim)


def dag(op: np.ndarray) -> np.ndarray:
    """Apply dagger."""
    return np.conjugate(np.transpose(op))


# pylint: disable=invalid-name
__funcdict = {"dag": dag}


def apply_func(name: str, op: np.ndarray) -> np.ndarray:
    """Apply function of given name, or do nothing if func not found"""
    return __funcdict.get(name, lambda x: x)(op)
