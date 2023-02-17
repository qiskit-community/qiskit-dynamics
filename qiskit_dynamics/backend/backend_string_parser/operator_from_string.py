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

"""Generate operators from string.

This file is meant for internal use and may be changed at any point.
"""

from typing import Dict
import numpy as np

from qiskit import QiskitError
import qiskit.quantum_info as qi


def _operator_from_string(
    op_label: str, subsystem_label: int, subsystem_dims: Dict[int, int]
) -> np.ndarray:
    r"""Generates a dense operator acting on a single subsystem, tensoring
    identities for remaining subsystems.

    The single system operator is specified via a string in ``op_label``,
    the list of subsystems and their corresponding dimensions are specified in the
    dictionary ``subsystem_dims``, with system label being the keys specified as ``int``s,
    and system dimensions the values also specified as ``int``s, and ``subsystem_label``
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
        subsystem_label: Index of the subsystem to apply the operator.
        subsystem_dims: Dictionary of subsystem labels and dimensions.

    Returns:
        np.ndarray corresponding to the specified operator.

    Raises:
        QiskitError: If op_label is invalid.
    """

    # construct single system operator
    op_func = __operdict.get(op_label, None)
    if op_func is None:
        raise QiskitError(f"String {op_label} does not correspond to a known operator.")

    dim = subsystem_dims[subsystem_label]
    out = qi.Operator(op_func(dim), input_dims=[dim], output_dims=[dim])

    # sort subsystem labels and dimensions according to subsystem label
    sorted_subsystem_keys, sorted_subsystem_dims = zip(
        *sorted(zip(subsystem_dims.keys(), subsystem_dims.values()))
    )

    # get subsystem location in ordered list
    subsystem_location = sorted_subsystem_keys.index(subsystem_label)

    # construct full operator
    return qi.ScalarOp(sorted_subsystem_dims).compose(out, [subsystem_location]).data


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
