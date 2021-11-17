# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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

def operator_from_string(op_label: str, subsystem_index: int, subsystem_dims: dict) -> np.ndarray:
    """ Generates an operator acting on a single subsystem, tensoring identities for remaining
    subsystems.

    ###########################################################################################################
    to document:
    - single operator definitions
    - reverse ordering of tensor factors

    inputs:
        - op_label: label for a single-subsystem operator
        - subsystem_index: index of the subsystem that the operator applies to
        - subsystem_dims: dimensions of all subsystems.

    returns:
        np.ndarray corresponding to the specified operator.
    """
    out = single_operator_from_string(op_label, subsystem_dims[subsystem_index])

    sorted_subsystem_keys, sorted_subsystem_dims = zip(*sorted(zip(subsystem_dims.keys(), subsystem_dims.values())))

    subsystem_location = sorted_subsystem_keys.index(subsystem_index)

    # tensor identity on right if subsystem_index is not first subsystem
    if subsystem_location != 0:
        total_dim = np.prod(sorted_subsystem_dims[:subsystem_location])
        out = np.kron(out, ident(total_dim))

    # tensor identity on left if subsystem_index not the last subsystem
    if subsystem_location + 1 != len(sorted_subsystem_keys):
        total_dim = np.prod(sorted_subsystem_dims[(subsystem_location + 1):])
        out = np.kron(ident(total_dim), out)

    return out

# derived operators:
# Sp -> a, Sm -> adag, O -> N

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

def single_operator_from_string(op_label: str, dim: int) -> np.ndarray:
    """Generate a single operator from a string.

    Helper function for operator_from_string, see its documentation for
    label interpretation.

    Args:
        op_label: String representing operator.
        dim: Dimension of operator.

    Returns:
        np.ndarray
    """

    op_func = __operdict.get(op_label, None)
    if op_func is None:
        raise QiskitError('String {} does not correspond to a known operator.'.format(op_label))

    return op_func(dim)

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
    "I": ident
}
