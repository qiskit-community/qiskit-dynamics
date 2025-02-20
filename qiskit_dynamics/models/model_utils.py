# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Utilities model module."""

from typing import Union, List
import numpy as np
from scipy.sparse import csr_matrix, issparse
from scipy.sparse import kron as sparse_kron
from scipy.sparse import identity as sparse_identity

from qiskit_dynamics.arraylias.alias import ArrayLike, _to_dense, _numpy_multi_dispatch


def _kron(A, B):
    return _numpy_multi_dispatch(A, B, path="kron")


def vec_commutator(
    A: Union[ArrayLike, csr_matrix, List[csr_matrix]],
) -> Union[ArrayLike, csr_matrix, List[csr_matrix]]:
    r"""Linear algebraic vectorization of the linear map X -> -i[A, X]
    in column-stacking convention. In column-stacking convention we have

    .. math::
        vec(ABC) = C^T \otimes A vec(B),

    so for the commutator we have

    .. math::
        -i[A, \cdot] = -i(A \cdot - \cdot A \mapsto id \otimes A - A^T \otimes id)

    Note: this function is also "vectorized" in the programming sense for dense arrays.

    Args:
        A: Either a 2d array representing the matrix A described above,
           a 3d array representing a list of matrices, a sparse matrix, or a
           list of sparse matrices.

    Returns:
        ArrayLike: Vectorized version of the map.
    """

    if issparse(A):
        # single, sparse matrix
        sp_iden = sparse_identity(A.shape[-1], format="csr")
        return -1j * (sparse_kron(sp_iden, A) - sparse_kron(A.T, sp_iden))
    if isinstance(A, (list, np.ndarray)) and issparse(A[0]):
        # taken to be 1d array of 2d sparse matrices
        sp_iden = sparse_identity(A[0].shape[-1], format="csr")
        out = [-1j * (sparse_kron(sp_iden, mat) - sparse_kron(mat.T, sp_iden)) for mat in A]
        return np.array(out)

    A = _to_dense(A)
    iden = np.eye(A.shape[-1])
    axes = list(range(A.ndim))
    axes[-1] = axes[-2]
    axes[-2] += 1
    return -1j * (_kron(iden, A) - _kron(A.transpose(axes), iden))


def vec_dissipator(
    L: Union[ArrayLike, csr_matrix, List[csr_matrix]],
) -> Union[ArrayLike, csr_matrix, List[csr_matrix]]:
    r"""Linear algebraic vectorization of the linear map
    X -> L X L^\dagger - 0.5 * (L^\dagger L X + X L^\dagger L)
    in column stacking convention.

    This gives

    .. math::
        \overline{L} \otimes L - 0.5(id \otimes L^\dagger L +
            (L^\dagger L)^T \otimes id)

    Note: this function is also "vectorized" in the programming sense for dense matrices.
    """

    if issparse(L):
        sp_iden = sparse_identity(L[0].shape[-1], format="csr")
        return sparse_kron(L.conj(), L) - 0.5 * (
            sparse_kron(sp_iden, L.conj().T * L) + sparse_kron(L.T * L.conj(), sp_iden)
        )
    if isinstance(L, (list, np.ndarray)) and issparse(L[0]):
        # taken to be 1d array of 2d sparse matrices
        sp_iden = sparse_identity(L[0].shape[-1], format="csr")
        out = [
            sparse_kron(mat.conj(), mat)
            - 0.5
            * (sparse_kron(sp_iden, mat.conj().T * mat) + sparse_kron(mat.T * mat.conj(), sp_iden))
            for mat in L
        ]
        return np.array(out)

    iden = np.eye(L.shape[-1])
    axes = list(range(L.ndim))

    L = _to_dense(L)
    axes[-1] = axes[-2]
    axes[-2] += 1
    Lconj = L.conj()
    LdagL = Lconj.transpose(axes) @ L
    LdagLtrans = LdagL.transpose(axes)

    return _kron(Lconj, iden) @ _kron(iden, L) - 0.5 * (
        _kron(iden, LdagL) + _kron(LdagLtrans, iden)
    )
