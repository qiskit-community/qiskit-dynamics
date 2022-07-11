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
Module containing Lanczos diagonalization and time evolution algorithms
"""

from typing import Union, Optional
import numpy as np
from scipy.sparse import csr_matrix


def lanczos_basis(A: Union[csr_matrix, np.ndarray], y0: np.ndarray, k_dim: int):
    """Tridiagonalises a hermitian array in a krylov subspace of dimension k_dim
    using Lanczos algorithm.
    reference: https://tensornetwork.org/mps/algorithms/timeevo/global-krylov.html

    Args:
        A : Array to tridiagonalise. Must be hermitian.
        y0 : Vector to initialise Lanczos iteration.
        k_dim : Dimension of the krylov subspace.

    Returns:
        tridiagonal : Tridiagonal projection of ``A``.
        q_basis : Basis of the krylov subspace.
    """

    data_type = np.result_type(A.dtype, y0.dtype)
    y0 = np.array(y0).reshape(-1, 1)
    array_dim = A.shape[0]
    q_basis = np.zeros((k_dim, array_dim), dtype=data_type)

    v_p = np.zeros_like(y0)
    projection = np.zeros_like(y0)

    beta = np.zeros((k_dim,), dtype=data_type)
    alpha = np.zeros((k_dim,), dtype=data_type)

    y0 = y0 / np.linalg.norm(y0)
    q_basis[[0], :] = y0.T

    projection = A @ y0
    alpha[0] = y0.conj().T @ projection
    projection = projection - alpha[0] * y0
    beta[0] = np.linalg.norm(projection)

    error = np.finfo(np.float64).eps

    for i in range(1, k_dim, 1):
        if beta[i - 1] < error:
            k_dim = i
            break

        v_p = q_basis[i - 1, :]

        q_basis[[i], :] = projection.T / beta[i - 1]
        projection = A @ q_basis[i, :]
        alpha[i] = q_basis[i, :].conj().T @ projection
        projection = projection - alpha[i] * q_basis[i, :] - beta[i - 1] * v_p
        beta[i] = np.linalg.norm(projection)

        # additional steps to increase accuracy
        delta = q_basis[i, :].conj().T @ projection
        projection -= delta * q_basis[i, :]
        alpha[i] += delta

    tridiagonal = (
        np.diag(alpha[:k_dim], k=0)
        + np.diag(beta[: k_dim - 1], k=-1)
        + np.diag(beta[: k_dim - 1], k=1)
    )
    q_basis = q_basis[:k_dim]
    q_basis = q_basis.T
    return tridiagonal, q_basis


def lanczos_eig(A: Union[csr_matrix, np.ndarray], y0: np.ndarray, k_dim: int):
    """Finds the lowest (Algebraic) ``k_dim`` eigenvalues and corresponding eigenvectors of a
    hermitian array using Lanczos algorithm.
    Args:
        A : Array to diagonalize. Must be hermitian.
        y0 : Vector to initialise Lanczos iteration.
        k_dim : Dimension of the krylov subspace.

    Returns:
        q_basis : Basis of the krylov subspace.
        eigen_values : lowest ``k_dim`` Eigenvalues.
        eigen_vectors_t : Eigenvectors in krylov-space.
        eigen_vectors_a : Eigenvectors in hilbert-space.
    """

    tridiagonal, q_basis = lanczos_basis(A, y0, k_dim)
    eigen_values, eigen_vectors_t = np.linalg.eigh(tridiagonal)

    eigen_vectors_a = q_basis @ eigen_vectors_t

    return q_basis, eigen_values, eigen_vectors_t, eigen_vectors_a


def lanczos_expm(
    A: Union[csr_matrix, np.ndarray],
    y0: np.ndarray,
    k_dim: int,
    max_dt: Optional[float] = 1,
):
    """Calculates action of matrix exponential of an anti-hermitian array on the state using
    Lanczos algorithm.

    Args:
        A : Array to exponentiate. Must be anti-hermitian.
        y0 : Initial state.
        k_dim : Dimension of the krylov subspace.
        max_dt : Maximum step size.

    Returns:
        y_dt : Action of matrix exponential on state.

    Raises:
        QiskitError: If ``y0`` is not 1d or 2d
    """

    if y0.ndim == 1:
        A = 1j * A  # make hermitian
        q_basis, eigen_values, eigen_vectors_t, _ = lanczos_eig(A, y0, k_dim)
        y_dt = (
            q_basis
            @ eigen_vectors_t
            @ (np.exp(-1j * max_dt * eigen_values) * eigen_vectors_t[0, :])
        )

    elif y0.ndim == 2:
        y_dt = [lanczos_expm(A, yi, k_dim, max_dt) for yi in y0.T]
        y_dt = np.array(y_dt).T

    else:
        ValueError("y0 must be 1d or 2d")

    return y_dt
