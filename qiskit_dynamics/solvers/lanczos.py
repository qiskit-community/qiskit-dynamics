# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
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
Module containing Lanczos diagonalization algorithm
"""

from typing import Union
import numpy as np
from scipy.sparse import csr_matrix

try:
    import jax.numpy as jnp
except ImportError:
    pass


def lanczos_basis(array: Union[csr_matrix, np.ndarray], v_0: np.ndarray, k_dim: int):
    """Tridiagonalises a hermitian array in a krylov subspace of dimension k_dim
    using Lanczos algorithm.

    Args:
        array : Array to tridiagonalise.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.

    Returns:
        tridiagonal : Tridiagonal projection of ``array``.
        q_basis : Basis of the krylov subspace.
    """

    data_type = np.result_type(array.dtype, v_0.dtype)
    v_0 = np.array(v_0).reshape(-1, 1)  # ket
    array_dim = array.shape[0]
    q_basis = np.zeros((k_dim, array_dim), dtype=data_type)

    v_p = np.zeros_like(v_0)
    projection = np.zeros_like(v_0)  # v1

    beta = np.zeros((k_dim,), dtype=data_type)
    alpha = np.zeros((k_dim,), dtype=data_type)

    v_0 = v_0 / np.sqrt(np.abs(v_0.conj().T @ v_0))
    q_basis[[0], :] = v_0.T

    projection = array @ v_0
    alpha[0] = v_0.conj().T @ projection
    projection = projection - alpha[0] * v_0
    beta[0] = np.sqrt(np.abs(projection.conj().T @ projection))

    error = np.finfo(np.float64).eps

    for i in range(1, k_dim, 1):
        v_p = q_basis[i - 1, :]

        q_basis[[i], :] = projection.T / beta[i - 1]
        projection = array @ q_basis[i, :]  # |array_dim>
        alpha[i] = q_basis[i, :].conj().T @ projection  # real?
        projection = projection - alpha[i] * q_basis[i, :] - beta[i - 1] * v_p
        beta[i] = np.sqrt(np.abs(projection.conj().T @ projection))

        # additional steps to increase accuracy
        delta = q_basis[i, :].conj().T @ projection
        projection -= delta * q_basis[i, :]
        alpha[i] += delta

        if beta[i] < error:
            k_dim = i
            break

    tridiagonal = (
        np.diag(alpha[:k_dim], k=0)
        + np.diag(beta[: k_dim - 1], k=-1)
        + np.diag(beta[: k_dim - 1], k=1)
    )
    q_basis = q_basis[:k_dim]
    q_basis = q_basis.T
    return tridiagonal, q_basis


def lanczos_eig(array: Union[csr_matrix, np.ndarray], v_0: np.ndarray, k_dim: int):
    """
    Finds the lowest k_dim eigenvalues and corresponding eigenvectors of a hermitian array
    using Lanczos algorithm.
    Args:
        array : Array to diagonalize.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.

    Returns:
        q_basis : Basis of the krylov subspace.
        eigen_values : lowest ``k_dim`` Eigenvalues.
        eigen_vectors_t : Eigenvectors in both krylov-space.
        eigen_vectors_a : Eigenvectors in both hilbert-space.
    """

    tridiagonal, q_basis = lanczos_basis(array, v_0, k_dim)
    eigen_values, eigen_vectors_t = np.linalg.eigh(tridiagonal)

    eigen_vectors_a = q_basis @ eigen_vectors_t

    return q_basis, eigen_values, eigen_vectors_t, eigen_vectors_a


def lanczos_expm(
    array: Union[csr_matrix, np.ndarray],
    v_0: np.ndarray,
    k_dim: int,
    max_dt: float,
):
    """Calculates action of matrix exponential on the state using Lanczos algorithm.

    Args:
        array : Array to exponentiate.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.
        max_dt : Maximum step size.

    Returns:
        y_dt : Action of matrix exponential on state.
    """

    q_basis, eigen_values, eigen_vectors_t, _ = lanczos_eig(array, v_0, k_dim)
    y_dt = q_basis @ eigen_vectors_t @ (np.exp(-1j * max_dt * eigen_values) * eigen_vectors_t[0, :])
    return y_dt


def jax_lanczos_basis(array: jnp.ndarray, v_0: jnp.ndarray, k_dim: int):
    """Tridiagonalises a hermitian array in a krylov subspace of dimension k_dim
    using Lanczos algorithm implemented with ``jax``.

    Args:
        array : Array to tridiagonalise.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.

    Returns:
        tridiagonal : Tridiagonal projection of ``array``.
        q_basis : Basis of the krylov subspace.
    """

    data_type = jnp.result_type(array.dtype, v_0.dtype)
    v_0 = v_0.reshape(-1, 1)  # ket
    array_dim = array.shape[0]
    q_basis = jnp.zeros((k_dim, array_dim), dtype=data_type)

    v_p = jnp.zeros_like(v_0)
    projection = jnp.zeros_like(v_0)  # v1

    beta = jnp.zeros((k_dim,), dtype=data_type)
    alpha = jnp.zeros((k_dim,), dtype=data_type)

    v_0 = v_0 / jnp.sqrt(jnp.abs(v_0.conj().T @ v_0))

    #     x[idx] = y  -->  x = x.at[idx].set(y)
    q_basis = q_basis.at[[0], :].set(v_0.T)
    #     q_basis[[0], :] = v_0.T

    projection = array @ v_0

    #     x[idx] = y  -->  x = x.at[idx].set(y)
    alpha = alpha.at[0].set((v_0.conj().T @ projection)[0, 0])
    #     alpha[0] = v_0.conj().T @ projection

    projection = projection - alpha[0] * v_0

    #     x[idx] = y  -->  x = x.at[idx].set(y)
    beta = beta.at[0].set((jnp.sqrt(jnp.abs(projection.conj().T @ projection)))[0, 0])
    #     beta[0] = jnp.sqrt(jnp.abs(projection.conj().T @ projection))

    error = jnp.finfo(jnp.float64).eps

    for i in range(1, k_dim, 1):
        v_p = q_basis[i - 1, :]

        # x[idx] = y  -->  x = x.at[idx].set(y)
        q_basis = q_basis.at[[i], :].set(projection.T / beta[i - 1])
        # q_basis[[i], :] = projection.T / beta[i - 1]

        projection = array @ q_basis[i, :]  # |array_dim>

        # x[idx] = y  -->  x = x.at[idx].set(y)
        alpha = alpha.at[i].set(q_basis[i, :].conj().T @ projection)
        # alpha[i] = q_basis[i, :].conj().T @ projection  # real?

        projection = projection - alpha[i] * q_basis[i, :] - beta[i - 1] * v_p

        # x[idx] = y  -->  x = x.at[idx].set(y)
        beta = beta.at[i].set(jnp.sqrt(jnp.abs(projection.conj().T @ projection)))
        # beta[i] = jnp.sqrt(jnp.abs(projection.conj().T @ projection))

        # additional steps to increase accuracy
        delta = q_basis[i, :].conj().T @ projection
        projection -= delta * q_basis[i, :]

        # x[idx] = y  -->  x = x.at[idx].set(y)
        alpha = alpha.at[i].set(alpha[i] + delta)
        # alpha[i] += delta

        # if beta[i] < error:
        #     k_dim = i
        #     break

    tridiagonal = (
        jnp.diag(alpha[:k_dim], k=0)
        + jnp.diag(beta[: k_dim - 1], k=-1)
        + jnp.diag(beta[: k_dim - 1], k=1)
    )
    q_basis = q_basis[:k_dim]
    q_basis = q_basis.T
    return tridiagonal, q_basis


def jax_lanczos_eig(array: jnp.ndarray, v_0: jnp.ndarray, k_dim: int):
    """
    Finds the lowest k_dim eigenvalues and corresponding eigenvectors of a hermitian array
    using Lanczos algorithm implemented with ``jax``.
    Args:
        array : Array to diagonalize.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.

    Returns:
        q_basis : Basis of the krylov subspace.
        eigen_values : lowest ``k_dim`` Eigenvalues.
        eigen_vectors_t : Eigenvectors in both krylov-space.
        eigen_vectors_a : Eigenvectors in both hilbert-space.
    """

    tridiagonal, q_basis = jax_lanczos_basis(array, v_0, k_dim)
    eigen_values, eigen_vectors_t = jnp.linalg.eigh(tridiagonal)

    eigen_vectors_a = q_basis @ eigen_vectors_t

    return q_basis, eigen_values, eigen_vectors_t, eigen_vectors_a


def jax_lanczos_expm(
    array: jnp.ndarray,
    v_0: jnp.ndarray,
    k_dim: int,
    max_dt: float,
):
    """Calculates action of matrix exponential on the state using Lanczos algorithm
    implemented with ``jax``.

    Args:
        array : Array to exponentiate.
        v_0 : Initial state.
        k_dim : Dimension of the krylov subspace.
        max_dt : Maximum step size.

    Returns:
        y_dt : Action of matrix exponential on state.
    """

    q_basis, eigen_values, eigen_vectors_t, _ = jax_lanczos_eig(array, v_0, k_dim)
    y_dt = (
        q_basis @ eigen_vectors_t @ (jnp.exp(-1j * max_dt * eigen_values) * eigen_vectors_t[0, :])
    )
    return y_dt
