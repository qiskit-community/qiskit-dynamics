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

from qiskit_dynamics.dispatch import requires_backend
from qiskit_dynamics.array import Array

try:
    import jax.numpy as jnp
    from jax.lax import scan
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


@requires_backend("jax")
def jax_lanczos_basis(array: Array, v_0: Array, k_dim: int):
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

    projection_0 = array @ v_0
    alpha_0 = v_0.conj().T @ projection_0
    projection_0 = projection_0 - alpha_0 * v_0
    beta_0 = jnp.sqrt(jnp.abs(projection_0.conj().T @ projection_0))

    initial = jnp.array([*v_0, *projection_0, beta_0])

    def lanczos_iter(carry, dummy):
        dummy += 0
        size = (len(carry) - 1) // 2
        q_p = jnp.array(carry[:size])
        projection = jnp.array(carry[size : 2 * size])
        beta_p = carry[2 * size]

        q_i = projection.T / beta_p
        projection = array @ q_i
        alpha_i = q_i.conj().T @ projection
        projection = projection - alpha_i * q_i - beta_p * q_p
        beta_i = jnp.sqrt(jnp.abs(projection.conj().T @ projection))

        delta = q_i.conj().T @ projection
        projection = projection - delta * q_i
        alpha_i = alpha_i + delta

        carry_next = jnp.array([*q_i, *projection, beta_i])
        accumulate = jnp.array([alpha_i, beta_i, *q_i])

        return carry_next, accumulate

    _, accumulate_stack = scan(lanczos_iter, initial, None, length=k_dim - 1)

    alpha = jnp.array([alpha_0, *accumulate_stack[:, 0]])
    beta = jnp.array([beta_0, *accumulate_stack[:, 1]])
    q_basis = jnp.array([v_0, *accumulate_stack[:, 2:]])

    tridiagonal = (
        jnp.diag(alpha, k=0) + jnp.diag(beta[: k_dim - 1], k=-1) + jnp.diag(beta[: k_dim - 1], k=1)
    )
    q_basis = q_basis.T
    return tridiagonal, q_basis


@requires_backend("jax")
def jax_lanczos_eig(array: Array, v_0: Array, k_dim: int):
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


@requires_backend("jax")
def jax_lanczos_expm(
    array: Array,
    v_0: Array,
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
