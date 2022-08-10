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
Module containing Lanczos diagonalization and time evolution algorithms.
"""

from typing import Union, Optional
import numpy as np
from scipy.sparse import csr_matrix

from qiskit_dynamics.dispatch import requires_backend
from qiskit_dynamics.array import Array

try:
    import jax.numpy as jnp
    from jax.lax import scan, cond
except ImportError:
    pass


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


def lanczos_eigh(A: Union[csr_matrix, np.ndarray], y0: np.ndarray, k_dim: int):
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
    """

    tridiagonal, q_basis = lanczos_basis(A, y0, k_dim)
    eigen_values, eigen_vectors_t = np.linalg.eigh(tridiagonal)

    return q_basis, eigen_values, eigen_vectors_t


def lanczos_expm(
    A: Union[csr_matrix, np.ndarray],
    y0: np.ndarray,
    k_dim: int,
    scale_factor: Optional[float] = 1,
):
    """Calculates action of matrix exponential of an anti-hermitian array on the state using
    Lanczos algorithm.

    Args:
        A : Array to exponentiate. Must be anti-hermitian.
        y0 : Initial state.
        k_dim : Dimension of the krylov subspace.
        scale_factor : Maximum step size.

    Returns:
        y_dt : Action of matrix exponential on state.
    """

    if y0.ndim == 1:
        A = 1j * A  # make hermitian
        y0_norm = np.linalg.norm(y0)
        q_basis, eigen_values, eigen_vectors_t = lanczos_eigh(A, y0 / y0_norm, k_dim)
        y_dt = (
            q_basis
            @ eigen_vectors_t
            @ (np.exp(-1j * scale_factor * eigen_values) * eigen_vectors_t[0, :])
        ) * y0_norm

    else:
        y_dt = [lanczos_expm(A, yi, k_dim, scale_factor) for yi in y0.T]
        y_dt = np.array(y_dt).T

    return y_dt


@requires_backend("jax")
def jax_lanczos_basis(A: Array, y0: Array, k_dim: int):
    """JAX version of lanczos_basis."""

    data_type = jnp.result_type(A.dtype, y0.dtype)
    y0 = y0.astype(data_type)

    projection_0 = A @ y0
    alpha_0 = y0.conj().T @ projection_0
    projection_0 = projection_0 - alpha_0 * y0
    beta_0 = jnp.sqrt(jnp.abs(projection_0.conj().T @ projection_0))

    initial = [y0, projection_0, beta_0]

    def lanczos_iter(carry, _):
        q_p, projection, beta_p = carry

        q_i = projection.T / beta_p
        projection = A @ q_i
        alpha_i = q_i.conj().T @ projection
        projection = projection - alpha_i * q_i - beta_p * q_p
        beta_i = jnp.sqrt(jnp.abs(projection.conj().T @ projection))

        delta = q_i.conj().T @ projection
        projection = projection - delta * q_i
        alpha_i = alpha_i + delta

        carry_next = [q_i, projection, beta_i]
        accumulate = [alpha_i, beta_i, q_i]

        return carry_next, accumulate

    def cond_func(qpb, _):
        _, _, beta_i = qpb
        zeros_func = lambda _: (
            [jnp.zeros_like(y0), jnp.zeros_like(y0), 0.0],
            [jnp.zeros(1, dtype=data_type)[0], 0.0, jnp.zeros_like(y0)],
        )
        carry_next2, accumulate2 = cond(
            beta_i > 0, lambda carry: lanczos_iter(carry, _), zeros_func, qpb
        )
        return carry_next2, accumulate2

    _, (alpha, beta, q_basis) = scan(cond_func, initial, None, length=k_dim - 1)

    alpha = jnp.array([alpha_0, *alpha])
    beta = jnp.array([beta_0, *beta])
    q_basis = jnp.array([y0, *q_basis])

    tridiagonal = (
        jnp.diag(alpha, k=0) + jnp.diag(beta[: k_dim - 1], k=-1) + jnp.diag(beta[: k_dim - 1], k=1)
    )
    q_basis = q_basis.T
    return tridiagonal, q_basis


@requires_backend("jax")
def jax_lanczos_eigh(A: Array, y0: Array, k_dim: int):
    """JAX version of lanczos_eigh."""

    tridiagonal, q_basis = jax_lanczos_basis(A, y0, k_dim)
    eigen_values, eigen_vectors_t = jnp.linalg.eigh(tridiagonal)

    return q_basis, eigen_values, eigen_vectors_t


@requires_backend("jax")
def jax_lanczos_expm(
    A: Array,
    y0: Array,
    k_dim: int,
    scale_factor: Optional[float] = 1,
):
    """JAX version of lanczos_expm."""

    if y0.ndim == 1:
        A = 1j * A  # make hermitian
        y0_norm = jnp.linalg.norm(y0)
        q_basis, eigen_values, eigen_vectors_t = jax_lanczos_eigh(A, y0 / y0_norm, k_dim)
        y_dt = (
            q_basis
            @ eigen_vectors_t
            @ (jnp.exp(-1j * scale_factor * eigen_values) * eigen_vectors_t[0, :])
        ) * y0_norm

    else:
        _, y_dt = scan(
            lambda _, yi: (None, jax_lanczos_expm(A, yi, k_dim, scale_factor)), None, y0.T
        )
        y_dt = jnp.array(y_dt).T

    return y_dt
