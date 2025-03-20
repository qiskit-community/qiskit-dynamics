# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
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
Orthonormal basis classes.
"""

from typing import List, Optional, Callable, Tuple, Union
from itertools import product

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics.arraylias.alias import _preferred_lib

from .subsystem import Subsystem
from .abstract_subsystem_operators import AbstractSubsystemOperator


class ONBasis:
    """Represents a list of orthonormal vectors."""

    def __init__(
        self,
        subsystems: List[Subsystem],
        basis_vectors: Optional[np.ndarray] = None,
        labels: Optional[List] = None,
    ):
        """An orthonormal basis specified as a 2d array whose columns are the vectors.

        Args:
            subsystems: A list of Subsystem instances. Tensor product ordering is assumed to be
                reversed.
            basis_vectors: The vectors given as a 2d array, with the vectors being the columns. If
                ``None``, defaults to the standard basis.
            labels: Labels for the basis vectors. If ``None``, defaults to integer ordering based
                on subsystem dimensions.
        """
        self._subsystems = subsystems
        subsystem_dims = [x.dim for x in subsystems]

        if basis_vectors is None:
            basis_vectors = np.eye(np.prod(subsystem_dims), dtype=complex)

        if basis_vectors.ndim != 2:
            raise ValueError("basis_vectors must be supplied as a 2d array.")

        if basis_vectors.shape[0] != np.prod(subsystem_dims):
            raise QiskitError("Vector dimension does not match subsystem dimension.")

        self._basis_vectors = basis_vectors
        self._basis_vectors_adj = self.basis_vectors.conj().transpose()

        gram_matrix = self.basis_vectors_adj @ self.basis_vectors

        if not np.allclose(gram_matrix, np.eye(len(gram_matrix))):
            raise QiskitError("Vectors are not orthonormal.")

        self._labels = labels or _default_indexing(subsystems, basis_vectors.shape[1])

    @property
    def subsystems(self):
        """Subsystems representing the tensor product space on which the basis is defined."""
        return self._subsystems

    @property
    def labels(self):
        """Basis element labels."""
        return self._labels

    @property
    def basis_vectors(self):
        """Basis vectors as the columns of an 2-d array."""
        return self._basis_vectors

    @property
    def basis_vectors_adj(self):
        """Adjoint of the basis vectors matrix."""
        return self._basis_vectors_adj

    @property
    def projection(self):
        """Matrix for the orthogonal projection onto the subspace spanned by the basis."""
        return self.basis_vectors @ self.basis_vectors_adj

    def probabilities(self, x: np.ndarray):
        """Treating x as a state vector or density matrix, compute the probabilities of observing
        the outcomes of a measurement defined by the basis vectors.

        Args:
            x: The statevector or density matrix. Which case is determined by ``x.ndim``.
        Returns:
            A 1-d array representing probabilities computed according to the Born rule.
        """
        if x.ndim == 1:
            return unp.abs(self.decompose(x)) ** 2
        elif x.ndim == 2:
            return unp.real(unp.diag(self.basis_vectors_adj @ x @ self.basis_vectors))
        else:
            raise QiskitError("ONBasis.probabilities not defined for a >2d array.")

    def decompose(self, x):
        """Return the coefficients of the projection of a vector x in the basis."""
        return self.basis_vectors_adj @ x

    def project(self, x):
        """Project a vector x onto the subspace spanned by the basis."""
        return self.projection @ x

    def subset(self, condition: Callable):
        """Get a new ONBasis consisting of a subset of this one filtered according to the condition
        function defined on the basis labels.

        Args:
            condition: A boolean-valued function on the labels of this instance of ``ONBasis``.
        Returns:
            ONBasis: An ``ONBasis`` consisting of the (vector, label) pairs in this one for which
                ``condition(label) == True``.
        """

        indices_to_include = []
        sublabels = []
        for idx, label in enumerate(self.labels):
            if condition(label):
                indices_to_include.append(idx)
                sublabels.append(label)

        indices_to_include = np.array(indices_to_include)
        return ONBasis(
            basis_vectors=self.basis_vectors[:, indices_to_include],
            subsystems=self.subsystems,
            labels=sublabels,
        )

    def __len__(self):
        return self.basis_vectors.shape[1]


class DressedBasis(ONBasis):
    """A basis with labels of the form {"index": index, "eval": eval}, where each eval is a float
    representing an eigenvalue.
    """

    def __init__(self, subsystems, basis_vectors, evals, indices: Optional[List] = None):
        """Initialize a basis where each element has an associated eval.

        The labels for the basis vectors will be dictionaries with keys "index" and "eval". Note
        that the ``ground_state`` property will always return the first basis vector.

        Args:
            subsytems: List of subsystems.
            basis_vectors: 2d array of basis vectors.
            evals: 1d array or list of eigenvalues.
            indices: Optional indexing of basis vectors (defaults to standard).
        """

        indices = indices or _default_indexing(subsystems, basis_vectors.shape[1])
        labels = [{"index": idx, "eval": eval} for idx, eval in zip(indices, evals)]

        super().__init__(subsystems=subsystems, basis_vectors=basis_vectors, labels=labels)

    @classmethod
    def from_hamiltonian(
        cls,
        hamiltonian: Union[np.ndarray, AbstractSubsystemOperator],
        subsystems: List[Subsystem],
        ordering: str = "default",
    ):
        """Build a DressedBasis instance from the eigendecomposition of a Hamiltonian, ordered in
        terms of non-decreasing eigenvalues.

        Args:
            hamiltonian: The Hamiltonian operator.
            subsystems: A list of subsystem instances.
            ordering: The ordering with which to set the basis. The value ``"default"`` returns the
                basis ordered according to the entries of the vectors with the maximum absolute
                value.
        """

        if isinstance(hamiltonian, AbstractSubsystemOperator):
            hamiltonian = hamiltonian.matrix(ordered_subsystems=subsystems)

        if ordering == "default":
            evals, evecs = _sorted_eigh(hamiltonian)
        else:
            raise QiskitError(f"Ordering type '{ordering}' not recognized.")

        return cls(subsystems=subsystems, basis_vectors=evecs, evals=evals)

    @property
    def ground_state(self):
        """Return the first basis vector."""
        return self.basis_vectors[:, 0]

    @property
    def evals(self):
        """The eigenvalues."""
        return unp.array([x["eval"] for x in self.labels])

    @property
    def computational_states(self):
        """Get subspace of all states with subsystem indices <=0."""

        def condition(label):
            return not any(x > 1 for x in label["index"])

        return self.subset(condition)

    def low_energy_states(self, cutoff_energy):
        """Return a DressedBasis object corresponding to a low energy subspace."""

        new_evals = []
        new_indices = []
        new_basis_vectors = []
        for label, vector in zip(self.labels, self.basis_vectors.transpose()):
            if label["eval"] < cutoff_energy:
                new_evals.append(label["eval"])
                new_indices.append(label["index"])
                new_basis_vectors.append(vector)

        return DressedBasis(
            subsystems=self.subsystems,
            basis_vectors=unp.array(new_basis_vectors).transpose(),
            evals=new_evals,
            indices=new_indices,
        )


def _default_indexing(subsystems, num):
    subsystem_dims = [s.dim for s in subsystems]
    subsystem_state_labels = [range(dim) for dim in subsystem_dims]
    big_prod = product(*subsystem_state_labels)
    labels = []
    for x in big_prod:
        label = list(x)
        label.reverse()
        labels.append(tuple(label))
    return labels[:num]


def _sorted_eigh(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given a Hermitian operator ``H``, return the output of ``jnp.linalg.eigh``, but with
    the eigenvalues and eigenvectors sorted so that the argmax of the absolute value of the
    eigenvectors is in non-decreasing order. This also ensures that the diagonal of the matrix
    containing the eigenvectors is positive.

    The motivation for this function is when computing dressed bases, in which ``H`` is nearly
    diagonal, and therefore the eigenvectors (if there are no degeneracies) are guaranteed to be
    close to the elementary basis vectors. In such a scenario, this function ensures that the
    eigenvectors are ordered according to which elementary basis vector they are closest to, and the
    largest (in absolute value) entry of each eigenvector will be positive. If the conditions of
    this motivating example are not met, this function will still return a valid diagonalization of
    ``H``, however the ordering of the eigenvectors will not have any specific meaning.

    Args:
        H: The Hermitian operator.

    Returns:
        Tuple: The sorted eigenvalues and eigenvectors.
    """

    evals, evecs = unp.linalg.eigh(H)

    # sort based on largest entry of each column
    evecs_trans = evecs.transpose()

    if _preferred_lib(H) == "jax":
        import jax.numpy as jnp
        from jax import vmap

        max_indices_argsort = vmap(jnp.argmax)(unp.abs(evecs_trans)).argsort()
    else:
        max_indices_argsort = np.array([np.argmax(np.abs(x)) for x in evecs_trans]).argsort()

    sorted_evecs_trans = evecs_trans[max_indices_argsort]
    sorted_evals = evals[max_indices_argsort]

    # ensure all largest entries are positive
    exp_factors = unp.exp(-1j * unp.angle(unp.diag(sorted_evecs_trans)))
    sorted_evecs_trans = exp_factors[:, np.newaxis] * sorted_evecs_trans

    return sorted_evals, sorted_evecs_trans.transpose()
