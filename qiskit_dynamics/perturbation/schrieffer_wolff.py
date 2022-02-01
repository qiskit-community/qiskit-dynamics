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

"""
Schrieffer-Wolff perturbation computation.
"""

from typing import List, Optional

import numpy as np
from scipy.linalg import solve_sylvester

from qiskit import QiskitError

from qiskit_dynamics.perturbation import ArrayPolynomial, Multiset
from qiskit_dynamics.perturbation.multiset import get_all_submultisets
from qiskit_dynamics.perturbation.solve_lmde_perturbation import merge_expansion_order_indices


def schrieffer_wolff(
    H0: np.ndarray,
    perturbations: List[np.ndarray],
    expansion_order: Optional[int] = None,
    expansion_labels: Optional[List[Multiset]] = None,
    perturbation_labels: Optional[List[Multiset]] = None,
    tol: Optional[float] = 1e-15,
) -> ArrayPolynomial:
    """Construct truncated multi-variable Schrieffer-Wolff transformation.

    Not sure what the correct output should be but the ``ArrayPolynomial`` will contain
    all relevant information.

    Also, for now we assume H0 is non-degenerate for initial development, but we
    should add a "degeneracy_tol" argument for detecting degeneracies when we add this case.
    More generally we could think about allowing the user to choose the projection,
    so they could e.g. choose a block-diagonal structure even if there are no degeneracies.

    Further notes to self:
        - I wrote this initially thinking it could be vectorized, but I think solve_sylvester
          only works for 2d arrays. Also I just realized that even if solve_sylvester
          was vectorized I haven't written the solve_commutator_projection function to
          be vectorized (due to the conditionals). May want to consider modifying things so
          that it explicitly works with 2d arrays, otherwise the "vectorized" stuff is just
          confusing to read

    To do:
        - Maybe validate that that everything is hermitian, and add an anti-hermitian projection
        step at the end for the perturbation terms
        - Should we rename H0 to Hd given the usage of 0 as an index?
    """

    # validate H0 is diagonal and hermitian
    if H0.ndim == 1:
        H0 = np.diag(H0)

    if np.max(np.abs(np.diag(np.diag(H0)).real - H0)) > tol:
        raise QiskitError("H0 must be a diagonal Hermitian matrix.")

    # validate H0 is non-degenerate
    for idx1 in range(H0.shape[-1]):
        for idx2 in range(idx1 + 1, H0.shape[-1]):
            if np.abs(H0[idx1, idx1] - H0[idx2, idx2]) < tol:
                raise QiskitError("The eigenvalues of H0 must be non-degenerate.")

    # validate perturbations are Hermitian
    for perturbation in perturbations:
        if np.max(np.abs(perturbation.conj().transpose() - perturbation)) > tol:
            raise QiskitError("Perturbations must be Hermitian.")

    ##################################################################################################
    # To do: add validation. For validating the expansion_order/labels args we could
    # move the validation for both solve_lmde_perturbation and this function into
    # merge_expansion_order_indices (and maybe also move this function into multiset.py)
    ##################################################################################################

    perturbations = np.array(perturbations)
    mat_shape = perturbations[0].shape

    if perturbation_labels is None:
        perturbation_labels = [Multiset({k: 1}) for k in range(len(perturbations))]

    # get all requested terms in the expansion
    expansion_labels = merge_expansion_order_indices(
        expansion_order, expansion_labels, perturbation_labels, symmetric=True
    )
    expansion_labels = get_all_submultisets(expansion_labels)

    # construct labels for recursive computation
    recursive_labels = []
    for label in expansion_labels:
        for k in range(len(label), 0, -1):
            recursive_labels.append((label, k))

    # initialize arrays used to store results
    recursive_A = np.zeros((len(recursive_labels),) + mat_shape, dtype=complex)
    recursive_B = np.zeros((len(recursive_labels),) + mat_shape, dtype=complex)
    expansion_terms = np.zeros((len(expansion_labels),) + mat_shape, dtype=complex)

    # right hand side storage
    rhs_mat = None

    # recursively compute all matrices
    for (recursive_idx, (expansion_label, commutator_order)) in enumerate(recursive_labels):
        expansion_idx = expansion_labels.index(expansion_label)

        # initialize rhs matrix at first occurence of current expansion_label
        if commutator_order == len(expansion_label):
            rhs_mat = np.zeros(mat_shape, dtype=complex)

        # if commutator_order == 1, all terms required to determine term for current
        # expansion_label are computed, so solve for the term and initialize
        # base cases of recursive matrices that depend on it
        if commutator_order == 1:
            if expansion_label in perturbation_labels:
                rhs_mat = rhs_mat + perturbations[expansion_idx]
                recursive_B[recursive_idx] = perturbations[expansion_idx]
            # solve for expansion term
            expansion_terms[expansion_idx] = solve_commutator_projection(H0, rhs_mat, tol=tol)

            # initialize recursive_A base case
            recursive_A[recursive_idx] = commutator(expansion_terms[expansion_idx], H0)
        else:
            # get all 2-fold partitions, bounding the submultisets to have size
            # <= len(expansion_label) - (commutator_order - 1)
            submultisets, complements = expansion_label.submultisets_and_complements(
                len(expansion_label) - commutator_order + 2
            )
            for submultiset, complement in zip(submultisets, complements):
                SI = expansion_terms[expansion_labels.index(submultiset)]
                recursive_lower_idx = recursive_labels.index((complement, commutator_order - 1))
                recursive_A[recursive_idx] += commutator(SI, recursive_A[recursive_lower_idx])
                recursive_B[recursive_idx] += commutator(SI, recursive_B[recursive_lower_idx])

            recursive_A[recursive_idx] = recursive_A[recursive_idx] / commutator_order
            recursive_B[recursive_idx] = recursive_B[recursive_idx] / (commutator_order - 1)
            rhs_mat = rhs_mat + recursive_A[recursive_idx] + recursive_B[recursive_idx]

    # project onto anti-hermitian matrices
    expansion_terms = 0.5 * (expansion_terms - expansion_terms.conj().transpose((0, 2, 1)))

    return ArrayPolynomial(array_coefficients=expansion_terms, monomial_multisets=expansion_labels)


def solve_commutator_projection(A: np.ndarray, B: np.ndarray, tol: Optional[float] = 1e-15) -> np.ndarray:
    """Solve [A, X] = OD(B) assuming A is diagonal, where OD is the projection onto
    off-diagonal matrices.

    Args:
        A: 2d array on the LHS of the equation.
        B: 2d array on the RHS of the equation.
        tol: Tolerance for determining if OD(B) is 0.

    Returns:
        Solution to the above problem.
    """

    # project B onto off diagonal matrices
    B = B.copy()
    k = B.shape[-1]
    B[range(k), range(k)] = 0.0

    # if B is zero after projection, return 0
    if np.max(np.abs(B)) < tol:
        return np.zeros(A.shape, dtype=complex)

    return solve_sylvester(A, -A, B)


def commutator(A, B):
    """Matrix commutator."""
    return A @ B - B @ A
