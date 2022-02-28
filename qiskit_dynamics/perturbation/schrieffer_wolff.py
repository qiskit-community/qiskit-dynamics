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
Schrieffer-Wolff perturbation computation.
"""

from typing import List, Optional

import numpy as np
from scipy.linalg import solve_sylvester

from qiskit import QiskitError

from qiskit_dynamics.perturbation import ArrayPolynomial, Multiset
from qiskit_dynamics.perturbation.multiset import get_all_submultisets
from qiskit_dynamics.perturbation.perturbation_utils import merge_multiset_expansion_order_labels


def schrieffer_wolff(
    H0: np.ndarray,
    perturbations: List[np.ndarray],
    expansion_order: Optional[int] = None,
    expansion_labels: Optional[List[Multiset]] = None,
    perturbation_labels: Optional[List[Multiset]] = None,
    tol: Optional[float] = 1e-15,
) -> ArrayPolynomial:
    r"""Construct truncated multi-variable expansion for the generator of the unitary
    Schrieffer-Wolff transformation.

    Schrieffer-Wolff perturbation theory seeks to perturbatively construct the generator
    of a unitary transformation which either block-diagonalizes or diagonalizes a Hamiltonian
    which perturbatively is either non-block-diagonal or non-diagonal
    [:footcite:`wikipedia_schriefferwolff_2021`, :footcite:`bravyi_schriefferwolff_2011`,
    :footcite:`schrieffer_relation_1966`, :footcite:`luttinger_motion_1955`]. This
    function considers the diagonalization version of the problem.
    Given a multi-variable power series decomposition of a Hamiltonian:

    .. math::

        H(c) = H_0 + \sum_{I \in S} c_I H_I,

    where:

        - :math:`H_0` is the unperturbed diagonal Hamiltonian, given by argument ``H0``,
        - :math:`H_I` are the multi-variable perturbations, given by the argument
          ``perturbations``, and
        - :math:`S` are the perturbation indices specified as
          :class:`~qiskit_dynamics.perturbation.multiset.Multiset`\s, and given
          in the argument ``perturbation_labels``,

    this function produces a truncated power series decomposition for the generator of the
    associated Schrieffer-Wolff transformation, where:

        - The expansion terms computed are specified by a combination of ``expansion_order``
          and ``expansion_labels``. All terms up to order ``expansion_order`` are computed,
          along with any additional terms given by ``expansion_labels``.

    The generator of the Schrieffer-Wolff transformation as a function of the perturbation
    variables, :math:`S(c)`, is defined implicitly as an anti-Hermitian operator for which

    .. math::

        e^{S(c)}H(c)e^{-S(c)}

    is diagonal. Expanding :math:`S(c)` in a perturbative expansion about :math:`c=0`:

    .. math::

        S(c) = \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I S_I,

    (where the constant term is taken to be :math:`0` as :math:`H(0) = H_0`
    is assumed already diagonal), we follow [:footcite:`wikipedia_schriefferwolff_2021`] and
    expand the following using the BCH formula and collect terms in the coefficients :math:`c_I`:

    .. math::

        e^{S(c)}H(c)e^{-S(c)} = H_0 + \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)}
             c_I \left(H_0 + [S_I, H_0] + \sum_{m=2}^k (A_I^{(m)} + B_I^{(m)})\right),

    where the :math:`A_I^{(m)}` and :math:`B_I^{(m)}` are defined recursively via

    .. math::

        A_I^{(m)} = \frac{1}{m} \sum_{J \subsetneq I, |J| \leq |I| - m + 1}
            [S_J, A_{I \setminus J}^{(m-1)}]

    and

    .. math::

        B_I^{(m)} = \frac{1}{(m-1)} \sum_{J \subsetneq I, |J| \leq |I| - m + 1}
            [S_J, B_{I \setminus J}^{(m-1)}],

    with base cases :math:`A_I^{(1)} = [S_I, H_0]` and :math:`B_I^{(1)} = H_I`.

    This function recursively computes the desired :math:`S_I` at successive orders, where
    each :math:`S_I` is constructed by solving

    .. math::

        [H_0, S_I] = \Delta(H_0 + \sum_{m=2}^k (A_I^{(m)} + B_I^{(m)}))

    where :math:`\Delta` is the projection onto matrices with zero diagonal, and each
    successive order of the :math:`A_I^{(m)}` and :math:`B_I^{(m)}` is computed
    using the recursion relation above.

    Args:
        H0: Diagonal Hamiltonian, assumed and validated to be non-degenerate.
        perturbations: Perturbation terms.
        expansion_order: Expansion order to compute to.
        expansion_labels: Expansion labels to compute.
        perturbation_labels: Labels for perturbation terms in the form of multisets.
        tol: Tolerance for validation and for determining if matrix entries are 0.
    Returns:
        ArrayPolynomial object containing the perturbative Shrieffer-Wolff coefficients.
    Raises:
        QiskitError: If any assumptions of the function are not met: H0 is diagonal, real,
        and non-degenerate, the perturbations are Hermitian, and at least one of
        expansion_order or expansion_labels is specified.

    .. footbibliography::
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

    # setup labels for perturbations and expansion terms
    if perturbation_labels is None:
        perturbation_labels = [Multiset({k: 1}) for k in range(len(perturbations))]

    # get all requested terms in the expansion
    expansion_labels = merge_multiset_expansion_order_labels(
        perturbation_labels=perturbation_labels,
        expansion_order=expansion_order,
        expansion_labels=expansion_labels,
    )
    expansion_labels = get_all_submultisets(expansion_labels)

    # construct labels for recursive computation
    recursive_labels = []
    for label in expansion_labels:
        for k in range(len(label), 0, -1):
            recursive_labels.append((label, k))

    # setup results arrays
    perturbations = np.array(perturbations)
    mat_shape = perturbations[0].shape

    storage_shape = (len(recursive_labels),) + mat_shape
    recursive_A = np.zeros(storage_shape, dtype=complex)
    recursive_B = np.zeros(storage_shape, dtype=complex)
    expansion_terms = np.zeros((len(expansion_labels),) + mat_shape, dtype=complex)

    # recursively compute all matrices
    current_rhs = None
    current_expansion_idx = None
    for (recursive_idx, (expansion_label, commutator_order)) in enumerate(recursive_labels):

        # initialize rhs matrix at first occurence of current expansion_label
        if commutator_order == len(expansion_label):
            current_expansion_idx = expansion_labels.index(expansion_label)
            current_rhs = np.zeros(mat_shape, dtype=complex)

        if commutator_order == 1:
            # if commutator_order == 1, solve expansion_term and initialize next layer of recursion
            if expansion_label in perturbation_labels:
                current_rhs += perturbations[current_expansion_idx]
                recursive_B[recursive_idx] = perturbations[current_expansion_idx]

            expansion_terms[current_expansion_idx] = solve_commutator_projection(
                H0, current_rhs, tol=tol
            )
            recursive_A[recursive_idx] = commutator(expansion_terms[current_expansion_idx], H0)
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
            current_rhs += recursive_A[recursive_idx] + recursive_B[recursive_idx]

    # project onto anti-hermitian matrices
    expansion_terms = 0.5 * (expansion_terms - expansion_terms.conj().transpose((0, 2, 1)))

    return ArrayPolynomial(array_coefficients=expansion_terms, monomial_labels=expansion_labels)


def solve_commutator_projection(
    A: np.ndarray, B: np.ndarray, tol: Optional[float] = 1e-15
) -> np.ndarray:
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
