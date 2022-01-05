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

r"""
Class for storing results of perturbation theory computations.
"""

from typing import List, Optional
from copy import copy

from qiskit import QiskitError
from qiskit_dynamics.array import Array


class PerturbationResults:
    """Storage container for results of perturbation theory computation.
    All terms are stored in a single array, with individual terms being retrievable via
    subscript-style access of this class using the label of the terms, which are stored
    in the attribute ``term_labels``.
    """

    def __init__(
        self,
        perturbation_method: str,
        term_labels: List,
        perturbation_terms: Array,
        sort_requested_labels: Optional[bool] = False,
    ):
        """Initialize.

        Args:
            perturbation_method: The perturbation method used for the results, e.g. ``'dyson'``.
            term_labels: A list of labels for the stored terms.
            perturbation_terms: A 4d array storing the results. The first axis specifies a term,
                                with the same ordering as in ``term_labels``. The second axis
                                specifies a time that the given term is evaluated at, and the
                                last two axes are the terms themselves.
            sort_requested_labels: Whether to try to sort labels when terms are retrieved
                                   via subscripting.
        """
        self.perturbation_method = perturbation_method
        self.term_labels = term_labels
        self.perturbation_terms = perturbation_terms
        self.sort_requested_labels = sort_requested_labels

    def __getitem__(self, label: any) -> Array:
        """Return the entry of ``self.perturbation_terms`` at the index at which
        ``label`` is stored in ``term_labels``. If ``self.sort_labels == True``,
        ``label`` is assumed to be a list and is sorted before attempting to index
        ``term_labels``.

        Args:
            label: Label.

        Returns:
            Array: Perturbation results for the labelled term.

        Raises:
            QiskitError: If ``label`` is not in ``term_labels``.
        """

        if self.sort_requested_labels:
            label = copy(label)
            label.sort()

        if label in self.term_labels:
            idx = self.term_labels.index(label)
            return Array(self.perturbation_terms[idx], backend=self.perturbation_terms.backend)

        raise QiskitError("label is not present in term_labels.")
