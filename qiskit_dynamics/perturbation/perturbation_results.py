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
    in the attribute ``expansion_indices``.
    """

    def __init__(
        self,
        expansion_method: str,
        expansion_indices: List,
        expansion_terms: Array,
        sort_requested_labels: Optional[bool] = False,
    ):
        """Initialize.

        Args:
            expansion_method: The perturbation method used for the results, e.g. ``'dyson'``.
            expansion_indices: A list of labels for the stored terms.
            expansion_terms: A 4d array storing the results. The first axis specifies a term,
                                with the same ordering as in ``expansion_indices``. The second axis
                                specifies a time that the given term is evaluated at, and the
                                last two axes are the terms themselves.
            sort_requested_labels: Whether to try to sort labels when terms are retrieved
                                   via subscripting.
        """
        self.expansion_method = expansion_method
        self.expansion_indices = expansion_indices
        self.expansion_terms = expansion_terms
        self.sort_requested_labels = sort_requested_labels

    def __getitem__(self, label: any) -> Array:
        """Return the entry of ``self.expansion_terms`` at the index at which
        ``label`` is stored in ``expansion_indices``. If ``self.sort_labels == True``,
        ``label`` is assumed to be a list and is sorted before attempting to index
        ``expansion_indices``.

        Args:
            label: Label.

        Returns:
            Array: Perturbation results for the labelled term.

        Raises:
            QiskitError: If ``label`` is not in ``expansion_indices``.
        """

        if self.sort_requested_labels:
            label = copy(label)
            label.sort()

        if label in self.expansion_indices:
            idx = self.expansion_indices.index(label)
            return Array(self.expansion_terms[idx], backend=self.expansion_terms.backend)

        raise QiskitError("label is not present in expansion_indices.")
