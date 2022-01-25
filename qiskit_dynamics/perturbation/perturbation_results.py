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

from typing import List, Union
from dataclasses import dataclass

from qiskit import QiskitError
from qiskit_dynamics.array import Array
from qiskit_dynamics.perturbation import Multiset


@dataclass
class PerturbationResults:
    """Storage container for results of perturbation theory computation.

    Attributes are:

        - ``expansion_method``: Which perturbative expansion the terms correspond to.
        - ``expansion_labels``: A list of labels for the stored expanion terms.
        - ``expansion_terms``: A single array containing all expansion terms, whose first
          index is assumed to have corresponding ordering with ``expansion_labels``.

    Aside storing the above, this class can be subscripted to retrieve an entry of
    ``expansion_terms`` at the location at which a given ``label`` appears in
    ``expansion_labels``. E.g. the perturbation term with label ``[0, 1, 2]``
    can be retrieved from an instance named ``perturbation_results`` via:

    .. code:: python

        perturbation_results[[0, 1, 2]]

    .. automethod:: __getitem__
    """

    expansion_method: str
    expansion_labels: List
    expansion_terms: Array

    def __getitem__(self, label: Union[list, Multiset]) -> Array:
        """Return the expansion term with a given label.

        Return the entry of ``self.expansion_terms`` at the index at which
        ``label`` is stored in ``expansion_labels``. If ``self.sort_labels == True``,
        ``label`` is assumed to be a list and is sorted before attempting to index
        ``expansion_labels``.

        Args:
            label: Label.

        Returns:
            Array: Perturbation results for the labelled term.

        Raises:
            QiskitError: If ``label`` is not in ``expansion_labels``.
        """

        if "symmetric" in self.expansion_method:
            if isinstance(label, list):
                label = Multiset.from_list(label)
            elif isinstance(label, dict):
                label = Multiset(label)

        if label in self.expansion_labels:
            idx = self.expansion_labels.index(label)
            return Array(self.expansion_terms[idx], backend=self.expansion_terms.backend)

        raise QiskitError("label is not present in expansion_labels.")
