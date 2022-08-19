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

from typing import List, Mapping, Optional
from dataclasses import dataclass

from multiset import Multiset

from qiskit import QiskitError
from qiskit_dynamics.array import Array


@dataclass
class _LabeledData:
    """Container for arbitrarily "labeled data", i.e. data whose indices are arbitrary python
    objects. The method ``get_item`` looks up an item according to the label.
    """

    data: Mapping[int, any]
    labels: List[any]
    metadata: Optional[any] = None

    def get_item(self, label: any) -> any:
        """Look up an item in self.data according to the location of label in self.labels."""
        label = self._preprocess_label(label)

        if label in self.labels:
            return self._post_process_item(self.data[self.labels.index(label)])

        raise QiskitError("label is not present in self.labels.")

    def _preprocess_label(self, label: any) -> any:
        return label

    def _post_process_item(self, item: any) -> any:
        return item


class PowerSeriesData(_LabeledData):
    """Storage container for power series data. Labels are assumed to be ``Multiset`` instances,
    and data is assumed to be an ``Array``.
    """

    def _preprocess_label(self, label: Multiset) -> Multiset:
        """Cast to a Multiset."""
        return Multiset(label)

    def _post_process_item(self, item: Array) -> Array:
        """Wrap in an Array."""
        return Array(item, backend=self.data.backend)


class DysonLikeData(_LabeledData):
    """Storage container for DysonLike series data. Labels are assumed to be lists of ints,
    and data is assumed to be an ``Array``.
    """

    def _preprocess_label(self, label: list) -> list:
        """Cast to a list."""
        return list(label)

    def _post_process_item(self, item: Array) -> Array:
        """Wrap in an array."""
        return Array(item, backend=self.data.backend)
