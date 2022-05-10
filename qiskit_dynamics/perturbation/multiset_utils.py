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
Utility functions for working with multisets.
"""

from typing import List

from multiset import Multiset

from qiskit import QiskitError


def validate_non_negative_ints(multiset: Multiset) -> bool:
    """Validate that a multiset only contains non-negative integers."""

    for elem in multiset.distinct_elements():
        if not isinstance(elem, int) or not elem >= 0:
            raise QiskitError("Only Multisets whose entries are non-negative integers are accepted.")


def multiset_to_sorted_list(multiset: Multiset) -> List:
    """Convert multiset to a sorted list. Assumes elements of multiset can be sorted."""

    distinct_elems = list(multiset.distinct_elements())
    distinct_elems.sort()

    sorted_list = []
    for elem in distinct_elems:
        sorted_list = sorted_list + [elem] * multiset[elem]

    return sorted_list


def clean_multisets(multisets: List[Multiset]) -> List[Multiset]:
    """Given a list of multisets, remove duplicates, and sort in non-decreasing order.

    Entries of all multisets are assumed to be sortable (e.g. all are ints, or all are strings).
    For the non-decreasing order sorting, ms1 <= ms2 if len(ms1) < len(ms2), or if
    len(ms1) == len(ms2) and if
    str(multiset_to_sorted_list(ms1)) <= str(multiset_to_sorted_list(ms2)).
    """

    unique_multisets = []
    for multiset in multisets:
        multiset = Multiset(multiset)

        if multiset not in unique_multisets:
            unique_multisets.append(multiset)

    # sort by length and lexicographic order within length
    unique_multisets.sort(key=lambda x: str(len(x)) + ', ' + str(multiset_to_sorted_list(x)))

    return unique_multisets
