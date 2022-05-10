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

from multiset import Multiset

from qiskit import QiskitError

def validate_non_negative_ints(multiset: Multiset) -> bool:
    """Validate that a multiset only contains non-negative integers."""

    for elem in multiset.distinct_elements():
        if not isinstance(elem, int) or not elem >= 0:
            raise QiskitError("Only Multisets whose entries are non-negative integers are accepted.")

def multiset_to_sorted_list(multiset: Multiset) -> list:
    """Convert multiset to a sorted list. Assumes elements of multiset can be sorted."""

    distinct_elems = list(multiset.distinct_elements())
    distinct_elems.sort()

    sorted_list = []
    for elem in distinct_elems:
        sorted_list = sorted_list + [elem] * multiset[elem]

    return sorted_list
