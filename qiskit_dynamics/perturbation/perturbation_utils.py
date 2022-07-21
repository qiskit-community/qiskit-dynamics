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
Utility functions for perturbation module.
"""

from typing import List, Optional, Union
from itertools import product

from multiset import Multiset

from qiskit import QiskitError

from qiskit_dynamics.perturbation.multiset_utils import (
    _validate_non_negative_ints,
    _clean_multisets,
)


def _merge_multiset_expansion_order_labels(
    perturbation_labels: Union[List[int], List[Multiset]],
    expansion_order: Optional[int] = None,
    expansion_labels: Optional[List[Multiset]] = None,
) -> List[Multiset]:
    """Helper function for merging expansion_order and expansion_labels arguments
    in the multiset case for functions that require specifying expansion terms to compute.

    Generates a list of all Multisets of a given size given by expansion_order,
    and includes any additional multisets specified by expansion_labels. The elements
    of the multisets are drawn from perturbation_labels, which is either a list of
    ints, or a list of Multisets from which the ints are drawn.

    At least one of expansion_order or expansion_labels must be specified. Accepts
    only multisets and labels consisting of non-negative integers.

    Args:
        perturbation_labels: Specification of elements of the multisets to generate.
        expansion_order: Size of multisets to generate.
        expansion_labels: Additional multisets to keep.
    Returns:
        List of multisets merging expansion_order and expansion_labels.
    Raises:
        QiskitError: If neither expansion_order nor expansion_labels is specified.
    """

    # validate
    if expansion_order is None and expansion_labels is None:
        raise QiskitError("At least one of expansion_order or expansion_labels must be specified.")

    # clean expansion_labels if specified
    if expansion_labels is not None:
        expansion_labels = _clean_multisets(expansion_labels)
        for label in expansion_labels:
            _validate_non_negative_ints(label)

    # if no expansion_order passed, just return expansion_labels
    if expansion_order is None:
        return expansion_labels

    # generate all multisets of size expansion_order with entries in perturbation_labels

    # get unique multiset elements
    unique_labels = set()
    for perturbation_label in perturbation_labels:
        if isinstance(perturbation_label, int):
            unique_labels = unique_labels.union({perturbation_label})
        else:
            perturbation_label = Multiset(perturbation_label)
            _validate_non_negative_ints(perturbation_label)
            unique_labels = unique_labels.union(perturbation_label.distinct_elements())
    unique_labels = list(unique_labels)
    unique_labels.sort()

    # get all possible counts for multisets of a given size with a given number of labels
    all_multiset_counts = _ordered_partitions(expansion_order, len(unique_labels))

    # create all such multisets
    output_multisets = []
    for multiset_count in all_multiset_counts:
        output_multisets.append(Multiset(dict(zip(unique_labels, multiset_count))))

    if expansion_labels is not None:
        output_multisets = output_multisets + expansion_labels

    return _clean_multisets(output_multisets)


def _merge_list_expansion_order_labels(
    perturbation_num: int,
    expansion_order: Optional[int] = None,
    expansion_labels: Optional[List[List[int]]] = None,
) -> List[int]:
    """Helper function for merging expansion_order and expansion_labels arguments
    in the list case for functions that require specifying expansion terms to compute.

    Generates a list of all lists of integers in [0, ..., perturbation_num - 1] of a given size
    given by expansion_order, and includes any additional lists specified by expansion_labels.

    At least one of expansion_order or expansion_labels must be specified.

    Args:
        perturbation_num: Number of perturbations.
        expansion_order: Size of lists to generate.
        expansion_labels: Additional lists to keep.
    Returns:
        List of multisets merging expansion_order and expansion_labels.
    Raises:
        QiskitError: If neither expansion_order nor expansion_labels is specified.
    """

    # validate
    if expansion_order is None and expansion_labels is None:
        raise QiskitError(
            """At least one of expansion_order or
                          expansion_labels must be specified."""
        )

    if expansion_order is None:
        return expansion_labels

    # generate all possible lists of length expansion_order with integers between
    # [0, ..., perturbation_num - 1]
    unique_indices = list(range(perturbation_num))
    output_lists = list(map(list, product(unique_indices, repeat=expansion_order)))

    if expansion_labels is not None:
        for label in expansion_labels:
            if label not in output_lists:
                output_lists.append(label)

        output_lists.sort(key=str)
        output_lists.sort(key=len)

    return output_lists


def _ordered_partitions(n: int, length: int) -> List[List[int]]:
    """Return the ordered integer partitions of n of a given length, including zeros.

    Args:
        n: Number to partition.
        length: Length of partitions.
    Returns:
        Ordered partitions.
    """

    if length == 1:
        return [[n]]

    full_list = []
    for k in range(n + 1):
        full_list = full_list + [[k] + part for part in _ordered_partitions(n - k, length - 1)]

    return full_list
