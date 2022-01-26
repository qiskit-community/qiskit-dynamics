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

"""Multiset classes and functions."""

from typing import List, Tuple, Optional, Union, Dict
from itertools import combinations
from collections import OrderedDict

from qiskit import QiskitError


class Multiset:
    """A multiset whose elements are integers.

    Represents multiset of integers, which is an unordered collection of integers with
    repeated entries allowed. Contains methods similar to ``set`` for checking
    if one ``Multiset`` is a submultiset of another, performing ``union`` and ``difference``,
    etc.

    A ``Multiset`` can be instantiated using a ``dict`` whose keys are the elements of the
    multiset, and whose values are the counts, e.g.

    .. code-block:: python

        Multiset({1: 3, 0: 2})

    represents a multiset with ``3`` copies of ``1``, and ``2`` copies of ``0``.

    Alternatively, a ``Multiset`` may be instantiated from a list of integers that explicitly
    contains repeated entries. For example, the following code creates the same multiset as
    above:

    .. code-block:: python

        Multiset.from_list([0, 0, 0, 1, 1])

    Note that, internally, the contents of the multiset are stored using an ``OrderedDict``,
    ordered in terms of increasing keys. While a multiset is inherently unordered,
    having a standard ordering can aid in algorithms.

    This class also implements a partial order on multisets. It holds that
    ``multiset1 < multiset2`` if either:

    - ``len(multset1) < len(multiset2)``, or
    - Iterating through the ordered unique elements in ``multiset1.union(multiset2)``,
      one of the elements satisfies ``multiset1.count(element) < multiset2.count(element)``
      after all preceding elements satisfy ``multiset1.count(element) == multiset2.count(element)``.

    This ordering corresponds to sorting multisets by size, and within a given size,
    ordering according to lexicographic ordering when the multiset is represented as a
    sorted list with repeated entries.
    """

    def __init__(self, counts_dict: Dict[int, int]):
        """Construct a multiset from a dictionary of counts.

        Args:
            counts_dict: Dictionary of counts, with keys giving the entries of
                         the multiset.
        """

        validate_counts_dict(counts_dict)
        self._counts_dict = canonicalize_counts_dict(counts_dict)

    @classmethod
    def from_list(cls, multiset_as_list: List[int]) -> "Multiset":
        """Construct a multiset from a list of indices."""
        keys = set(multiset_as_list)
        return Multiset({key: multiset_as_list.count(key) for key in keys})

    @property
    def counts_dict(self) -> 'OrderedDict[int, int]':
        """Counts dictionary storing element counts."""
        return self._counts_dict

    def unique(self) -> set:
        """Return the unique elements of self as a set."""
        return set(self.counts_dict.keys())

    def count(self, element: int) -> int:
        """Get the count of an element in self."""
        return self.counts_dict.get(element, 0)

    def union(self, other: "Multiset") -> "Multiset":
        """Multiset union of self with other."""

        unique_elements = self.unique().union(other.unique())

        new_dict = {}
        for elem in unique_elements:
            new_dict[elem] = self.count(elem) + other.count(elem)

        return Multiset(new_dict)

    def difference(self, other: "Multiset") -> "Multiset":
        """Multiset difference of other from self."""
        unique_elements = self.unique()

        new_dict = {}
        for elem in unique_elements:
            count = self.count(elem) - other.count(elem)
            if count > 0:
                new_dict[elem] = count

        return Multiset(new_dict)

    def as_list(self) -> List[int]:
        """Return multiset in an ordered list format."""
        self_list = []
        for key, value in self.counts_dict.items():
            self_list += [key] * value

        return self_list

    def issubmultiset(self, other: "Multiset") -> bool:
        """Check if self is a submultiset of other."""

        for elem in self.unique():
            if self.count(elem) > other.count(elem):
                return False

        return True

    def submultisets_and_complements(
        self, submultiset_bound: Optional[int] = None
    ) -> Tuple[List["Multiset"], List["Multiset"]]:
        """Return a pair of lists giving all submultisets of size smaller than
        submultiset_bound, and corresponding complements.

        Note: Does not include the empty set in the submultisets, and the default behaviour is to
        not include the full multiset.

        Args:
            submultiset_bound: Strict upper bound on submultiset to include.
                               Defaults to len(self).

        Returns:
            Submultisets and corresponding complements.
        """

        if submultiset_bound is None or submultiset_bound > len(self):
            submultiset_bound = len(self)

        self_list = self.as_list()

        submultisets = []
        complements = []

        for k in range(1, submultiset_bound):
            location_subsets = combinations(range(len(self_list)), k)
            for location_subset in location_subsets:
                subset = []
                complement = []

                for loc, multiset_entry in enumerate(self_list):
                    if loc in location_subset:
                        subset.append(multiset_entry)
                    else:
                        complement.append(multiset_entry)

                if subset not in submultisets:
                    submultisets.append(subset)
                    complements.append(complement)

        # convert back to proper dict representation
        formatted_submultisets = [Multiset.from_list(submultiset) for submultiset in submultisets]
        formatted_complements = [Multiset.from_list(complement) for complement in complements]

        return formatted_submultisets, formatted_complements

    def __eq__(self, other: "Multiset") -> bool:
        """Check if other == self."""
        return self.counts_dict == other.counts_dict

    def __add__(self, other: "Multiset") -> "Multiset":
        """Dunder method for union."""
        return self.union(other)

    def __sub__(self, other: "Multiset") -> "Multiset":
        """Dunder method for difference."""
        return self.difference(other)

    def __lt__(self, other: "Multiset") -> bool:
        """Implements an ordering on multisets."""
        if len(self) < len(other):
            return True

        if len(other) < len(self):
            return False

        unique_entries = list(self.unique().union(other.unique()))
        unique_entries.sort()

        for element in unique_entries:
            self_count = self.count(element)
            other_count = other.count(element)

            if self_count < other_count:
                return False

            if self_count > other_count:
                return True

        return False

    def __len__(self) -> int:
        """Size of multiset."""
        return sum(self.counts_dict.values())

    def __str__(self) -> str:
        return "Multiset({})".format(str(dict(self.counts_dict)))

    def __repr__(self) -> str:
        return "Multiset({})".format(str(self.counts_dict))


def validate_counts_dict(counts_dict: dict):
    """Validate that all keys and values are integers, and that values are non-negative.

    Args:
        counts_dict: Dictionary of counts.
    Raises:
        QiskitError: If any key or value is not an integer, or if any
        value is less than 0.
    """

    for key, val in counts_dict.items():
        if not isinstance(key, int) or not isinstance(val, int):
            raise QiskitError("counts_dict keys and values must be integers.")

        if val < 0:
            raise QiskitError("counts_dict values must be non-negative integers.")


def canonicalize_counts_dict(counts_dict: Dict[int, int]) -> 'OrderedDict[int, int]':
    """Delete empty parts of the counts dict, and return as an OrderedDict sorted by keys.

    Args:
        counts_dict: Dictionary of counts.
    Returns:
        OrderedDict: OrderedDict of counts, ordered according to key value, and
        with entries with non-positive counts removed.
    """

    sorted_keys = list(counts_dict.keys())
    sorted_keys.sort()

    new_dict = OrderedDict()
    for key in sorted_keys:
        if counts_dict[key] >= 1:
            new_dict[int(key)] = counts_dict[key]

    return new_dict


# pylint: disable=invalid-name
def to_Multiset(x: Union[Multiset, List[int], dict]) -> Multiset:
    """Convert x to a Multiset if possible."""

    if isinstance(x, Multiset):
        return x

    if isinstance(x, list):
        return Multiset.from_list(x)

    if isinstance(x, dict):
        return Multiset(x)

    raise QiskitError("input is not a valid Multiset specification.")


def submultiset_filter(
    multiset_candidates: List[Multiset], multiset_list: List[Multiset]
) -> List[Multiset]:
    """Filter the list of multiset_candidates based on whether they are a
    submultiset of an element in multiset_list.
    """

    filtered_multisets = []
    for candidate in multiset_candidates:
        for multiset in multiset_list:
            if candidate.issubmultiset(multiset):
                filtered_multisets.append(candidate)
                break

    return filtered_multisets


def clean_multisets(multisets: List[Union[Multiset, dict, list]]) -> List[Multiset]:
    """Given a list of multisets, put them in order of non-decreasing length and
    remove duplicates.

    Args:
        multisets: List of multisets.
    Returns:
        List[Multiset]
    """

    unique_multisets = []
    for multiset in multisets:
        multiset = to_Multiset(multiset)

        if multiset not in unique_multisets:
            unique_multisets.append(multiset)

    unique_multisets.sort(key=len)

    return unique_multisets


def get_all_submultisets(multisets: List[Multiset]) -> List[Multiset]:
    """Given a list of multisets, return a list of all possible submultisets
    of multisets in the list.

    This list is sorted in the canonical way: for index multisets I, J, I < J iff:
        - len(I) < len(J) or
        - len(I) == len(J) and I < J when viewed as strings

    Args:
        multisets: List of multisets (not necessarilly correctly formatted).

    Returns:
        List: Complete list of index multisets generated by the argument.
    """

    if multisets == []:
        return []

    # clean list to unique list of properly formatted terms
    multisets = clean_multisets(multisets)

    max_order = max(map(len, multisets))

    # partition multisets according to size
    order_dict = {k: [] for k in range(1, max_order + 1)}
    for multiset in multisets:
        order = len(multiset)
        if multiset not in order_dict[order]:
            order_dict[order].append(multiset)

    # loop through orders in reverse order, adding subterms to lower levels if necessary
    for order in range(max_order, 1, -1):

        for multiset in order_dict[order]:
            submultisets = multiset.submultisets_and_complements(2)[1]

            for submultiset in submultisets:
                if submultiset not in order_dict[order - 1]:
                    order_dict[order - 1].append(submultiset)

    full_list = []
    for submultisets in order_dict.values():
        full_list += submultisets

    full_list.sort()

    return full_list
