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

from typing import List, Tuple, Optional
from itertools import combinations
from collections import OrderedDict

from qiskit import QiskitError

class Multiset:
    """A multiset whose elements are integers."""

    def __init__(self, counts_dict: dict[int, int]):
        """Construct a multiset from a dictionary of counts.

        Args:
            entries: Dictionary of counts, with keys giving the entries of
                     the multiset.
        """

        validate_counts_dict(counts_dict)
        self._counts_dict = canonicalize_counts_dict(counts_dict)

    @property
    def counts_dict(self) -> OrderedDict[int, int]:
        """Counts dictionary storing element counts."""
        return self._counts_dict

    def unique(self) -> set:
        """Return the unique elements of self as a set."""
        return set(self.counts_dict.keys())

    def count(self, element: int) -> int:
        """Get the count of an element in self."""
        return self.counts_dict.get(element, 0)

    def union(self, other: 'Multiset') -> 'Multiset':
        """Multiset union of self with other."""

        unique_elements = self.unique().union(other.unique())

        new_dict = {}
        for elem in unique_elements:
            new_dict[elem] = self.count(elem) + other.count(elem)

        return Multiset(new_dict)

    def difference(self, other: 'Multiset') -> 'Multiset':
        """Multiset difference of other from self."""
        unique_elements = self.unique()

        new_dict = {}
        for elem in unique_elements:
            count = self.count(elem) - other.count(elem)
            if count > 0:
                new_dict[elem] = count

        return Multiset(new_dict)

    def issubmultiset(self, other: 'Multiset') -> bool:
        """Check if self is a submultiset of other."""

        for elem in self.unique():
            if self.count(elem) > other.count(elem):
                return False

        return True

    def submultisets_and_complements(self, submultiset_bound: Optional[int] = None) -> Tuple[List['Multiset'], List['Multiset']]:
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

        if submultiset_bound is None:
            submultiset_bound = len(self)

        # convert self to a list representation
        self_list = []
        for key, value in self.counts_dict.items():
            self_list += [key] * value

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

        keys = self.counts_dict.keys()

        # convert back to proper dict representation
        formatted_submultisets = [Multiset({key: submultiset.count(key) for key in keys}) for submultiset in submultisets]
        formatted_complements = [Multiset({key: complement.count(key) for key in keys}) for complement in complements]

        return formatted_submultisets, formatted_complements



    def __le__(self, other: 'Multiset') -> 'Multiset':
        """Dunder method for self.issubmultiset(other)."""
        return self.issubmultiset(other)

    def __ge__(self, other: 'Multiset') -> 'Multiset':
        """Dunder method for other.issubmultiset(self)."""
        return other.issubmultiset(self)

    def __eq__(self, other: 'Multiset') -> bool:
        """Check if other == self."""
        return self.counts_dict == other.counts_dict

    def __add__(self, other: 'Multiset') -> 'Multiset':
        """Dunder method for union."""
        return self.union(other)

    def __sub__(self, other: 'Multiset') -> 'Multiset':
        """Dunder method for difference."""
        return self.difference(other)

    def __len__(self) -> int:
        """Size of multiset."""
        return sum(self.counts_dict.values())

    def __str__(self) -> str:
        return 'Multiset({})'.format(str(dict(self.counts_dict)))

    def __repr__(self) -> str:
        return 'Multiset({})'.format(str(self.counts_dict))


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
            raise QiskitError('counts_dict keys and values must be integers.')

        if val < 0:
            raise QiskitError('counts_dict values must be non-negative integers.')


def canonicalize_counts_dict(counts_dict: dict[int, int]) -> OrderedDict[int, int]:
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


def submultiset_filter(multiset_candidates: List[Multiset], multiset_list: List[Multiset]) -> List[Multiset]:
    """Filter the list of multiset_candidates based on whether they are a
    submultiset of an element in multiset_list.
    """

    filtered_multisets = []
    for candidate in multiset_candidates:
        for multiset in multiset_list:
            if candidate <= multiset:
                filtered_multisets.append(candidate)
                break

    return filtered_multisets
