# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
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
Subsystem  class
"""


class Subsystem:
    """A Hilbert space with a name and a dimension."""

    def __init__(self, name: str, dim: int):
        """Initialize with name and dimension.

        Args:
            name: Name of the subsystem.
            dim: Dimension of the subsystem.
        """
        self._name = name
        self._dim = dim

    @property
    def name(self) -> str:
        """Name of subsystem."""
        return self._name

    @property
    def dim(self) -> int:
        """Dimension of subsystem."""
        return self._dim

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"Subsystem(name={self.name}, dim={self.dim})"

    def __eq__(self, other: "Subsystem") -> bool:
        if not isinstance(other, Subsystem):
            return False
        return self.name == other.name
