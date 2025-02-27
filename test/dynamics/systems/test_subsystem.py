# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Tests for subsystem.py."""

from qiskit_dynamics.systems import Subsystem
from ..common import QiskitDynamicsTestCase


class TestSubsystem(QiskitDynamicsTestCase):
    """Tests for Subsystem class."""

    def test_properties(self):
        """Test basic properties."""
        s = Subsystem(name="Q0", dim=3)
        self.assertEqual(s.name, "Q0")
        self.assertEqual(s.dim, 3)

    def test_string_representations(self):
        """Test string representations."""
        s = Subsystem(name="Q0", dim=3)
        self.assertEqual(str(s), "Q0")
        self.assertEqual(repr(s), "Subsystem(name=Q0, dim=3)")

    def test_equality(self):
        """Test equality check."""
        s0 = Subsystem(name="Q0", dim=3)
        s1 = Subsystem(name="Q0", dim=3)
        s2 = Subsystem(name="Q1", dim=3)
        self.assertNotEqual(s0, s2)
        self.assertEqual(s0, s1)
