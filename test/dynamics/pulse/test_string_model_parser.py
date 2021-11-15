# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for pulse.string_model_parser for importing qiskit.pulse model string representation.
"""

from qiskit import QiskitError

from qiskit_dynamics.pulse.string_model_parser import hamiltonian_pre_parse_exceptions
from ..common import QiskitDynamicsTestCase

class TestHamiltonianPreParseExceptions(QiskitDynamicsTestCase):
    """Tests for preparse formatting exceptions."""
    ####################################################################################################
    # reconsider where this lives, but for now we need to test
    ###################################################################################################

    def test_no_h_str(self):
        """Test no h_str empty raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({})
        self.assertTrue("requires a non-empty 'h_str'" in str(qe.exception))

    def test_empty_qub(self):
        """Test qub dict empty raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({'h_str': ['a * X0|||D0']})
        self.assertTrue("requires non-empty 'qub'" in str(qe.exception))

    def test_too_many_vertical_bars(self):
        """Test that too many vertical bars raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({'h_str': ['a * X0|||D0'],
                                              'qub': {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_too_few_vertical_bars(self):
        """Test that too few vertical bars raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({'h_str': ['a * X0 * D0'],
                                              'qub': {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_multiple_channel_error(self):
        """Test multiple channels raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({'h_str': ['a * X0||D0*D1'],
                                              'qub': {0: 2}})
        self.assertTrue("more than one channel label" in str(qe.exception))

    def test_divider_no_channel(self):
        """Test that divider with no channel raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({'h_str': ['a * X0||'],
                                              'qub': {0: 2}})
        self.assertTrue("Operator-channel divider" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({'h_str': ['a * X0|'],
                                              'qub': {0: 2}})
        self.assertTrue("Operator-channel divider" in str(qe.exception))
