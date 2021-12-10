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

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator

from qiskit_dynamics.pulse.string_model_parser import hamiltonian_pre_parse_exceptions, parse_hamiltonian_dict
from qiskit_dynamics.type_utils import to_array

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
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0|||D0"]})
        self.assertTrue("requires non-empty 'qub'" in str(qe.exception))

    def test_too_many_vertical_bars(self):
        """Test that too many vertical bars raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0|||D0"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_too_few_vertical_bars(self):
        """Test that too few vertical bars raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0 * D0"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_multiple_channel_error(self):
        """Test multiple channels raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0||D0*D1"], "qub": {0: 2}})
        self.assertTrue("more than one channel label" in str(qe.exception))

    def test_divider_no_channel(self):
        """Test that divider with no channel raises error."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0||"], "qub": {0: 2}})
        self.assertTrue("Operator-channel divider" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0|"], "qub": {0: 2}})
        self.assertTrue("Operator-channel divider" in str(qe.exception))

    def test_non_digit_after_channel(self):
        """Test that everything after the D or U is an int."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0||Da"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0||D1a"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_nothing_after_channel(self):
        """Test that everything after the D or U is an int."""

        with self.assertRaises(QiskitError) as qe:
            hamiltonian_pre_parse_exceptions({"h_str": ["a * X0||U"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))


class TestParseHamiltonianDict(QiskitDynamicsTestCase):
    """Tests for parse_hamiltonian_dict."""

    def setUp(self):
        """Build operators."""
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data

        dim = 4
        self.a = np.array(np.diag(np.sqrt(range(1, dim)), k=1), dtype=complex)
        self.adag = self.a.conj().transpose()
        self.N = np.array(np.diag(range(dim)), dtype=complex)

    def test_only_static_terms(self):
        """Test a basic system."""
        ham_dict = {
                    'h_str': ['v*np.pi*Z0'],
                    'qub': {
                        '0': 2
                    },
                    'vars': {'v': 2.1}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)
        self.assertAllClose(static_ham, 2.1 * np.pi * self.Z)
        self.assertTrue(ham_ops == [])
        self.assertTrue(channels == [])

    def test_simple_single_q_system(self):
        """Test a basic system."""
        ham_dict = {
                    'h_str': ['v*np.pi*Z0', '0.02*np.pi*X0||D0'],
                    'qub': {
                        '0': 2
                    },
                    'vars': {'v': 2.1}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        self.assertAllClose(static_ham, 2.1 * np.pi * self.Z)
        self.assertAllClose(to_array(ham_ops), [0.02 * np.pi * self.X])
        self.assertTrue(channels == ['d0'])

    def test_simple_single_q_system_repeat_entries(self):
        """Test merging of terms with same channel or no channel."""
        ham_dict = {
                    'h_str': ['v*np.pi*Z0', '0.02*np.pi*X0||D0', 'v*np.pi*Z0', '0.02*np.pi*X0||D0'],
                    'qub': {
                        '0': 2
                    },
                    'vars': {'v': 2.1}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        self.assertAllClose(static_ham, 2 * 2.1 * np.pi * self.Z)
        self.assertAllClose(to_array(ham_ops), [2 * 0.02 * np.pi * self.X])
        self.assertTrue(channels == ['d0'])

    def test_simple_single_q_system_repeat_entries_different_case(self):
        """Test merging of terms with same channel or no channel,
        with upper and lower case.
        """
        ham_dict = {
                    'h_str': ['v*np.pi*Z0', '0.02*np.pi*X0||D0', 'v*np.pi*Z0', '0.02*np.pi*X0||d0'],
                    'qub': {
                        '0': 2
                    },
                    'vars': {'v': 2.1}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        self.assertAllClose(static_ham, 2 * 2.1 * np.pi * self.Z)
        self.assertAllClose(to_array(ham_ops), [2 * 0.02 * np.pi * self.X])
        self.assertTrue(channels == ['d0'])

    def test_simple_two_q_system(self):
        """Test a two qubit system."""

        ham_dict = {
                    'h_str': ['v0*np.pi*Z0', 'v1*np.pi*Z1', 'j*np.pi*X0*Y1', '0.03*np.pi*X1||D1', '0.02*np.pi*X0||D0'],
                    'qub': {
                        '0': 2,
                        '1': 2
                    },
                    'vars': {'v0': 2.1, 'v1': 2.0, 'j': 0.02}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        ident = np.eye(2)
        self.assertAllClose(static_ham, 2.1 * np.pi * np.kron(ident, self.Z) + 2.0 * np.pi * np.kron(self.Z, ident) + 0.02 * np.pi * np.kron(self.Y, self.X))
        self.assertAllClose(to_array(ham_ops), [0.02 * np.pi * np.kron(ident, self.X), 0.03 * np.pi * np.kron(self.X, ident)])
        self.assertTrue(channels == ['d0', 'd1'])


    def test_simple_two_q_system_measurement_channel(self):
        """Test a two qubit system with a measurement-labelled channel."""

        ham_dict = {
                    'h_str': ['v0*np.pi*Z0', 'v1*np.pi*Z1', 'j*np.pi*X0*Y1', '0.03*np.pi*X1||M1', '0.02*np.pi*X0||D0'],
                    'qub': {
                        '0': 2,
                        '1': 2
                    },
                    'vars': {'v0': 2.1, 'v1': 2.0, 'j': 0.02}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        ident = np.eye(2)
        self.assertAllClose(static_ham, 2.1 * np.pi * np.kron(ident, self.Z) + 2.0 * np.pi * np.kron(self.Z, ident) + 0.02 * np.pi * np.kron(self.Y, self.X))
        self.assertAllClose(to_array(ham_ops), [0.02 * np.pi * np.kron(ident, self.X), 0.03 * np.pi * np.kron(self.X, ident)])
        self.assertTrue(channels == ['d0', 'm1'])


    def test_single_oscillator_system(self):
        """Test single oscillator system."""

        ham_dict = {
                    'h_str': ['v*np.pi*O0', 'alpha*np.pi*O0*O0', 'r*np.pi*X0||D0'],
                    'qub': {
                        '0': 4
                    },
                    'vars': {'v': 2.1, 'alpha': -0.33, 'r': 0.02}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        self.assertAllClose(static_ham, 2.1 * np.pi * self.N - 0.33 * np.pi * self.N * self.N)
        self.assertAllClose(to_array(ham_ops), [0.02 * np.pi * (self.a + self.adag)])
        self.assertTrue(channels == ['d0'])


    def test_two_oscillator_system(self):
        """Test a two qubit system."""

        ham_dict = {
                    'h_str': ['v0*np.pi*O0', 'alpha0*np.pi*O0*O0',
                              'v1*np.pi*O1', 'alpha1*np.pi*O1*O1',
                              'j*np.pi*X0*Y1', '0.03*np.pi*X1||D1',
                              '0.02*np.pi*X0||D0'],
                    'qub': {
                        '0': 4,
                        '1': 4
                    },
                    'vars': {'v0': 2.1, 'v1': 2.0, 'alpha0': -0.33, 'alpha1': -0.33, 'j': 0.02}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        ident = np.eye(4)

        self.assertAllClose(static_ham, 2.1 * np.pi * np.kron(ident, self.N)
                                        - 0.33 * np.pi * np.kron(ident, self.N * self.N)
                                        + 2.0 * np.pi * np.kron(self.N, ident)
                                        - 0.33 * np.pi * np.kron(self.N * self.N, ident)
                                        + 0.02 * np.pi * np.kron(-1j * (self.a - self.adag), self.a + self.adag))
        self.assertAllClose(to_array(ham_ops), [0.02 * np.pi * np.kron(ident, self.a + self.adag), 0.03 * np.pi * np.kron(self.a + self.adag, ident)])
        self.assertTrue(channels == ['d0', 'd1'])


    def test_single_q_high_dim(self):
        """Test single q system but higher dim."""
        ham_dict = {
                    'h_str': ['v*np.pi*Z0', '0.02*np.pi*X0||D0'],
                    'qub': {
                        '0': 4
                    },
                    'vars': {'v': 2.1}
                }

        static_ham, ham_ops, channels = parse_hamiltonian_dict(ham_dict)

        ##############################################################################################
        # Is this what we want it to be?
        ##############################################################################################
        self.assertAllClose(static_ham, 2.1 * np.pi * (np.eye(4) - 2 * self.N))
        self.assertAllClose(to_array(ham_ops), [0.02 * np.pi * (self.a + self.adag)])
        self.assertTrue(channels == ['d0'])
