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
Tests for pulse.string_model_parser for importing qiskit.pulse model string representation.
"""

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator

from qiskit_dynamics.backend.backend_string_parser.hamiltonian_string_parser import (
    parse_backend_hamiltonian_dict,
)

from ...common import QiskitDynamicsTestCase


class TestHamiltonianParseExceptions(QiskitDynamicsTestCase):
    """Tests for preparse formatting exceptions."""

    def test_no_h_str(self):
        """Test no h_str empty raises error."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({})
        self.assertTrue("requires a non-empty 'h_str'" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": []})
        self.assertTrue("requires a non-empty 'h_str'" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": [""]})
        self.assertTrue("requires a non-empty 'h_str'" in str(qe.exception))

    def test_empty_qub(self):
        """Test qub dict empty raises error."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0|||D0"]})
        self.assertTrue("requires non-empty 'qub'" in str(qe.exception))

    def test_too_many_vertical_bars(self):
        """Test that too many vertical bars raises error."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0|||D0"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_single_vertical_bar(self):
        """Test that only a single vertical bar raises error."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0|D0"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_multiple_channel_error(self):
        """Test multiple channels raises error."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0||D0*D1"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_divider_no_channel(self):
        """Test that divider with no channel raises error."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0||"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0|"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_non_digit_after_channel(self):
        """Test that everything after the D or U is an int."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0||Da"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0||D1a"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))

    def test_nothing_after_channel(self):
        """Test that everything after the D or U is an int."""

        with self.assertRaises(QiskitError) as qe:
            parse_backend_hamiltonian_dict({"h_str": ["a * X0||U"], "qub": {0: 2}})
        self.assertTrue("does not conform" in str(qe.exception))


class TestParseHamiltonianDict(QiskitDynamicsTestCase):
    """Tests for parse_backend_hamiltonian_dict."""

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
        ham_dict = {"h_str": ["v*np.pi*Z0"], "qub": {"0": 2}, "vars": {"v": 2.1}}

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )
        self.assertAllClose(static_ham, 2.1 * np.pi * self.Z)
        self.assertTrue(not ham_ops)
        self.assertTrue(not channels)
        self.assertTrue(subsystem_dims_dict == {0: 2})

    def test_simple_single_q_system(self):
        """Test a basic system."""
        ham_dict = {
            "h_str": ["v*np.pi*Z0", "0.02*np.pi*X0||D0"],
            "qub": {"0": 2},
            "vars": {"v": 2.1},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        self.assertAllClose(static_ham, 2.1 * np.pi * self.Z)
        self.assertAllClose(ham_ops, [0.02 * np.pi * self.X])
        self.assertTrue(channels == ["d0"])
        self.assertTrue(subsystem_dims_dict == {0: 2})

    def test_simple_single_q_system_repeat_entries(self):
        """Test merging of terms with same channel or no channel."""
        ham_dict = {
            "h_str": ["v*np.pi*Z0", "0.02*np.pi*X0||D0", "v*np.pi*Z0", "0.02*np.pi*X0||D0"],
            "qub": {"0": 2},
            "vars": {"v": 2.1},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        self.assertAllClose(static_ham, 2 * 2.1 * np.pi * self.Z)
        self.assertAllClose(ham_ops, [2 * 0.02 * np.pi * self.X])
        self.assertTrue(channels == ["d0"])
        self.assertTrue(subsystem_dims_dict == {0: 2})

    def test_simple_single_q_system_repeat_entries_different_case(self):
        """Test merging of terms with same channel or no channel,
        with upper and lower case.
        """
        ham_dict = {
            "h_str": ["v*np.pi*Z0", "0.02*np.pi*X0||D0", "v*np.pi*Z0", "0.02*np.pi*X0||d0"],
            "qub": {"0": 2},
            "vars": {"v": 2.1},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        self.assertAllClose(static_ham, 2 * 2.1 * np.pi * self.Z)
        self.assertAllClose(ham_ops, [2 * 0.02 * np.pi * self.X])
        self.assertTrue(channels == ["d0"])
        self.assertTrue(subsystem_dims_dict == {0: 2})

    def test_simple_two_q_system(self):
        """Test a two qubit system."""

        ham_dict = {
            "h_str": [
                "v0*np.pi*Z0",
                "v1*np.pi*Z1",
                "j*np.pi*X0*Y1",
                "0.03*np.pi*X1||D1",
                "0.02*np.pi*X0||D0",
            ],
            "qub": {"0": 2, "1": 2},
            "vars": {"v0": 2.1, "v1": 2.0, "j": 0.02},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        ident = np.eye(2)
        self.assertAllClose(
            static_ham,
            2.1 * np.pi * np.kron(ident, self.Z)
            + 2.0 * np.pi * np.kron(self.Z, ident)
            + 0.02 * np.pi * np.kron(self.Y, self.X),
        )
        self.assertAllClose(
            ham_ops,
            [0.02 * np.pi * np.kron(ident, self.X), 0.03 * np.pi * np.kron(self.X, ident)],
        )
        self.assertTrue(channels == ["d0", "d1"])
        self.assertTrue(subsystem_dims_dict == {0: 2, 1: 2})

    def test_simple_two_q_system_measurement_channel(self):
        """Test a two qubit system with a measurement-labelled channel."""

        ham_dict = {
            "h_str": [
                "v0*np.pi*Z0",
                "v1*np.pi*Z1",
                "j*np.pi*X0*Y1",
                "0.03*np.pi*X1||M1",
                "0.02*np.pi*X0||D0",
            ],
            "qub": {"0": 2, "1": 2},
            "vars": {"v0": 2.1, "v1": 2.0, "j": 0.02},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        ident = np.eye(2)
        self.assertAllClose(
            static_ham,
            2.1 * np.pi * np.kron(ident, self.Z)
            + 2.0 * np.pi * np.kron(self.Z, ident)
            + 0.02 * np.pi * np.kron(self.Y, self.X),
        )
        self.assertAllClose(
            ham_ops,
            [0.02 * np.pi * np.kron(ident, self.X), 0.03 * np.pi * np.kron(self.X, ident)],
        )
        self.assertTrue(channels == ["d0", "m1"])
        self.assertTrue(subsystem_dims_dict == {0: 2, 1: 2})

    def test_single_oscillator_system(self):
        """Test single oscillator system."""

        ham_dict = {
            "h_str": ["v*np.pi*O0", "alpha*np.pi*O0*O0", "r*np.pi*X0||D0"],
            "qub": {"0": 4},
            "vars": {"v": 2.1, "alpha": -0.33, "r": 0.02},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        self.assertAllClose(static_ham, 2.1 * np.pi * self.N - 0.33 * np.pi * self.N * self.N)
        self.assertAllClose(ham_ops, [0.02 * np.pi * (self.a + self.adag)])
        self.assertTrue(channels == ["d0"])
        self.assertTrue(subsystem_dims_dict == {0: 4})

    def test_two_oscillator_system(self):
        """Test a two qubit system."""

        ham_dict = {
            "h_str": [
                "v0*np.pi*O0",
                "alpha0*np.pi*O0*O0",
                "v1*np.pi*O1",
                "alpha1*np.pi*O1*O1",
                "j*np.pi*X0*Y1",
                "0.03*np.pi*X1||D1",
                "0.02*np.pi*X0||D0",
            ],
            "qub": {"0": 4, "1": 4},
            "vars": {"v0": 2.1, "v1": 2.0, "alpha0": -0.33, "alpha1": -0.33, "j": 0.02},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        ident = np.eye(4)

        self.assertAllClose(
            static_ham,
            2.1 * np.pi * np.kron(ident, self.N)
            - 0.33 * np.pi * np.kron(ident, self.N * self.N)
            + 2.0 * np.pi * np.kron(self.N, ident)
            - 0.33 * np.pi * np.kron(self.N * self.N, ident)
            + 0.02 * np.pi * np.kron(-1j * (self.a - self.adag), self.a + self.adag),
        )
        self.assertAllClose(
            ham_ops,
            [
                0.02 * np.pi * np.kron(ident, self.a + self.adag),
                0.03 * np.pi * np.kron(self.a + self.adag, ident),
            ],
        )
        self.assertTrue(channels == ["d0", "d1"])
        self.assertTrue(subsystem_dims_dict == {0: 4, 1: 4})

    def test_single_q_high_dim(self):
        """Test single q system but higher dim."""
        ham_dict = {
            "h_str": ["v*np.pi*Z0", "0.02*np.pi*X0||D0"],
            "qub": {"0": 4},
            "vars": {"v": 2.1},
        }

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict
        )

        self.assertAllClose(static_ham, 2.1 * np.pi * (np.eye(4) - 2 * self.N))
        self.assertAllClose(ham_ops, [0.02 * np.pi * (self.a + self.adag)])
        self.assertTrue(channels == ["d0"])
        self.assertTrue(subsystem_dims_dict == {0: 4})

    def test_dagger(self):
        """Test correct parsing of dagger."""
        ham_dict = {
            "h_str": ["v*np.pi*dag(A0)"],
            "qub": {"0": 4},
            "vars": {"v": 2.1},
        }

        static_ham, _, _, _ = parse_backend_hamiltonian_dict(ham_dict)
        self.assertAllClose(static_ham, 2.1 * np.pi * self.adag)

    def test_5q_hamiltonian_reduced(self):
        """Test case for complicated Hamiltonian with reductions."""
        ham_dict = {
            "h_str": [
                "_SUM[i,0,4,wq{i}/2*(I{i}-Z{i})]",
                "_SUM[i,0,4,delta{i}/2*O{i}*O{i}]",
                "_SUM[i,0,4,-delta{i}/2*O{i}]",
                "_SUM[i,0,4,omegad{i}*X{i}||D{i}]",
                "jq1q2*Sp1*Sm2",
                "jq1q2*Sm1*Sp2",
                "jq3q4*Sp3*Sm4",
                "jq3q4*Sm3*Sp4",
                "jq0q1*Sp0*Sm1",
                "jq0q1*Sm0*Sp1",
                "jq2q3*Sp2*Sm3",
                "jq2q3*Sm2*Sp3",
                "omegad1*X0||U0",
                "omegad0*X1||U1",
                "omegad2*X1||U2",
                "omegad1*X2||U3",
                "omegad3*X2||U4",
                "omegad4*X3||U6",
                "omegad2*X3||U5",
                "omegad3*X4||U7",
            ],
            "qub": {"0": 4, "1": 4, "2": 4, "3": 4, "4": 4},
            "vars": {
                "delta0": -2.111793476400394,
                "delta1": -2.0894421352015744,
                "delta2": -2.1179183671068604,
                "delta3": -2.0410045431261215,
                "delta4": -2.1119885565086776,
                "jq0q1": 0.010495754104003914,
                "jq1q2": 0.010781715511200012,
                "jq2q3": 0.008920779377814226,
                "jq3q4": 0.008985191651087791,
                "omegad0": 0.9715458990879812,
                "omegad1": 0.9803812537440838,
                "omegad2": 0.9494756077681784,
                "omegad3": 0.9763998543087951,
                "omegad4": 0.9829308019780478,
                "wq0": 32.517894442809514,
                "wq1": 33.0948996120196,
                "wq2": 31.74518096417169,
                "wq3": 30.51062025552735,
                "wq4": 32.16082685025662,
            },
        }

        ident = np.eye(4, dtype=complex)
        X = self.a + self.adag
        X0 = np.kron(ident, X)
        X1 = np.kron(X, ident)
        N0 = np.kron(ident, self.N)
        N1 = np.kron(self.N, ident)

        # test case for subsystems [0, 1]

        w0 = ham_dict["vars"]["wq0"]
        w1 = ham_dict["vars"]["wq1"]
        delta0 = ham_dict["vars"]["delta0"]
        delta1 = ham_dict["vars"]["delta1"]
        j01 = ham_dict["vars"]["jq0q1"]
        omegad0 = ham_dict["vars"]["omegad0"]
        omegad1 = ham_dict["vars"]["omegad1"]
        omegad2 = ham_dict["vars"]["omegad2"]
        static_ham_expected = (
            w0 * N0
            + 0.5 * delta0 * (N0 @ N0 - N0)
            + w1 * N1
            + 0.5 * delta1 * (N1 @ N1 - N1)
            + j01 * (np.kron(self.a, self.adag) + np.kron(self.adag, self.a))
        )
        ham_ops_expected = np.array(
            [omegad0 * X0, omegad1 * X1, omegad1 * X0, omegad0 * X1, omegad2 * X1]
        )
        channels_expected = ["d0", "d1", "u0", "u1", "u2"]

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict, subsystem_list=[0, 1]
        )
        self.assertAllClose(static_ham, static_ham_expected)
        self.assertAllClose(ham_ops, ham_ops_expected)
        self.assertTrue(channels == channels_expected)
        self.assertTrue(subsystem_dims_dict == {0: 4, 1: 4})

        # test case for subsystems [3, 4]

        w3 = ham_dict["vars"]["wq3"]
        w4 = ham_dict["vars"]["wq4"]
        delta3 = ham_dict["vars"]["delta3"]
        delta4 = ham_dict["vars"]["delta4"]
        j34 = ham_dict["vars"]["jq3q4"]
        omegad3 = ham_dict["vars"]["omegad3"]
        omegad4 = ham_dict["vars"]["omegad4"]
        omegad2 = ham_dict["vars"]["omegad2"]
        static_ham_expected = (
            w3 * N0
            + 0.5 * delta3 * (N0 @ N0 - N0)
            + w4 * N1
            + 0.5 * delta4 * (N1 @ N1 - N1)
            + j34 * (np.kron(self.a, self.adag) + np.kron(self.adag, self.a))
        )
        ham_ops_expected = np.array(
            [omegad3 * X0, omegad4 * X1, omegad2 * X0, omegad4 * X0, omegad3 * X1]
        )
        channels_expected = ["d3", "d4", "u5", "u6", "u7"]

        static_ham, ham_ops, channels, subsystem_dims_dict = parse_backend_hamiltonian_dict(
            ham_dict, subsystem_list=[3, 4]
        )
        self.assertAllClose(static_ham, static_ham_expected)
        self.assertAllClose(ham_ops, ham_ops_expected)
        self.assertTrue(channels == channels_expected)
        self.assertTrue(subsystem_dims_dict == {3: 4, 4: 4})
