# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Tests for pulse_utils.py."""

from re import sub
import numpy as np
from collections import Counter
from qiskit_dynamics.pulse.pulse_utils import (
    compute_probabilities,
    convert_to_dressed,
    generate_ham,
    labels_generator,
    sample_counts,
)

from .common import QiskitDynamicsTestCase


def basis_vec(ind, dimension):
    vec = np.zeros(dimension, dtype=complex)
    vec[ind] = 1.0
    return vec


def two_q_basis_vec(inda, indb, dimension):
    vec_a = basis_vec(inda, dimension)
    vec_b = basis_vec(indb, dimension)
    return np.kron(vec_a, vec_b)


def get_dressed_state_index(inda, indb, dimension, evectors):
    b_vec = two_q_basis_vec(inda, indb, dimension)
    overlaps = np.abs(evectors @ b_vec)
    return overlaps.argmax()


def get_dressed_state_and_energy_3x3(evals, inda, indb, dimension, evecs):
    ind = get_dressed_state_index(inda, indb, dimension, evecs)
    return evals[ind], evecs[ind]


class TestDressedStateConverter(QiskitDynamicsTestCase):
    """DressedStateConverter tests"""

    def dressed_tester(self, dressed_states, subsystem_dims):
        labels = labels_generator(subsystem_dims, array=True)
        str_labels = labels_generator(subsystem_dims, array=False)
        for str_label, label in zip(str_labels, labels):
            id = np.argmax(np.abs(dressed_states[str_label]))
            labels[id]
            self.assertTrue((labels[id] == label))

    def test_convert_to_dressed_single_q(self):
        """Test convert_to_dressed with a single 3 level qubit system."""

        subsystem_dims = [3]
        H0 = generate_ham(subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, dressed_list = convert_to_dressed(
            H0, subsystem_dims
        )
        self.dressed_tester(dressed_states, subsystem_dims)

    def test_convert_to_dressed_two_q_states(self):
        """Test convert_to_dressed with a 2 qubit system with 3 levels per qubit."""
        """also test state and energy using alternative method"""
        subsystem_dims = [3, 3]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, dressed_list = convert_to_dressed(
            H0, subsystem_dims
        )

        dim = subsystem_dims[0]
        evals, evectors = np.linalg.eigh(H0)

        E00, dressed00 = get_dressed_state_and_energy_3x3(evals, 0, 0, dim, evectors.transpose())
        E01, dressed01 = get_dressed_state_and_energy_3x3(evals, 0, 1, dim, evectors.transpose())
        E10, dressed10 = get_dressed_state_and_energy_3x3(evals, 1, 0, dim, evectors.transpose())
        E11, dressed11 = get_dressed_state_and_energy_3x3(evals, 1, 1, dim, evectors.transpose())

        self.assertTrue(np.max(dressed00 - dressed_states["00"] < 1e-12))
        self.assertTrue(np.max(dressed01 - dressed_states["01"] < 1e-12))
        self.assertTrue(np.max(dressed10 - dressed_states["10"] < 1e-12))
        self.assertTrue(np.max(dressed11 - dressed_states["11"] < 1e-12))

        self.assertTrue(E00 - dressed_evals["00"] < 1e-12)
        self.assertTrue(E01 - dressed_evals["01"] < 1e-12)
        self.assertTrue(E10 - dressed_evals["10"] < 1e-12)
        self.assertTrue(E11 - dressed_evals["11"] < 1e-12)

        self.dressed_tester(dressed_states, subsystem_dims)

    def test_convert_to_dressed_three_q_states(self):
        """Test convert_to_dressed with a 3 qubit system with different levels per qubit."""
        subsystem_dims = [3, 4, 5]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, dressed_list = convert_to_dressed(
            H0, subsystem_dims
        )

        self.dressed_tester(dressed_states, subsystem_dims)

    def test_convert_to_dressed_three_q_states_high(self):
        """Test convert_to_dressed with a 3 qubit system with different levels per qubit."""
        subsystem_dims = [3, 8, 4]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, dressed_list = convert_to_dressed(
            H0, subsystem_dims
        )

        self.dressed_tester(dressed_states, subsystem_dims)


class TestComputeandSampleProbabilities(QiskitDynamicsTestCase):
    """
    How do we test compute probabilities? We can just take our systems
    """

    def test_compute_and_sample_probabilities_1q(self):
        "Test compute_probabilities for a 1q system"
        subsystem_dims = [3]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, _, _, _ = convert_to_dressed(H0, subsystem_dims)
        state = [0, 1 / np.sqrt(2), 1 / np.sqrt(2)]
        probs = compute_probabilities(state, dressed_states=dressed_states)

        self.assertTrue(1 - sum(probs.values()) < 1e-12)
        self.assertTrue(0.5 - probs["1"] < 1e-4)
        self.assertTrue(0.5 - probs["2"] < 1e-4)

        samples = sample_counts(probs, 1000)
        counts = Counter(samples)
        self.assertTrue(counts["1"] > 450)
        self.assertTrue(counts["2"] > 450)

    def test_compute_and_sample_probabilities_2q(self):
        "Test compute_probabilities for a 2q system"
        subsystem_dims = [3, 4]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, _, _, _ = convert_to_dressed(H0, subsystem_dims)
        state1 = [1 / np.sqrt(2), 0, 1 / np.sqrt(2)]
        state2 = [0, 1, 0, 0]
        state = np.kron(state2, state1)
        probs = compute_probabilities(state, dressed_states=dressed_states)
        self.assertTrue(1 - sum(probs.values()) < 1e-12)

        self.assertTrue(0.5 - probs["12"] < 1e-4)
        self.assertTrue(0.5 - probs["10"] < 1e-4)

        samples = sample_counts(probs, 1000)
        counts = Counter(samples)
        self.assertTrue(counts["12"] > 450)
        self.assertTrue(counts["10"] > 450)

    def test_compute_and_sample_probabilities_3q(self):
        "Test compute_probabilities for a 3q system"
        subsystem_dims = [3, 6, 3]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, _, _, _ = convert_to_dressed(H0, subsystem_dims)
        state1 = [1 / np.sqrt(2), 0, 1 / np.sqrt(2)]
        state2 = [0, 1 / np.sqrt(4), 0, 1 / np.sqrt(4), 1 / np.sqrt(4), 1 / np.sqrt(4)]
        state3 = [1, 0, 0]
        state = np.kron(state3, state2)
        state = np.kron(state, state1)
        probs = compute_probabilities(state, dressed_states=dressed_states)
        self.assertTrue(1 - sum(probs.values()) < 1e-12)

        self.assertTrue(0.125 - probs["010"] < 1e-4)
        self.assertTrue(0.125 - probs["030"] < 1e-4)
        self.assertTrue(0.125 - probs["040"] < 1e-4)
        self.assertTrue(0.125 - probs["050"] < 1e-4)
        self.assertTrue(0.125 - probs["012"] < 1e-4)
        self.assertTrue(0.125 - probs["032"] < 1e-4)
        self.assertTrue(0.125 - probs["042"] < 1e-4)
        self.assertTrue(0.125 - probs["052"] < 1e-4)

        samples = sample_counts(probs, 1000)
        counts = Counter(samples)
        self.assertTrue(counts["010"] > 100)
        self.assertTrue(counts["030"] > 100)
        self.assertTrue(counts["040"] > 100)
        self.assertTrue(counts["050"] > 100)
        self.assertTrue(counts["012"] > 100)
        self.assertTrue(counts["032"] > 100)
        self.assertTrue(counts["042"] > 100)
        self.assertTrue(counts["052"] > 100)
