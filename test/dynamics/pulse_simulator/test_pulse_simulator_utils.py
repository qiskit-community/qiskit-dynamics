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
Test pulse simulation utility functions.
"""

from ddt import ddt, data, unpack
import numpy as np

from qiskit import QiskitError

from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.pulse_simulator.pulse_simulator_utils import (
    _get_dressed_state_decomposition,
    _get_lab_frame_static_hamiltonian,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
)
from ..common import QiskitDynamicsTestCase


class TestDressedStateDecomposition(QiskitDynamicsTestCase):
    """Tests _get_dressed_state_decomposition."""

    def test_non_hermitian_error(self):
        """Test error is raised with non-Hermitian operator."""

        with self.assertRaisesRegex(QiskitError, "received non-Hermitian operator."):
            _get_dressed_state_decomposition(np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_failed_sorting(self):
        """Test failed dressed state sorting."""

        with self.assertRaisesRegex(QiskitError, "sorting failed"):
            _get_dressed_state_decomposition(np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_reordering_eigenvalues(self):
        """Test correct ordering when the real-number ordering of the eigenvalues does not
        coincide with dressed-state-overlap ordering.
        """

        a = 0.2j
        abar = -0.2j
        mat = np.array([[0.0, a, 0.0], [abar, 1.0, a], [0.0, abar, -1.0]])

        # compute and manually re-order
        evals, evecs = np.linalg.eigh(mat)
        expected_dressed_evals = np.array([evals[1], evals[2], evals[0]])
        expected_dressed_states = np.array([evecs[:, 1], evecs[:, 2], evecs[:, 0]]).transpose()

        # compare
        dressed_evals, dressed_states = _get_dressed_state_decomposition(mat)
        self.assertAllClose(dressed_evals, expected_dressed_evals)
        self.assertAllClose(dressed_states, expected_dressed_states)


@ddt
class TestLabFrameStaticHamiltonian(QiskitDynamicsTestCase):
    """Tests _get_lab_frame_static_hamiltonian."""

    def setUp(self):
        self.Z = np.array([[1.0, 0.0], [0.0, -1.0]])
        self.X = np.array([[0.0, 1.0], [1.0, 0.0]])

    @unpack
    @data(("dense",), ("sparse",))
    def test_HamiltonianModel(self, evaluation_mode):
        """Test correct functioning on HamiltonianModel."""

        model = HamiltonianModel(
            static_operator=self.Z + self.X,
            operators=[self.X],
            rotating_frame=self.X,
            evaluation_mode=evaluation_mode,
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    def test_HamiltonianModel_None(self):
        """Test correct functioning on HamiltonianModel if static_operator=None."""

        model = HamiltonianModel(static_operator=None, operators=[self.X], rotating_frame=self.X)

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, np.zeros((2, 2)))

    @unpack
    @data(("dense",), ("sparse",), ("dense_vectorized",), ("sparse_vectorized",))
    def test_LindbladModel(self, evaluation_mode):
        """Test correct functioning on LindbladModel."""

        model = LindbladModel(
            static_hamiltonian=self.Z + self.X,
            hamiltonian_operators=[self.X],
            rotating_frame=self.X,
            evaluation_mode=evaluation_mode,
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    def test_LindbladModel_None(self):
        """Test correct functioning on Lindblad if static_hamiltonian=None."""

        model = LindbladModel(
            static_hamiltonian=None, hamiltonian_operators=[self.X], rotating_frame=self.X
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, np.zeros((2, 2)))


class Test_get_memory_slot_probabilities(QiskitDynamicsTestCase):
    """Test _get_memory_slot_probabilities."""

    def test_trivial_case(self):
        """Test trivial case where no re-ordering is done."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.4, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict, memory_slot_indices=[0, 1, 2]
        )
        self.assertDictEqual(output, probability_dict)

    def test_basic_reordering(self):
        """Test case with simple re-ordering."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.4, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict, memory_slot_indices=[2, 0, 1]
        )
        self.assertDictEqual(output, {"000": 0.25, "100": 0.3, "020": 0.4, "001": 0.05})

    def test_extra_memory_slots(self):
        """Test case with more memory slots than there are digits in probability_dict keys."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.4, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict,
            memory_slot_indices=[3, 0, 1],
        )
        self.assertDictEqual(output, {"0000": 0.25, "1000": 0.3, "0020": 0.4, "0001": 0.05})

    def test_bound_and_merging(self):
        """Test case with max outcome bound."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.2, "100": 0.2, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict,
            memory_slot_indices=[2, 0, 1],
            num_memory_slots=4,
            max_outcome_value=1,
        )
        self.assertDictEqual(output, {"0000": 0.25, "0100": 0.3, "0010": 0.4, "0001": 0.05})


class Test_sample_probability_dict(QiskitDynamicsTestCase):
    """Test _sample_probability_dict."""

    def test_correct_formatting(self):
        """Basic test case."""
        probability_dict = {"a": 0.1, "b": 0.12, "c": 0.78}
        seed = 3948737
        outcome = _sample_probability_dict(probability_dict, shots=100, seed=seed)

        rng = np.random.default_rng(seed=seed)
        expected = rng.choice(["a", "b", "c"], size=100, replace=True, p=[0.1, 0.12, 0.78])

        for a, b in zip(outcome, expected):
            self.assertTrue(a == b)


class Test_get_counts_from_samples(QiskitDynamicsTestCase):
    """Test _get_counts_from_samples."""

    def test_basic_counting(self):
        """Basic test case."""
        samples = ["00", "01", "00", "20", "01", "01", "20"]
        output = _get_counts_from_samples(samples)
        self.assertDictEqual(output, {"00": 2, "01": 3, "20": 2})