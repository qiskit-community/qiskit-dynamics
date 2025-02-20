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
Test backend utility functions.
"""

from ddt import ddt, data, unpack
import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info import Statevector

from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.backend.backend_utils import (
    _get_dressed_state_decomposition,
    _get_lab_frame_static_hamiltonian,
    _get_memory_slot_probabilities,
    _sample_probability_dict,
    _get_counts_from_samples,
    _get_subsystem_probabilities,
    _get_iq_data,
)
from ..common import QiskitDynamicsTestCase, JAXTestBase


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
    @data(("numpy",), ("scipy_sparse",))
    def test_HamiltonianModel(self, array_library):
        """Test correct functioning on HamiltonianModel."""
        model = HamiltonianModel(
            static_operator=self.Z + self.X,
            operators=[self.X],
            rotating_frame=self.X,
            array_library=array_library,
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    @unpack
    @data(("numpy",), ("scipy_sparse",))
    def test_HamiltonianModel_diagonal(self, array_library):
        """Test correct functioning on HamiltonianModel with explicitly diagonal frame."""
        model = HamiltonianModel(
            static_operator=self.Z + self.X,
            operators=[self.X],
            rotating_frame=np.diag(self.Z),
            array_library=array_library,
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    @unpack
    @data(("numpy",), ("scipy_sparse",))
    def test_HamiltonianModel_lab_frame(self, array_library):
        """Test correct functioning on HamiltonianModel if no rotating frame."""
        model = HamiltonianModel(
            static_operator=self.Z + self.X,
            operators=[self.X],
            array_library=array_library,
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    def test_HamiltonianModel_None(self):
        """Test correct functioning on HamiltonianModel if static_operator=None."""

        model = HamiltonianModel(static_operator=None, operators=[self.X], rotating_frame=self.X)

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, np.zeros((2, 2)))

    @unpack
    @data(("numpy", False), ("scipy_sparse", False), ("numpy", True), ("scipy_sparse", True))
    def test_LindbladModel(self, array_library, vectorized):
        """Test correct functioning on LindbladModel."""

        model = LindbladModel(
            static_hamiltonian=self.Z + self.X,
            hamiltonian_operators=[self.X],
            rotating_frame=self.X,
            array_library=array_library,
            vectorized=vectorized,
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    @unpack
    @data(("numpy", False), ("scipy_sparse", False), ("numpy", True), ("scipy_sparse", True))
    def test_LindbladModel_diagonal(self, array_library, vectorized):
        """Test correct functioning on LindbladModel with explicitly diagonal frame."""

        model = LindbladModel(
            static_hamiltonian=self.Z + self.X,
            hamiltonian_operators=[self.X],
            rotating_frame=np.diag(self.Z),
            array_library=array_library,
            vectorized=vectorized,
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


class TestLabFrameStaticHamiltonianJAX(TestLabFrameStaticHamiltonian, JAXTestBase):
    """Tests _get_lab_frame_static_hamiltonian when in JAX mode."""


class Test_get_memory_slot_probabilities(QiskitDynamicsTestCase):
    """Test _get_memory_slot_probabilities."""

    def test_trivial_case(self):
        """Test trivial case where no re-ordering is done."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.4, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict, memory_slot_indices=[0, 1, 2], max_outcome_value=2
        )
        self.assertDictEqual(output, {hex(int(k, 3)): v for k, v in probability_dict.items()})

    def test_basic_reordering(self):
        """Test case with simple re-ordering."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.4, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict, memory_slot_indices=[2, 0, 1], max_outcome_value=2
        )
        expected = {"000": 0.25, "100": 0.3, "020": 0.4, "001": 0.05}

        self.assertDictEqual(output, {hex(int(k, 3)): v for k, v in expected.items()})

    def test_extra_memory_slots(self):
        """Test case with more memory slots than there are digits in probability_dict keys."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.4, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict, memory_slot_indices=[3, 0, 1], max_outcome_value=2
        )
        expected = {"0000": 0.25, "1000": 0.3, "0020": 0.4, "0001": 0.05}

        self.assertDictEqual(output, {hex(int(k, 3)): v for k, v in expected.items()})

    def test_bound_and_merging(self):
        """Test case with max outcome bound."""

        probability_dict = {"000": 0.25, "001": 0.3, "200": 0.2, "100": 0.2, "010": 0.05}

        output = _get_memory_slot_probabilities(
            probability_dict=probability_dict,
            memory_slot_indices=[2, 0, 1],
            num_memory_slots=4,
            max_outcome_value=1,
        )
        expected = {"0000": 0.25, "0100": 0.3, "0010": 0.4, "0001": 0.05}

        self.assertDictEqual(output, {hex(int(k, 2)): v for k, v in expected.items()})


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


class Test_get_subsystem_probabilities(QiskitDynamicsTestCase):
    """Test _get_subsystem_probabilities."""

    def test_basic(self):
        """Basic marginalization test case."""
        yf = Statevector(np.array([0.5, 1, 0, 0]) / np.sqrt(1.25), dims=(2, 2))
        prob_tensor = yf.probabilities().reshape(2, 2)
        sub_prob = _get_subsystem_probabilities(prob_tensor, 0)
        self.assertAllClose(sub_prob, [0.2, 0.8])


class Test_get_iq_data(QiskitDynamicsTestCase):
    """Test _get_iq_data."""

    def setUp(self):
        self.iq_to_counts = lambda iq_n: dict(
            zip(*np.unique(["1" if iq[0] > 0 else "0" for iq in iq_n], return_counts=True))
        )

    def test_basic_predict(self):
        """Basic predict test case."""
        iq_data = _get_iq_data(
            state=Statevector(np.array([0.5, 1]) / np.sqrt(1.25)),
            measurement_subsystems=[0],
            iq_centers=[[(1, 0), (-1, 0)]],
            iq_width=0.1,
            shots=100,
            memory_slot_indices=[1],
            seed=83248,
        )
        counts = self.iq_to_counts(iq_data[:, 1, :])
        self.assertDictEqual(counts, {"0": 74, "1": 26})
        counts = self.iq_to_counts(iq_data[:, 0, :])
        self.assertDictEqual(counts, {"0": 100})

    def test_multi_qubit_predict(self):
        """Multi qubit predict test case."""
        iq_data = _get_iq_data(
            state=Statevector(np.array([0.5, 1, 0, 0]) / np.sqrt(1.25)),
            measurement_subsystems=[0, 1],
            iq_centers=[[(1, 0), (-1, 0)], [(1, 0), (-1, 0)]],
            iq_width=0.1,
            shots=100,
            memory_slot_indices=[0, 1],
            seed=83248,
        )

        counts0 = self.iq_to_counts(iq_data[:, 0, :])
        counts1 = self.iq_to_counts(iq_data[:, 1, :])
        self.assertDictEqual(counts0, {"0": 74, "1": 26})
        self.assertDictEqual(counts1, {"1": 100})

    def test_multi_qubit_with_1d_subsystem(self):
        """Multi qubit with 1d (trivial) subsystems."""
        iq_data = _get_iq_data(
            state=Statevector(np.array([0.5, 1, 0, 0]) / np.sqrt(1.25), dims=(1, 1, 2, 2, 1, 1)),
            measurement_subsystems=[0, 1],
            iq_centers=[[(1, 0), (-1, 0)], [(1, 0), (-1, 0)]],
            iq_width=0.1,
            shots=100,
            memory_slot_indices=[0, 1],
            seed=83248,
        )

        counts0 = self.iq_to_counts(iq_data[:, 0, :])
        counts1 = self.iq_to_counts(iq_data[:, 1, :])
        self.assertDictEqual(counts0, {"0": 74, "1": 26})
        self.assertDictEqual(counts1, {"1": 100})

    def test_mixed_subsystem_predict(self):
        """Multi qubit predict test case."""
        iq_data = _get_iq_data(
            state=Statevector(
                np.kron(np.array([0.5, 1]), np.array([0, 0, 1])) / np.sqrt(1.25), dims=(3, 2)
            ),
            measurement_subsystems=[0, 1],
            iq_centers=[[(-1, -1), (1, -1), (1, 1)], [(1, 0), (-1, 0)]],
            iq_width=0.1,
            shots=100,
            memory_slot_indices=[0, 1],
            seed=83248,
        )

        def qutrit_iq_to_counts(iq_n):
            results = []
            for iq in iq_n:
                if iq[0] < 0 and iq[1] < 0:
                    results.append("0")
                elif iq[0] > 0 > iq[1]:
                    results.append("1")
                elif iq[0] > 0 and iq[1] > 0:
                    results.append("2")
            return dict(zip(*np.unique(results, return_counts=True)))

        counts0 = qutrit_iq_to_counts(iq_data[:, 0, :])
        counts1 = self.iq_to_counts(iq_data[:, 1, :])
        self.assertDictEqual(counts0, {"2": 100})
        self.assertDictEqual(counts1, {"0": 75, "1": 25})

    def test_multi_qubit_iq(self):
        """Multi qubit IQ test case."""

        iq_data_01 = _get_iq_data(
            state=Statevector(
                np.kron(np.array([0.5, 1]), np.array([0, 0, 1])) / np.sqrt(1.25), dims=(3, 2)
            ),
            measurement_subsystems=[0, 1],
            iq_centers=[[(-1, -1), (1, -1), (1, 1)], [(1, 0), (-1, 0)]],
            iq_width=0.1,
            shots=3,
            memory_slot_indices=[0, 1],
            seed=83248,
        )
        iq_data_10 = _get_iq_data(
            state=Statevector(
                np.kron(np.array([0.5, 1]), np.array([0, 0, 1])) / np.sqrt(1.25), dims=(3, 2)
            ),
            measurement_subsystems=[0, 1],
            iq_centers=[[(-1, -1), (1, -1), (1, 1)], [(1, 0), (-1, 0)]],
            iq_width=0.1,
            shots=3,
            memory_slot_indices=[1, 0],
            seed=83248,
        )

        true_result = np.array(
            [
                [[1.04443995, 1.02002225], [1.04436925, 0.0745472]],
                [[0.99445976, 1.13454847], [-1.05369125, -0.0284215]],
                [[1.00486134, 1.19089796], [-0.99570855, 0.10100498]],
            ]
        )
        self.assertAllClose(
            iq_data_01,
            true_result,
        )
        self.assertAllClose(
            iq_data_10,
            true_result[:, ::-1, :],
        )
