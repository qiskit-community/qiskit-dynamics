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
"""
Test pulse simulation utility functions.
"""

from ddt import ddt, data, unpack
import numpy as np

from qiskit import QiskitError

from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.pulse.pulse_utils import _get_dressed_state_decomposition, _get_lab_frame_static_hamiltonian
from ..common import QiskitDynamicsTestCase


class TestDressedStateDecomposition(QiskitDynamicsTestCase):
    """Tests _get_dressed_state_decomposition."""

    def test_non_hermitian_error(self):
        """Test error is raised with non-Hermitian operator."""

        with self.assertRaisesRegex(QiskitError, "received non-Hermitian operator."):
            _get_dressed_state_decomposition(np.array([[0., 1.], [0., 0.]]))

    def test_failed_sorting(self):
        """Test failed dressed state sorting."""

        with self.assertRaisesRegex(QiskitError, "sorting failed"):
            _get_dressed_state_decomposition(np.array([[0., 1.], [1., 0.]]))

    def test_reordering_eigenvalues(self):
        """Test correct ordering when the real-number ordering of the eigenvalues does not
        coincide with dressed-state-overlap ordering.
        """

        a = 0.2j
        abar = -0.2j
        mat = np.array([[0., a, 0.], [abar, 1., a], [0., abar, -1.]])

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
        self.Z = np.array([[1., 0.], [0., -1.]])
        self.X = np.array([[0., 1.], [1., 0.]])

    @unpack
    @data(("dense",), ("sparse",))
    def test_HamiltonianModel(self, evaluation_mode):
        """Test correct functioning on HamiltonianModel."""

        model = HamiltonianModel(
            static_operator=self.Z + self.X,
            operators=[self.X],
            rotating_frame=self.X,
            evaluation_mode=evaluation_mode
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    def test_HamiltonianModel_None(self):
        """Test correct functioning on HamiltonianModel if static_operator=None."""

        model = HamiltonianModel(
            static_operator=None,
            operators=[self.X],
            rotating_frame=self.X
        )

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
            evaluation_mode=evaluation_mode
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, self.Z + self.X)

    def test_LindbladModel_None(self):
        """Test correct functioning on Lindblad if static_hamiltonian=None."""

        model = LindbladModel(
            static_hamiltonian=None,
            hamiltonian_operators=[self.X],
            rotating_frame=self.X
        )

        output = _get_lab_frame_static_hamiltonian(model)
        self.assertAllClose(output, np.zeros((2, 2)))
