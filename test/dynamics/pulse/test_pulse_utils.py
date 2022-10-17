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

from qiskit_dynamics.pulse.pulse_utils import _get_dressed_state_decomposition
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
