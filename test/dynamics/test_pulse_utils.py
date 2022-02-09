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

import numpy as np
from qiskit_dynamics.pulse.pulse_utils import convert_to_dressed, generate_ham, labels_generator
from qiskit_dynamics.pulse.test_pulse_utils_utils import get_dressed_state_and_energy_3x3

from .common import QiskitDynamicsTestCase

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
        dressed_states, dressed_freqs, dressed_evals, dressed_list= convert_to_dressed(H0, subsystem_dims)
        self.dressed_tester(dressed_states, subsystem_dims)

    def test_convert_to_dressed_two_q_states(self):
        """Test convert_to_dressed with a 2 qubit system with 3 levels per qubit."""
        subsystem_dims=[3,3]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, dressed_list= convert_to_dressed(H0, subsystem_dims)


        dim = subsystem_dims[0]
        evals, evectors = np.linalg.eigh(H0)

        E00, dressed00 = get_dressed_state_and_energy_3x3(evals, 0, 0, dim, evectors.transpose())
        E01, dressed01 = get_dressed_state_and_energy_3x3(evals, 0, 1, dim, evectors.transpose())
        E10, dressed10 = get_dressed_state_and_energy_3x3(evals, 1, 0, dim, evectors.transpose())
        E11, dressed11 = get_dressed_state_and_energy_3x3(evals, 1, 1, dim, evectors.transpose())

        self.assertTrue(np.max(dressed00 - dressed_states['00'] < 1e-12))
        self.assertTrue(np.max(dressed01 - dressed_states['01'] < 1e-12))
        self.assertTrue(np.max(dressed10 - dressed_states['10'] < 1e-12))
        self.assertTrue(np.max(dressed11 - dressed_states['11'] < 1e-12))

        self.assertTrue(E00 - dressed_evals['00'] < 1e-12)
        self.assertTrue(E01 - dressed_evals['01'] < 1e-12)
        self.assertTrue(E10 - dressed_evals['10'] < 1e-12)
        self.assertTrue(E11 - dressed_evals['11'] < 1e-12)

        self.dressed_tester(dressed_states, subsystem_dims)

    
    def test_convert_to_dressed_three_q_states(self):
        """Test convert_to_dressed with a 3 qubit system with different levels per qubit."""
        subsystem_dims=[3,4,5]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, dressed_list= convert_to_dressed(H0, subsystem_dims)

        self.dressed_tester(dressed_states, subsystem_dims)


    def test_convert_to_dressed_three_q_states_high(self):
        """Test convert_to_dressed with a 3 qubit system with different levels per qubit."""
        subsystem_dims=[3,8,4]
        H0 = generate_ham(subsystem_dims=subsystem_dims)
        dressed_states, dressed_freqs, dressed_evals, dressed_list= convert_to_dressed(H0, subsystem_dims)

        self.dressed_tester(dressed_states, subsystem_dims)






class TestComputeandSampleProbabilities(QiskitDynamicsTestCase):
    