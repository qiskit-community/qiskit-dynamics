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
Tests for pulse.operator_from_string
"""

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator

from qiskit_dynamics.pulse.operator_from_string import operator_from_string
from qiskit_dynamics.type_utils import to_array

from ..common import QiskitDynamicsTestCase


class TestOperatorFromString(QiskitDynamicsTestCase):
    """Test cases for operator_from_string."""

    def test_correct_single_ops_dim2(self):
        """Test that single operators give correct outputs."""

        ident = np.eye(2)
        a = np.array([[0., 1.],
                      [0., 0.]])
        adag = a.conj().transpose()
        N = np.diag(np.arange(2))
        X = a + adag
        Y = -1j * (a - adag)
        Z = ident - 2 * N


        self.assertAllClose(operator_from_string('X', 0, {0: 2}), np.array([[0., 1.], [1., 0.]]))
        self.assertAllClose(operator_from_string('Y', 0, {0: 2}), np.array([[0., -1j], [1j, 0.]]))
        self.assertAllClose(operator_from_string('Z', 0, {0: 2}), np.array([[1., 0.], [0., -1.]]))
        self.assertAllClose(operator_from_string('a', 0, {0: 2}), np.array([[0., 1.], [0., 0.]]))
        self.assertAllClose(operator_from_string('A', 0, {0: 2}), np.array([[0., 1.], [0., 0.]]))
        self.assertAllClose(operator_from_string('Sm', 0, {0: 2}), np.array([[0., 1.], [0., 0.]]))
        self.assertAllClose(operator_from_string('C', 0, {0: 2}), np.array([[0., 0.], [1., 0.]]))
        self.assertAllClose(operator_from_string('Sp', 0, {0: 2}), np.array([[0., 0.], [1., 0.]]))
        self.assertAllClose(operator_from_string('N', 0, {0: 2}), np.array([[0., 0.], [0., 1.]]))
        self.assertAllClose(operator_from_string('O', 0, {0: 2}), np.array([[0., 0.], [0., 1.]]))
        self.assertAllClose(operator_from_string('I', 0, {0: 2}), np.eye(2))

    def test_correct_single_ops_dim4(self):
        """Test that single operators give correct outputs."""

        sq2 = np.sqrt(2)
        sq3 = np.sqrt(3)
        ident = np.eye(4)
        a = np.array([[0., 1., 0., 0.],
                    [0., 0., sq2, 0.],
                    [0., 0., 0., sq3],
                    [0., 0., 0., 0.]])
        adag = a.conj().transpose()
        N = np.diag(np.arange(4))
        X = a + adag
        Y = -1j * (a - adag)
        Z = ident - 2 * N

        self.assertAllClose(operator_from_string('X', 0, {0: 4}), X)
        self.assertAllClose(operator_from_string('Y', 0, {0: 4}), Y)
        self.assertAllClose(operator_from_string('Z', 0, {0: 4}), Z)
        self.assertAllClose(operator_from_string('a', 0, {0: 4}), a)
        self.assertAllClose(operator_from_string('A', 0, {0: 4}), a)
        self.assertAllClose(operator_from_string('Sm', 0, {0: 4}), a)
        self.assertAllClose(operator_from_string('C', 0, {0: 4}), adag)
        self.assertAllClose(operator_from_string('Sp', 0, {0: 4}), adag)
        self.assertAllClose(operator_from_string('N', 0, {0: 4}), N)
        self.assertAllClose(operator_from_string('O', 0, {0: 4}), N)
        self.assertAllClose(operator_from_string('I', 0, {0: 4}), ident)

    def test_ident_before(self):
        """Test adding identity before the subsystem in question."""

        out = operator_from_string('X', 2, {0: 4, 1: 2, 2: 2})
        expected = np.kron(np.array([[0., 1.], [1., 0.]]), np.eye(8))
        self.assertAllClose(out, expected)

    def test_ident_after(self):
        """Test adding identity after the subsystem in question."""

        out = operator_from_string('Z', 0, {0: 2, 1: 2, 2: 4})
        expected = np.kron(np.eye(8), np.array([[1., 0.], [0., -1.]]))
        self.assertAllClose(out, expected)

    def test_ident_before_and_after(self):
        """Test adding identity before and after the subsystem in question."""

        out = operator_from_string('a', 1, {0: 2, 1: 2, 2: 4})
        expected = np.kron(np.kron(np.eye(4), np.array([[0., 1.], [0., 0.]])), np.eye(2))
        self.assertAllClose(out, expected)
