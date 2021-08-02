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

"""Tests for operator_collections.py"""

from qiskit.circuit.library.standard_gates import z
from qiskit_dynamics.models import operator_collections
from qiskit_dynamics.signals.signals import SignalList
import numpy as np
import numpy.random as rand
from scipy.linalg import expm
from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import GeneratorModel
from qiskit_dynamics.models.operator_collections import (
    DenseOperatorCollection,
    DenseLindbladCollection,
    DenseVectorizedLindbladCollection,
)
from qiskit_dynamics.signals import Signal
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestDenseOperatorCollection(QiskitDynamicsTestCase):
    """Tests for GeneratorModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        self.test_operator_list = Array([self.X, self.Y, self.Z])

        self.signals = SignalList([Signal(1, j / 3) for j in range(3)])
        self.sigvals = np.real(self.signals.complex_value(2))

        self.simple_collection = DenseOperatorCollection(self.test_operator_list, drift=None)

    def test_known_values_basic_functionality(self):
        res = self.simple_collection(self.sigvals)
        self.assertAllClose(res, Array([[-0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 0.5 + 0j]]))

        res = (DenseOperatorCollection(self.test_operator_list, drift=np.eye(2)))(self.sigvals)
        self.assertAllClose(res, Array([[0.5 + 0j, 1.0 + 0.5j], [1.0 - 0.5j, 1.5 + 0j]]))

    def test_basic_functionality_pseudorandom(self):
        rand.seed(0)
        vals = rand.uniform(-1, 1, 32) + 1j * rand.uniform(-1, 1, (10, 32))
        arr = rand.uniform(-1, 1, (32, 128, 128))
        res = (DenseOperatorCollection(arr))(vals)
        for i in range(10):
            total = 0
            for j in range(32):
                total = total + vals[i, j] * arr[j]
            self.assertAllClose(res[i], total)


class TestDenseOperatorCollectionJax(TestDenseOperatorCollection, TestJaxBase):
    """Jax version of TestGeneratorModel tests.

    Note: This class has no body but contains tests due to inheritance.
    """


class TestDenseLindbladCollection(QiskitDynamicsTestCase):
    """Tests for GeneratorModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

    def test_basic_functionality_pseudorandom(self):
        """Tests to ensure that the Lindblad Master Equation
        is being implemented correctly. Chosen so that, if future
        optimizations are made, the actual math is still correct"""
        rand.seed(2134024)
        n = 16
        k = 8
        m = 4
        l = 2
        hamiltonian_operators = rand.uniform(-1, 1, (k, n, n))
        dissipator_operators = rand.uniform(-1, 1, (m, n, n))
        drift = rand.uniform(-1, 1, (n, n))
        rho = rand.uniform(-1, 1, (n, n))
        ham_sig_vals = rand.uniform(-1, 1, (k))
        dis_sig_vals = rand.uniform(-1, 1, (m))

        # Test first that having no drift or dissipator is OK
        ham_only_collection = DenseLindbladCollection(
            hamiltonian_operators, drift=np.zeros((n, n)), dissipator_operators=None
        )
        hamiltonian = np.tensordot(ham_sig_vals, hamiltonian_operators, axes=1)
        res = ham_only_collection([ham_sig_vals, None], rho)
        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(rho) - rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

        # Now, test adding a drift
        ham_drift_collection = DenseLindbladCollection(
            hamiltonian_operators, drift=drift, dissipator_operators=None
        )
        hamiltonian = np.tensordot(ham_sig_vals, hamiltonian_operators, axes=1) + drift
        res = ham_drift_collection([ham_sig_vals, None], rho)
        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(rho) - rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

        # Now, test if the dissipator terms also work as intended
        (dissipator_operators @ rho @ dissipator_operators).shape

        # Now, test if the dissipator terms also work as intended
        full_lindblad_collection = DenseLindbladCollection(
            hamiltonian_operators, drift=drift, dissipator_operators=dissipator_operators
        )
        res = full_lindblad_collection([ham_sig_vals, dis_sig_vals], rho)
        hamiltonian = np.tensordot(ham_sig_vals, hamiltonian_operators, axes=1) + drift
        ham_terms = -1j * (hamiltonian.dot(rho) - rho.dot(hamiltonian))
        dis_anticommutator = (-1 / 2) * np.tensordot(
            dis_sig_vals,
            np.conjugate(np.transpose(dissipator_operators, [0, 2, 1])) @ dissipator_operators,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(rho) + rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            dis_sig_vals,
            dissipator_operators
            @ rho
            @ np.conjugate(np.transpose(dissipator_operators, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(ham_terms + dis_anticommutator + dis_extra, res)

        # Now, test if vectorization works as intended
        rhos = rand.uniform(-1, 1, (l, n, n))
        res = full_lindblad_collection([ham_sig_vals, dis_sig_vals], rhos)
        for i in range(l):
            self.assertAllClose(
                res[i], full_lindblad_collection([ham_sig_vals, dis_sig_vals], rhos[i])
            )


class TestDenseLindbladCollectionJax(TestDenseOperatorCollection, TestJaxBase):
    """Jax version of TestGeneratorModel tests.

    Note: This class has no body but contains tests due to inheritance.
    """


class TestDenseVectorizedLindbladCollection(QiskitDynamicsTestCase):
    """Tests for DenseVectorizedLindbladCollection.
    Assumes that DenseLindbladCollection is functioning
    correctly, and–as such–only checks that the results
    from DenseVectorizedLindbladCollection are consistent
    with those from DenseLindbladCollection"""

    def setUp(self) -> None:
        pass

    def test_consistency_pseudorandom(self):
        rand.seed(123098341)
        n = 16
        k = 4
        m = 2
        r = lambda *args: rand.uniform(-1, 1, [*args]) + 1j * rand.uniform(-1, 1, [*args])

        rand_ham = r(k, n, n)
        rand_dis = r(m, n, n)
        rand_dft = r(n, n)
        rho = r(n, n)
        t = r()
        rand_ham_sigs = SignalList([Signal(r(), r(), r()) for j in range(k)])
        rand_dis_sigs = SignalList([Signal(r(), r(), r()) for j in range(m)])

        # Check consistency when hamiltonian, drift, and dissipator terms defined
        stdLindblad = DenseLindbladCollection(
            rand_ham, drift=rand_dft, dissipator_operators=rand_dis
        )
        vecLindblad = DenseVectorizedLindbladCollection(
            rand_ham, drift=rand_dft, dissipator_operators=rand_dis
        )

        a = stdLindblad.evaluate_hamiltonian(rand_ham_sigs(t)).flatten(order="F")
        b = vecLindblad.evaluate_hamiltonian(rand_ham_sigs(t))
        self.assertAllClose(a, b)

        a = stdLindblad.evaluate_rhs([rand_ham_sigs(t), rand_dis_sigs(t)], rho).flatten(order="F")
        b = vecLindblad.evaluate_rhs([rand_ham_sigs(t), rand_dis_sigs(t)], rho.flatten(order="F"))
        self.assertAllClose(a, b)

        # Check consistency when only hamiltonian and drift terms defined
        stdLindblad = DenseLindbladCollection(rand_ham, drift=rand_dft, dissipator_operators=None)
        vecLindblad = DenseVectorizedLindbladCollection(
            rand_ham, drift=rand_dft, dissipator_operators=None
        )

        a = stdLindblad.evaluate_hamiltonian(rand_ham_sigs(t)).flatten(order="F")
        b = vecLindblad.evaluate_hamiltonian(rand_ham_sigs(t))
        self.assertAllClose(a, b)

        a = stdLindblad.evaluate_rhs([rand_ham_sigs(t), 0], rho).flatten(order="F")
        b = vecLindblad.evaluate_rhs([rand_ham_sigs(t), 0], rho.flatten(order="F"))
        self.assertAllClose(a, b)
