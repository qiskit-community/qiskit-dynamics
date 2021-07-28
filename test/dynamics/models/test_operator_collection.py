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
from qiskit_dynamics.models.operator_collections import DenseOperatorCollection,DenseLindbladCollection
from qiskit_dynamics.signals import Signal
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestDenseOperatorCollection(QiskitDynamicsTestCase):
    """Tests for GeneratorModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        self.test_operator_list = Array([self.X,self.Y,self.Z])

        self.signals = SignalList([Signal(1,j/3) for j in range(3)])
        self.sigvals = np.real(self.signals.complex_value(2))

        self.simple_collection = DenseOperatorCollection(self.test_operator_list,drift = None)

    def test_known_values_basic_functionality(self):
        res = self.simple_collection(self.sigvals)
        self.assertAllClose(res,Array([[-0.5+0j, 1. + 0.5j], [1. - 0.5j, 0.5+0j]]))

        res = (DenseOperatorCollection(self.test_operator_list,drift=np.eye(2)))(self.sigvals)
        self.assertAllClose(res,Array([[0.5+0j, 1. + 0.5j], [1. - 0.5j, 1.5+0j]]))

    def test_basic_functionality_pseudorandom(self):
        rand.seed(0)
        vals = rand.uniform(-1,1,32)+1j*rand.uniform(-1,1,(10,32))
        arr = rand.uniform(-1,1,(32,128,128))
        res = (DenseOperatorCollection(arr))(vals)
        for i in range(10):
            total = 0
            for j in range(32):
                total = total + vals[i,j]*arr[j]
            self.assertAllClose(res[i],total)
    
    def test_apply_function(self):
        res = self.simple_collection(self.sigvals)
        self.simple_collection.apply_function_to_operators(lambda x: x/2.1231)
        newres = self.simple_collection(self.sigvals)
        self.assertAllClose(newres*2.1231,res)
        self.simple_collection.apply_function_to_operators(lambda x: 2.1231*x)

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
        hamiltonian_operators = rand.uniform(-1,1,(k,n,n))
        dissipator_operators = rand.uniform(-1,1,(m,n,n))
        drift = rand.uniform(-1,1,(n,n))
        rho = rand.uniform(-1,1,(n,n))
        ham_sig_vals = rand.uniform(-1,1,(k))
        dis_sig_vals = rand.uniform(-1,1,(m))

        #Test first that having no drift or dissipator is OK
        ham_only_collection = DenseLindbladCollection(hamiltonian_operators,drift = None, dissipator_operators = None)
        hamiltonian = np.tensordot(ham_sig_vals,hamiltonian_operators,axes=1)
        res = ham_only_collection([ham_sig_vals,None],rho)
        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j*(hamiltonian.dot(rho) - rho.dot(hamiltonian))
        self.assertAllClose(res,expected)

        # Now, test adding a drift
        ham_drift_collection = DenseLindbladCollection(hamiltonian_operators,drift = drift,dissipator_operators = None)
        hamiltonian = np.tensordot(ham_sig_vals,hamiltonian_operators,axes=1) + drift
        res = ham_drift_collection([ham_sig_vals,None],rho)
        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j*(hamiltonian.dot(rho) - rho.dot(hamiltonian))
        self.assertAllClose(res,expected)

        # Now, test if the dissipator terms also work as intended
        (dissipator_operators @ rho @ dissipator_operators).shape

        # Now, test if the dissipator terms also work as intended
        full_lindblad_collection = DenseLindbladCollection(hamiltonian_operators,drift = drift,dissipator_operators = dissipator_operators)
        res = full_lindblad_collection([ham_sig_vals,dis_sig_vals],rho)
        hamiltonian = np.tensordot(ham_sig_vals,hamiltonian_operators,axes=1) + drift
        ham_terms = -1j*(hamiltonian.dot(rho) - rho.dot(hamiltonian))
        dis_anticommutator = (-1/2) * np.tensordot(dis_sig_vals, np.conjugate(np.transpose(dissipator_operators,[0,2,1])) @ dissipator_operators,axes=1)
        dis_anticommutator = dis_anticommutator.dot(rho) + rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(dis_sig_vals,dissipator_operators @ rho @ np.conjugate(np.transpose(dissipator_operators,[0,2,1])),axes=1)
        self.assertAllClose(ham_terms + dis_anticommutator + dis_extra,res)

        # Now, test if vectorization works as intended
        rhos = rand.uniform(-1,1,(l,n,n))
        res = full_lindblad_collection([ham_sig_vals,dis_sig_vals],rhos)
        for i in range(l):
            self.assertAllClose(res[i],full_lindblad_collection([ham_sig_vals,dis_sig_vals],rhos[i]))
    
    def test_apply_function(self):
        """Tests that the Lindblad collection is applying functions correctly"""
        rand.seed(824103)
        n = 16
        k = 8
        m = 4
        hamiltonian_operators = rand.uniform(-1,1,(k,n,n))
        dissipator_operators = rand.uniform(-1,1,(m,n,n))
        drift = rand.uniform(-1,1,(n,n))
        rho = rand.uniform(-1,1,(n,n))
        rho = rho + np.conjugate(rho.transpose())
        eval,evect = np.linalg.eigh(rho)
        ham_sig_vals = rand.uniform(-1,1,(k))
        dis_sig_vals = rand.uniform(-1,1,(m))
        collection = DenseLindbladCollection(hamiltonian_operators,drift=drift,dissipator_operators=dissipator_operators)
        f = lambda x: np.conjugate(np.transpose(evect)) @ x @ evect
        res = collection([ham_sig_vals,dis_sig_vals],rho)
        collection.apply_function_to_operators(f)
        newres = collection([ham_sig_vals,dis_sig_vals],f(rho))

        self.assertAllClose(f(res),newres)
        self.assertAllClose(f(res),newres)

class TestDenseLindbladCollectionJax(TestDenseOperatorCollection, TestJaxBase):
    """Jax version of TestGeneratorModel tests.

    Note: This class has no body but contains tests due to inheritance.
    """