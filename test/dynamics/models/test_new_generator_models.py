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

"""Tests for generator_models.py after
beginning to use OperatorCollections. """

from qiskit_dynamics.signals.signals import SignalList
import numpy as np
import numpy.random as rand
from scipy.linalg import expm
from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import GeneratorModel,Frame
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

        # define a basic model
        w = 2.0
        r = 0.5
        operators = [-1j * 2 * np.pi * self.Z / 2, -1j * 2 * np.pi * r * self.X / 2]
        signals = [w, Signal(1.0, w)]

        self.w = 2
        self.r = r
        self.basic_model = GeneratorModel(operators=operators, signals=signals)
        
    def test_known_values_basic_functionality(self):
        """Test for checking that with known operators that
        the Model returns the analyticlly known values."""

        test_operator_list = Array([self.X,self.Y,self.Z])
        signals = SignalList([Signal(1,j/3) for j in range(3)])
        simple_model = GeneratorModel(test_operator_list,drift=None,signals=signals,frame=None)

        res = simple_model.evaluate_without_state(2)
        self.assertAllClose(res,Array([[-0.5+0j, 1. + 0.5j], [1. - 0.5j, 0.5+0j]]))

        simple_model._operator_collection.drift = np.eye(2)
        res = simple_model.evaluate_without_state(2)
        self.assertAllClose(res,Array([[0.5+0j, 1. + 0.5j], [1. - 0.5j, 1.5+0j]]))
        simple_model._operator_collection.drift = None

    def test_frame_basis_transformation(self):
        """Test for checking that the frame basis transformations,
        pre- and post-rotation routines, as well as taking operators
        out of the frame basis are producing the analytically known
        answers. """
        test_operator_list = Array([self.X,self.Y,self.Z])
        signals = SignalList([Signal(1,j/3) for j in range(3)])
        simple_model = GeneratorModel(test_operator_list,drift=None,signals=signals,frame=None)

        simple_model._operator_collection.drift = np.eye(2)
        fop = Array([[0,1j],[1j,0]])
        simple_model.frame = fop
        res = simple_model(2,in_frame_basis=False)
        expected = expm(np.array(-2*fop)) @ (Array([[0.5+0j, 1. + 0.5j], [1. - 0.5j, 1.5+0j]]) - fop) @ expm(np.array(2*fop))
        self.assertAllClose(res,expected)

        res = simple_model(2,in_frame_basis=True)
        expected = np.array([[1,1],[1,-1]])/np.sqrt(2) @ expected @ np.array([[1,1],[1,-1]])/np.sqrt(2)

        self.assertAllClose(res,expected)

        simple_model._operator_collection.drift = None

    def test_order_of_application_cases(self):
        """Test to see if the (nontrivial) setter methods
        of GeneratorModel are (a) working with all possible 
        working types of input, (b) whether adding properties
        as part of the constructor phase or afterwards makes
        a difference, and (c) if added after the constructor
        phase, if the order in which they're set matters."""

        paulis = Array([
            [[0,1],[1,0]],
            [[0,-1j],[1j,0]],
            [[1,0],[0,-1]]])
        extra = Array(np.eye(2))
        state = Array([0.2,0.5])

        t = 2

        sarr = [Signal(1,j/3) for j in range(3)]
        sigvals = np.real(SignalList(sarr).complex_value(t))

        farr = Array(np.array([[3j,2j],[2j,0]]))
        farr2 = Array(np.array([[1j,2],[-2,3j]]))
        evals,evect = np.linalg.eig(farr)
        diafarr = np.diag(evals)

        paulis_in_frame_basis = np.conjugate(np.transpose(evect)) @ paulis @ evect


        ## Run checks without frame for now
        gm1 = GeneratorModel(paulis,extra,sarr)
        gm2 = GeneratorModel(paulis,extra)
        gm2.signals = sarr
        gm3 = GeneratorModel(paulis,extra)
        gm3.signals = SignalList(sarr)

        # All should be the same, because there are no frames involved
        t11 = gm1.evaluate_without_state(t,False)
        t12 = gm1.evaluate_without_state(t,True)
        t21 = gm2.evaluate_without_state(t,False)
        t22 = gm2.evaluate_without_state(t,True)
        t31 = gm3.evaluate_without_state(t,False)
        t32 = gm3.evaluate_without_state(t,True)
        t_analytical = Array([[0.5, 1. + 0.5j], [1. - 0.5j, 1.5]])

        self.assertAllClose(t11,t12)
        self.assertAllClose(t11,t21)
        self.assertAllClose(t11,t22)
        self.assertAllClose(t11,t31)
        self.assertAllClose(t11,t32)
        self.assertAllClose(t11,t_analytical)

        # now work with a specific statevector
        ts1 = gm1.evaluate_with_state(t,state,in_frame_basis=False)
        ts2 = gm1.evaluate_with_state(t,state,in_frame_basis=True)
        ts_analytical = Array([0.6+0.25j,0.95-0.1j])

        self.assertAllClose(ts1,ts2)
        self.assertAllClose(ts1,ts_analytical)

        ## Now, run checks with frame
        # If passing a frame in the first place, operators must be in frame basis.abs
        # Testing at the same time whether having Drift = None is an issue. 
        gm1 = GeneratorModel(paulis_in_frame_basis, signals = sarr, frame = Frame(farr))
        gm2 = GeneratorModel(paulis_in_frame_basis, frame = farr)
        gm2.signals = SignalList(sarr)
        gm3 = GeneratorModel(paulis_in_frame_basis, frame = farr)
        gm3.signals = sarr
        # Does adding a frame after make a difference?
        # If so, does it make a difference if we add signals or the frame first?
        gm4 = GeneratorModel(paulis)
        gm4.signals = sarr
        gm4.frame = farr
        gm5 = GeneratorModel(paulis)
        gm5.frame = farr
        gm5.signals = sarr
        gm6 = GeneratorModel(paulis,signals = sarr)
        gm6.frame = farr
        # If we go to one frame, then transform back, does this make a difference?
        gm7 = GeneratorModel(paulis,signals=sarr)
        gm7.frame = farr2
        gm7.frame = farr

        t_in_frame_actual = Array(np.diag(np.exp(-t*evals)) @ (np.tensordot(sigvals,paulis_in_frame_basis,axes=1) - diafarr) @ np.diag(np.exp(t*evals)))
        tf1 = gm1.evaluate_without_state(t,in_frame_basis=True)
        tf2 = gm2.evaluate_without_state(t,in_frame_basis=True)
        tf3 = gm3.evaluate_without_state(t,in_frame_basis=True)
        tf4 = gm4.evaluate_without_state(t,in_frame_basis=True)
        tf5 = gm5.evaluate_without_state(t,in_frame_basis=True)
        tf6 = gm6.evaluate_without_state(t,in_frame_basis=True)
        tf7 = gm7.evaluate_without_state(t,in_frame_basis=True)

        self.assertAllClose(t_in_frame_actual,tf1)
        self.assertAllClose(tf1,tf2)
        self.assertAllClose(tf1,tf3)
        self.assertAllClose(tf1,tf4)
        self.assertAllClose(tf1,tf5)
        self.assertAllClose(tf1,tf6)
        self.assertAllClose(tf1,tf7)
    
    def test_vectorization_pseudorandom(self):
        """Test for whether evaluating a model at m different
        states, each with an n-length statevector, by passing
        an (m,n) Array provides the same value as passing each
        (n) Array individually."""
        rand.seed(9231)
        n = 8
        k = 4
        m = 2
        t = rand.rand()
        sig_list = SignalList([Signal(rand.rand(),rand.rand()) for j in range(k)])
        normal_states = rand.uniform(-1,1,(n))
        vectorized_states = rand.uniform(-1,1,(m,n))

        operators = rand.uniform(-1,1,(k,n,n))

        gm = GeneratorModel(operators,drift=None,signals=sig_list)
        self.assertTrue(gm.evaluate_with_state(t,normal_states).shape==(n,))
        self.assertTrue(gm.evaluate_with_state(t,vectorized_states).shape==(m,n))
        for i in range(m):
            self.assertAllClose(gm.evaluate_with_state(t,vectorized_states)[i],gm.evaluate_with_state(t,vectorized_states[i]))

        farr = rand.uniform(-1,1,(n,n))
        farr = farr - np.conjugate(np.transpose(farr))

        gm.frame = farr

        self.assertTrue(gm.evaluate_with_state(t,normal_states).shape==(n,))
        self.assertTrue(gm.evaluate_with_state(t,vectorized_states).shape==(m,n))
        for i in range(m):
            self.assertAllClose(gm.evaluate_with_state(t,vectorized_states)[i],gm.evaluate_with_state(t,vectorized_states[i]))

        vectorized_result = gm.evaluate_with_state(t,vectorized_states,in_frame_basis=True)

        self.assertTrue(gm.evaluate_with_state(t,normal_states,in_frame_basis=True).shape==(n,))
        self.assertTrue(vectorized_result.shape==(m,n))
        for i in range(m):
            self.assertAllClose(vectorized_result[i],gm.evaluate_with_state(t,vectorized_states[i],in_frame_basis=True))


class TestDenseOperatorCollectionJax(TestDenseOperatorCollection, TestJaxBase):
    """Jax version of TestGeneratorModel tests.

    Note: This class has no body but contains tests due to inheritance.
    """


# class TestDenseLindbladCollection(QiskitDynamicsTestCase):
#     """Tests for GeneratorModel."""

#     def setUp(self):
#         self.X = Array(Operator.from_label("X").data)
#         self.Y = Array(Operator.from_label("Y").data)
#         self.Z = Array(Operator.from_label("Z").data)


# class TestDenseLindbladCollectionJax(TestDenseOperatorCollection, TestJaxBase):
#     """Jax version of TestGeneratorModel tests.

#     Note: This class has no body but contains tests due to inheritance.
#     """