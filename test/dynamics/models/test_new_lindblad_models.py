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
# pylint: disable=invalid-name,redundant-keyword-arg

"""Tests for qiskit_dynamics.models.lindblad_models.py, 
after migration to the OperatorCollection system. Most
of the actual calculation checking is handled at the level of a
models.operator_collection.DenseLindbladOperatorCollection test."""

import numpy as np
from numpy import random as rand
from scipy.linalg import expm
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.dispatch import Array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestLindbladModel(QiskitDynamicsTestCase):
    """Tests for LindbladModel."""

    def setUp(self):
        self.X = Array(Operator.from_label("X").data)
        self.Y = Array(Operator.from_label("Y").data)
        self.Z = Array(Operator.from_label("Z").data)

        # define a basic hamiltonian
        w = 2.0
        r = 0.5
        ham_operators = [2 * np.pi * self.Z / 2, 2 * np.pi * r * self.X / 2]
        ham_signals = [w, Signal(1.0, w)]

        self.w = w
        self.r = r

        dissipator_operators = Array([[[0.0, 0.0], [1.0, 0.0]]])

        self.basic_lindblad = LindbladModel(
            hamiltonian_operators=ham_operators,
            hamiltonian_signals=ham_signals,
            dissipator_operators=dissipator_operators,
        )

    def test_frame_transformations(self):
        np.random.seed(1230983)
        r = lambda *args: np.random.uniform(-1,1,np.squeeze(tuple(args)))
        e = lambda arr: expm(np.array(arr))
        rn = np.random.rand
        n = 16
        k = 8
        l = 4
        hframe = r(n,n)
        hframe = hframe + hframe.conj().transpose()

        eval,evect = np.linalg.eigh(hframe)
        f = lambda x: evect.T.conj() @ x @ evect

        ham_ops = r(k,n,n)
        ham_sig = SignalList([Signal(rn(),rn()) for j in range(k)])
        dis_ops = r(l,n,n)
        dis_sig = SignalList([Signal(rn(),rn()) for j in range(l)])

        rho = r(n,n)
        t = rn()

        m1 = LindbladModel(ham_ops,ham_sig,dis_ops,dis_sig,-hframe,None)
        m2 = LindbladModel(ham_ops,ham_sig,dis_ops,dis_sig,None,hframe)

        svals = [ham_sig.complex_value(t).real,dis_sig.complex_value(t).real]

        rho_in_frame = e(1j*t*hframe).dot(rho).dot(e(-1j*t*hframe))

        self.assertAllClose(f(m1._operator_collection.operators[0]),m2._operator_collection.operators[0])
        self.assertAllClose(f(m1._operator_collection.operators[1]),m2._operator_collection.operators[1])
        self.assertAllClose(f(m1._operator_collection.drift),m2._operator_collection.drift)
        self.assertAllClose(f(m1._operator_collection(svals,rho)),m2._operator_collection(svals,f(rho)))
        self.assertAllClose(np.outer(np.exp(1j * t * eval),np.exp(-1j*t*eval))*f(rho),f(rho_in_frame))
        self.assertAllClose(f(m1._operator_collection(svals,rho_in_frame)),m2._operator_collection(svals,f(rho_in_frame)))

        self.assertAllClose(e(-1j*t*hframe).dot(m1._operator_collection(svals,rho_in_frame)).dot(e(1j*t*hframe)),m2(t,rho,in_frame_basis=False))
        self.assertAllClose(e(-1j*t*hframe).dot(m1(t,rho_in_frame)).dot(e(1j*t*hframe)),m2(t,rho,in_frame_basis=False))

    def test_basic_lindblad_lmult(self):
        """Test lmult method of Lindblad generator OperatorModel."""
        A = Array([[1.0, 2.0], [3.0, 4.0]])

        t = 1.123
        ham = (
            2 * np.pi * self.w * self.Z.data / 2
            + 2 * np.pi * self.r * np.cos(2 * np.pi * self.w * t) * self.X.data / 2
        )
        sm = Array([[0.0, 0.0], [1.0, 0.0]])

        expected = self._evaluate_lindblad_rhs(A, ham, [sm])
        value = self.basic_lindblad(t,A)
        self.assertAllClose(expected, value)

    # pylint: disable=no-self-use,too-many-arguments
    def _evaluate_lindblad_rhs(
        self, A, ham, dissipators=None, dissipator_coeffs=None, frame_op=None, t=0.0
    ):
        """Evaluate the Lindblad equation

        frame_op assumed anti-Hermitian

        Note: here we force everything into numpy arrays as these parts of
        the test are just for confirmation
        """
        # if a frame operator is given, transform the model pieces into
        # the frame
        if frame_op is not None:
            frame_op = np.array(frame_op)
            U = expm(-t * frame_op)
            Uadj = U.conj().transpose()

            ham = U @ ham @ Uadj - 1j * frame_op

            if dissipators is not None:
                dissipators = np.array(dissipators)
                dissipators = [U @ D @ Uadj for D in dissipators]

        ham = np.array(ham)
        A = np.array(A)
        ham_part = -1j * (ham @ A - A @ ham)

        if dissipators is None:
            return ham_part

        dissipators = np.array(dissipators)

        if dissipator_coeffs is None:
            dissipator_coeffs = np.ones(len(dissipators))
        else:
            dissipator_coeffs = np.array(dissipator_coeffs)

        diss_part = np.zeros_like(A)
        for c, D in zip(dissipator_coeffs, dissipators):
            Dadj = D.conj().transpose()
            DadjD = Dadj @ D
            diss_part += c * (D @ A @ Dadj - 0.5 * (DadjD @ A + A @ DadjD))

        return ham_part + diss_part


        

class TestLindbladModelJax(TestLindbladModel, TestJaxBase):
    """Jax version of TestLindbladModel tests.

    Note: This class has no body but contains tests due to inheritance.
    """


def get_const_func(const):
    """Helper function for defining a constant function."""
    # pylint: disable=unused-argument
    def env(t):
        return const

    return env
