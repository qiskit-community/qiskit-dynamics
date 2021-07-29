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
        rand.seed(2401923)
        n = 16
        m = 8
        k = 4
        hframe = rand.uniform(-1,1,(n,n))
        eval,evect = np.linalg.eigh(hframe)
        f = lambda x: np.conjugate(np.transpose(evect)) @ x @ evect
        ham_ops = rand.uniform(-1,1,(m,n,n))
        dis_ops = rand.uniform(-1,1,(k,n,n))
        rho = rand.uniform(-1,1,(n,n))
        ham_sig = SignalList([Signal(rand.rand(),rand.rand()) for j in range(m)])
        dis_sig = SignalList([Signal(rand.rand(),rand.rand()) for j in range(k)])
        # the hermitian H s.t. F = -iH and the frame transformation is \rho -> e^{iHt}\rho e^{-iHt}
        hframe = hframe + hframe.conj().transpose()
        t = 1
        m1 = LindbladModel(ham_ops,hamiltonian_signals=ham_sig,dissipator_operators=dis_ops,dissipator_signals=dis_sig,drift = -hframe)
        r1 = expm(t*np.array(1j*hframe)).dot(m1(t,rho)).dot(expm(-t*np.array(1j*hframe)))
        m2 = LindbladModel(ham_ops,hamiltonian_signals=ham_sig,dissipator_signals=dis_sig,dissipator_operators=dis_ops,frame=hframe)
        r2 = m2(t,f(expm(t*np.array(1j*hframe)).dot(rho).dot(expm(-t*np.array(1j*hframe)))),in_frame_basis=False)
        self.assertAllClose(r1,r2)

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


        

# class TestLindbladModelJax(TestLindbladModel, TestJaxBase):
#     """Jax version of TestLindbladModel tests.

#     Note: This class has no body but contains tests due to inheritance.
#     """


def get_const_func(const):
    """Helper function for defining a constant function."""
    # pylint: disable=unused-argument
    def env(t):
        return const

    return env
