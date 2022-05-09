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

"""Tests for qiskit_dynamics.models.rotating_wave_approximation"""

import numpy as np
from scipy.sparse import issparse
from qiskit.quantum_info import Operator
from qiskit_dynamics.array import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.models import (
    GeneratorModel,
    HamiltonianModel,
    LindbladModel,
    RotatingFrame,
    rotating_wave_approximation,
)
from qiskit_dynamics.models.rotating_wave_approximation import get_rwa_operators
from qiskit_dynamics.type_utils import to_array
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestRotatingWaveApproximation(QiskitDynamicsTestCase):
    """Tests the rotating_wave_approximation function."""

    def setUp(self):
        self.v = 5.0
        self.r = 0.1
        static_operator = 2 * np.pi * self.v * Operator.from_label("Z") / 2
        operators = [2 * np.pi * self.r * Operator.from_label("X") / 2]
        signals = [Signal(1.0, carrier_freq=self.v)]

        self.classic_hamiltonian = HamiltonianModel(
            static_operator=static_operator,
            operators=operators,
            signals=signals,
            rotating_frame=np.diag(static_operator),
        )

    def test_generator_model_rwa_with_frame(self):
        """Test analytic RWA with a frame operator."""
        frame_op = np.array([[4j, 1j], [1j, 2j]])
        sigs = SignalList([Signal(1 + 1j * k, k / 3, np.pi / 2 * k) for k in range(1, 4)])
        self.assertAllClose(sigs(0), np.array([-1, -1, 3]))
        ops = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

        GM = GeneratorModel(
            static_operator=None, operators=ops, signals=sigs, rotating_frame=frame_op
        )
        self.assertAllClose(GM(0), np.array([[3 - 4j, -1], [-1 - 2j, -3 - 2j]]))
        self.assertAllClose(
            GM(1),
            [[-0.552558 - 4j, 3.18653 - 1.43887j], [3.18653 - 0.561126j, 0.552558 - 2j]],
            rtol=1e-5,
        )
        GM2 = rotating_wave_approximation(GM, 1 / 2 / np.pi)
        self.assertAllClose(GM2.signals(0), [-1, -1, 3, 1, -2, -1])
        self.assertAllClose(
            GM2(0), [[0.25 - 4.0j, -0.25 - 0.646447j], [-0.25 - 1.35355j, -0.25 - 2.0j]], rtol=1e-5
        )
        self.assertAllClose(
            GM2(1),
            [
                [0.0181527 - 4.0j, -0.0181527 - 0.500659j],
                [-0.0181527 - 1.49934j, -0.0181527 - 2.0j],
            ],
            rtol=1e-5,
        )

    def test_generator_model_rwa_with_frame_in_frame_basis(self):
        """Test analytic RWA with a frame operator with model in frame basis."""
        frame_op = np.array([[4j, 1j], [1j, 2j]])
        sigs = SignalList([Signal(1 + 1j * k, k / 3, np.pi / 2 * k) for k in range(1, 4)])
        self.assertAllClose(sigs(0), np.array([-1, -1, 3]))
        ops = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

        GM = GeneratorModel(
            static_operator=None,
            operators=ops,
            signals=sigs,
            rotating_frame=frame_op,
            in_frame_basis=True,
        )
        rotating_frame = GM.rotating_frame
        self.assertAllClose(
            GM(0),
            rotating_frame.operator_into_frame_basis(np.array([[3 - 4j, -1], [-1 - 2j, -3 - 2j]])),
        )
        self.assertAllClose(
            GM(1),
            rotating_frame.operator_into_frame_basis(
                np.array(
                    [[-0.552558 - 4j, 3.18653 - 1.43887j], [3.18653 - 0.561126j, 0.552558 - 2j]]
                )
            ),
            rtol=1e-5,
        )
        GM2 = rotating_wave_approximation(GM, 1 / 2 / np.pi)
        self.assertAllClose(GM2.signals(0), [-1, -1, 3, 1, -2, -1])
        self.assertAllClose(
            GM2(0),
            rotating_frame.operator_into_frame_basis(
                np.array([[0.25 - 4.0j, -0.25 - 0.646447j], [-0.25 - 1.35355j, -0.25 - 2.0j]])
            ),
            rtol=1e-5,
        )
        self.assertAllClose(
            GM2(1),
            rotating_frame.operator_into_frame_basis(
                np.array(
                    [
                        [0.0181527 - 4.0j, -0.0181527 - 0.500659j],
                        [-0.0181527 - 1.49934j, -0.0181527 - 2.0j],
                    ]
                )
            ),
            rtol=1e-5,
        )
        self.assertTrue(GM2.in_frame_basis)

    def test_generator_model_no_rotating_frame(self):
        """Tests whether RWA works in the absence of a rotating frame"""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
        dft = Array(np.ones((2, 2)))
        GM = GeneratorModel(static_operator=dft, operators=ops, signals=sigs)
        GMP = rotating_wave_approximation(GM, 2)
        self.assertAllClose(GMP._get_static_operator(True), Array(np.ones((2, 2))))
        post_rwa_ops = Array(np.array([1, 0, 1, 0, 0, 0, 0, 0]).reshape((8, 1, 1))) * Array(
            np.ones((8, 2, 2))
        )
        self.assertAllClose(GMP._get_operators(True), post_rwa_ops)

    def test_generator_model_no_rotating_frame_no_static_operator(self):
        """Test case of no frame and no static_operator."""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]

        # test without static_operator
        GM = GeneratorModel(static_operator=None, operators=ops, signals=sigs)
        GMP = rotating_wave_approximation(GM, 2)
        self.assertTrue(GMP.static_operator is None)
        post_rwa_ops = Array(np.array([1, 0, 1, 0, 0, 0, 0, 0]).reshape((8, 1, 1))) * Array(
            np.ones((8, 2, 2))
        )
        self.assertAllClose(GMP._get_operators(True), post_rwa_ops)

    def test_generator_model_rotating_frame_no_operators(self):
        """Test case for GeneratorModel with rotating frame and no operators."""
        frame_op = 2 * np.pi * np.array([1, -1]) / 2

        GM = GeneratorModel(
            static_operator=np.array([[3.0, 1], [1, 2.0]]) - 1j * np.diag(frame_op),
            operators=None,
            signals=None,
            rotating_frame=frame_op,
        )
        GM2 = rotating_wave_approximation(GM, 1.0)
        self.assertAllClose(GM2(0), np.array([[3.0, 0], [0, 2.0]]))

    def test_lindblad_model_rotating_frame_only_static_hamiltonian(self):
        """Test case for LindbladModel with just a static hamiltonian."""
        frame_op = 2 * np.pi * np.array([1, -1]) / 2

        model = LindbladModel(
            static_hamiltonian=np.array([[3.0, 1], [1, 2.0]]) + np.diag(frame_op),
            rotating_frame=frame_op,
        )
        rwa_model = rotating_wave_approximation(model, 1.0)
        ham = np.array([[3.0, 0], [0, 2.0]])
        rho = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = -1j * (ham @ rho - rho @ ham)
        self.assertAllClose(rwa_model(0, rho), expected)

    def test_lindblad_model_rotating_frame_only_static_hamiltonian_in_frame_basis(self):
        """Test case for LindbladModel with just a static hamiltonian
        in frame basis.
        """

        U = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2)
        Uadj = U.conj().transpose()
        frame_op = 2 * np.pi * U @ np.diag([1, -1]) @ Uadj / 2

        model = LindbladModel(
            static_hamiltonian=U @ np.array([[3.0, 1], [1, 2.0]]) @ Uadj + frame_op,
            rotating_frame=frame_op,
            in_frame_basis=True,
        )
        rwa_model = rotating_wave_approximation(model, 0.99)
        # flipped due to eigenvalue ordering
        ham = np.array([[2.0, 0], [0, 3.0]])
        rho = np.array([[1.0, 2.0], [3.0, 4.0]])
        expected = -1j * (ham @ rho - rho @ ham)
        self.assertTrue(rwa_model.in_frame_basis)
        self.assertAllClose(rwa_model(0, rho), expected)

    def test_classic_hamiltonian_model(self):
        """Test classic analytic case for HamiltonianModel."""

        rwa_ham_model = rotating_wave_approximation(self.classic_hamiltonian, 2 * self.v)

        self.assertAllClose(rwa_ham_model.static_operator, np.zeros((2, 2)))
        expected_ops = (
            2 * np.pi * self.r * np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1j], [1j, 0.0]]]) / 4
        )
        self.assertAllClose(rwa_ham_model.operators, expected_ops)

    def test_static_dissipator_vs_non_static(self):
        """Compare evaluation of static dissipators with non-static."""

        np.random.seed(2314)
        random_mats = lambda *args: Array(np.random.uniform(-1, 1, args))
        random_complex_mats = lambda *args: random_mats(*args) + 1j * random_mats(*args)

        random_diss = random_complex_mats(3, 2, 2)

        lindblad_model1 = LindbladModel.from_hamiltonian(
            hamiltonian=self.classic_hamiltonian,
            dissipator_operators=random_diss,
            dissipator_signals=[1.0] * 3,
        )
        lindblad_model2 = LindbladModel.from_hamiltonian(
            hamiltonian=self.classic_hamiltonian, static_dissipators=random_diss
        )

        rwa_lindblad_model1 = rotating_wave_approximation(lindblad_model1, cutoff_freq=self.v)
        rwa_lindblad_model2 = rotating_wave_approximation(lindblad_model2, cutoff_freq=self.v)

        t = 0.23124
        A = random_complex_mats(2, 2)
        out1 = rwa_lindblad_model1(t, A)
        out2 = rwa_lindblad_model2(t, A)

        self.assertAllClose(out1, out2)

    def test_signal_translator_generator_model(self):
        """Tests signal translation from pre-RWA to post-RWA through
        rotating_wave_approximation.get_rwa_signals when passed a
        GeneratorModel."""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
        dft = Array(np.ones((2, 2)))
        GM = GeneratorModel(operators=ops, signals=sigs, static_operator=dft, rotating_frame=None)
        f = rotating_wave_approximation(GM, 100, return_signal_map=True)[1]
        vals = f(sigs).complex_value(3)
        self.assertAllClose(vals[:4], GM.signals.complex_value(3))
        s_prime = [
            Signal(1, 0, -np.pi / 2),
            Signal(1, -3, -np.pi / 2),
            Signal(1, 1, -np.pi / 2),
            Signal(1, 3, -np.pi / 2),
        ]
        self.assertAllClose(vals[4:], SignalList(s_prime).complex_value(3))
        self.assertTrue(f(None) is None)

    def test_signal_translator_lindblad_model(self):
        """Like test_signal_translator_generator_model, but for LindbladModels."""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
        s_prime = [
            Signal(1, 0, -np.pi / 2),
            Signal(1, -3, -np.pi / 2),
            Signal(1, 1, -np.pi / 2),
            Signal(1, 3, -np.pi / 2),
        ]
        dft = Array(np.ones((2, 2)))
        LM = LindbladModel(
            static_hamiltonian=dft,
            hamiltonian_operators=ops,
            hamiltonian_signals=sigs,
            dissipator_operators=ops,
            dissipator_signals=sigs,
        )
        f = rotating_wave_approximation(LM, 100, return_signal_map=True)[1]
        rwa_ham_sig, rwa_dis_sig = f((sigs, sigs))
        self.assertAllClose(rwa_ham_sig.complex_value(2)[:4], SignalList(sigs).complex_value(2))
        self.assertAllClose(rwa_dis_sig.complex_value(2)[:4], SignalList(sigs).complex_value(2))
        self.assertAllClose(rwa_ham_sig.complex_value(2)[4:], SignalList(s_prime).complex_value(2))
        self.assertAllClose(rwa_dis_sig.complex_value(2)[4:], SignalList(s_prime).complex_value(2))

        self.assertTrue(f((None, None)) == (None, None))

    def test_rwa_operators(self):
        """Tests get_rwa_operators using pseudorandom numbers."""
        np.random.seed(123098123)
        r = lambda *args: Array(np.random.uniform(-1, 1, args))
        rj = lambda *args: r(*args) + 1j * r(*args)
        ops = rj(4, 3, 3)
        carrier_freqs = r(4)
        cutoff_freq = 0.3
        sigs = SignalList([Signal(r(), freq, r()) for freq in carrier_freqs])

        frame_op = r(3, 3)
        frame_op = frame_op - frame_op.conj().T
        rotating_frame = RotatingFrame(frame_op)

        ops_in_fb = rotating_frame.operator_into_frame_basis(ops)
        diag = rotating_frame.frame_diag
        diff_matrix = np.broadcast_to(diag, (3, 3)) - np.broadcast_to(diag, (3, 3)).T
        frame_freqs = diff_matrix.imag / (2 * np.pi)

        rwa_ops = get_rwa_operators(ops_in_fb, sigs, rotating_frame, frame_freqs, cutoff_freq)

        carrier_freqs = carrier_freqs.reshape(4, 1, 1)
        frame_freqs = frame_freqs.reshape(1, 3, 3)
        G_p = ops_in_fb * (np.abs(carrier_freqs + frame_freqs) < cutoff_freq).astype(int)
        G_m = ops_in_fb * (np.abs(-carrier_freqs + frame_freqs) < cutoff_freq).astype(int)
        self.assertAllClose(
            rwa_ops[:4], rotating_frame.operator_out_of_frame_basis((G_p + G_m) / 2)
        )
        self.assertAllClose(
            rwa_ops[4:], rotating_frame.operator_out_of_frame_basis(1j * (G_p - G_m) / 2)
        )


class TestRotatingWaveApproximationJax(TestRotatingWaveApproximation, TestJaxBase):
    """Jax version of TestRotatingWaveApproximation tests."""

    def test_jitable_gradable_signal_map(self):
        """Tests that signal_map from the RWA is jitable and gradable."""

        sample_sigs = [Signal(1.0, 0.0), Signal(1.0, -3.0), Signal(1.0, 1.0), Signal(1.0, 3.0)]
        ops = Array(np.ones((4, 2, 2)))
        static_operator = Array(np.ones((2, 2)))
        model = GeneratorModel(
            operators=ops,
            signals=sample_sigs,
            static_operator=static_operator,
            rotating_frame=static_operator,
        )
        rwa_model, signal_map = rotating_wave_approximation(
            model=model, cutoff_freq=2.0, return_signal_map=True
        )

        def _simple_function_using_rwa(t, w):
            """Simple function that involves taking the rotating wave approximation."""
            sigs = [Signal(1, 0), Signal(lambda s: w * s, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
            rwa_model_copy = rwa_model.copy()
            rwa_model_copy.signals = signal_map(sigs)
            return rwa_model(t)

        self.jit_wrap(_simple_function_using_rwa)(1.0, 1.0)
        self.jit_grad_wrap(_simple_function_using_rwa)(1.0, 1.0)


class TestRotatingWaveApproximationSparse(QiskitDynamicsTestCase):
    """Tests the rotating_wave_approximation function for sparse models."""

    def setUp(self):
        self.v = 5.0
        self.r = 0.1
        static_operator = 2 * np.pi * self.v * Operator.from_label("Z") / 2
        operators = [2 * np.pi * self.r * Operator.from_label("X") / 2]
        signals = [Signal(1.0, carrier_freq=self.v)]

        self.classic_hamiltonian = HamiltonianModel(
            static_operator=static_operator,
            operators=operators,
            signals=signals,
            rotating_frame=np.diag(static_operator),
            evaluation_mode="sparse",
        )

    def test_classic_hamiltonian_model_sparse(self):
        """Test classic analytic case for HamiltonianModel with sparse operators."""

        rwa_ham_model = rotating_wave_approximation(self.classic_hamiltonian, 2 * self.v)

        self.assertSparseEquality(rwa_ham_model.static_operator, np.zeros((2, 2)))
        expected_ops = (
            2 * np.pi * self.r * np.array([[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1j], [1j, 0.0]]]) / 4
        )
        self.assertSparseEquality(rwa_ham_model.operators, expected_ops)

    def assertSparseEquality(self, op, expected):
        """Validate that op is sparse and is equal to expected."""
        if isinstance(op, list):
            for sub_op in op:
                self.assertTrue(issparse(sub_op))
        else:
            self.assertTrue(issparse(op))

        self.assertAllClose(to_array(op), to_array(expected))


class TestRotatingWaveApproximationSparseJax(TestRotatingWaveApproximationSparse, TestJaxBase):
    """JAX version of TestRotatingWaveApproximationSparse."""

    def assertSparseEquality(self, op, expected):
        """Validate that op is sparse and is equal to expected."""
        self.assertTrue(type(op).__name__ == "BCOO")
        self.assertAllClose(to_array(op), to_array(expected))
