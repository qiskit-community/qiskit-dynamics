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

"""tests for qiskit_dynamics.models.rotating_wave"""

import numpy as np
from qiskit_dynamics.dispatch.array import Array
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.models import GeneratorModel, rotating_wave_approximation
from ..common import QiskitDynamicsTestCase, TestJaxBase


class TestRotatingWave(QiskitDynamicsTestCase):
    """Tests the rotating_wave_approximation function."""

    def setUp(self):
        pass

    def test_generator_model_rwa_with_frame(self):
        """Tests whether RWA functionality mapping GeneratorModels -> GeneratorModels
        is correct by comparing it to an analytically calculable case."""
        frame_op = np.array([[4j, 1j], [1j, 2j]])
        sigs = SignalList([Signal(1 + 1j * k, k / 3, np.pi / 2 * k) for k in range(1, 4)])
        self.assertAllClose(sigs(0), np.array([-1, -1, 3]))
        ops = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]])

        GM = GeneratorModel(ops, drift=None, signals=sigs, rotating_frame=frame_op)
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

    def test_no_rotating_frame(self):
        """Tests whether RWA works in the absence of a rotating frame"""
        ops = Array(np.ones((4, 2, 2)))
        sigs = [Signal(1, 0), Signal(1, -3, 0), Signal(1, 1), Signal(1, 3, 0)]
        dft = Array(np.ones((2, 2)))
        GM = GeneratorModel(ops, signals=sigs, drift=dft, rotating_frame=None)
        GMP = rotating_wave_approximation(GM, 2)
        self.assertAllClose(GMP.get_drift(True), Array(np.ones((2, 2))))
        post_rwa_ops = Array(np.array([1, 0, 1, 0]).reshape((4, 1, 1))) * ops
        self.assertAllClose(GMP.get_operators(True), post_rwa_ops)


class TestRotatingWaveJax(TestRotatingWave, TestJaxBase):
    """Jax version of TestRotatingWave tests.

    Note: This class has no body but contains tests due to inheritance.
    """
