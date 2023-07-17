# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Tests for signals.
"""

import numpy as np

from qiskit_dynamics.signals import Signal, DiscreteSignal, DiscreteSignalSum, SignalList
from qiskit_dynamics.signals.signals import to_SignalSum
from qiskit_dynamics.array import Array

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit, grad
    import jax.numpy as jnp
except ImportError:
    pass


class TestSignal(QiskitDynamicsTestCase):
    """Tests for Signal object."""

    def setUp(self):
        self.signal1 = Signal(lambda t: 0.25, carrier_freq=0.3)
        self.signal2 = Signal(lambda t: 2.0 * (t**2), carrier_freq=0.1)
        self.signal3 = Signal(lambda t: 2.0 * (t**2) + 1j * t, carrier_freq=0.1, phase=-0.1)

    def test_envelope(self):
        """Test envelope evaluation."""
        self.assertAllClose(self.signal1.envelope(0.0), 0.25)
        self.assertAllClose(self.signal1.envelope(1.23), 0.25)

        self.assertAllClose(self.signal2.envelope(1.1), 2 * (1.1**2))
        self.assertAllClose(self.signal2.envelope(1.23), 2 * (1.23**2))

        self.assertAllClose(self.signal3.envelope(1.1), 2 * (1.1**2) + 1j * 1.1)
        self.assertAllClose(self.signal3.envelope(1.23), 2 * (1.23**2) + 1j * 1.23)

    def test_envelope_vectorized(self):
        """Test vectorized evaluation of envelope."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.signal1.envelope(t_vals), np.array([0.25, 0.25]))
        self.assertAllClose(
            self.signal2.envelope(t_vals), np.array([2 * (1.1**2), 2 * (1.23**2)])
        )
        self.assertAllClose(
            self.signal3.envelope(t_vals),
            np.array([2 * (1.1**2) + 1j * 1.1, 2 * (1.23**2) + 1j * 1.23]),
        )

        t_vals = np.array([[1.1, 1.23], [0.1, 0.24]])
        self.assertAllClose(self.signal1.envelope(t_vals), np.array([[0.25, 0.25], [0.25, 0.25]]))
        self.assertAllClose(
            self.signal2.envelope(t_vals),
            np.array([[2 * (1.1**2), 2 * (1.23**2)], [2 * (0.1**2), 2 * (0.24**2)]]),
        )
        self.assertAllClose(
            self.signal3.envelope(t_vals),
            np.array(
                [
                    [2 * (1.1**2) + 1j * 1.1, 2 * (1.23**2) + 1j * 1.23],
                    [2 * (0.1**2) + 1j * 0.1, 2 * (0.24**2) + 1j * 0.24],
                ]
            ),
        )

    def test_complex_value(self):
        """Test complex_value evaluation."""
        self.assertAllClose(
            self.signal1.complex_value(0.0), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0)
        )
        self.assertAllClose(
            self.signal1.complex_value(1.23), 0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23)
        )

        self.assertAllClose(
            self.signal2.complex_value(1.1), 2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1)
        )
        self.assertAllClose(
            self.signal2.complex_value(1.23), 2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23)
        )

        self.assertAllClose(
            self.signal3.complex_value(1.1),
            (2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)),
        )
        self.assertAllClose(
            self.signal3.complex_value(1.23),
            (2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1)),
        )

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(
            self.signal1.complex_value(t_vals),
            np.array(
                [
                    0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1),
                    0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23),
                ]
            ),
        )
        self.assertAllClose(
            self.signal2.complex_value(t_vals),
            np.array(
                [
                    2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1),
                    2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23),
                ]
            ),
        )
        self.assertAllClose(
            self.signal3.complex_value(t_vals),
            np.array(
                [
                    (2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)),
                    (2 * (1.23**2) + 1j * 1.23)
                    * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1)),
                ]
            ),
        )

        t_vals = np.array([[1.1, 1.23], [0.1, 0.24]])
        self.assertAllClose(
            self.signal1.complex_value(t_vals),
            np.array(
                [
                    [
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1),
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23),
                    ],
                    [
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.1),
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.24),
                    ],
                ]
            ),
        )
        self.assertAllClose(
            self.signal2.complex_value(t_vals),
            np.array(
                [
                    [
                        2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1),
                        2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23),
                    ],
                    [
                        2 * (0.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.1),
                        2 * (0.24**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.24),
                    ],
                ]
            ),
        )
        self.assertAllClose(
            self.signal3.complex_value(t_vals),
            np.array(
                [
                    [
                        (2 * (1.1**2) + 1j * 1.1)
                        * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)),
                        (2 * (1.23**2) + 1j * 1.23)
                        * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1)),
                    ],
                    [
                        (2 * (0.1**2) + 1j * 0.1)
                        * np.exp(1j * 2 * np.pi * 0.1 * 0.1 + 1j * (-0.1)),
                        (2 * (0.24**2) + 1j * 0.24)
                        * np.exp(1j * 2 * np.pi * 0.1 * 0.24 + 1j * (-0.1)),
                    ],
                ]
            ),
        )

    def test_call(self):
        """Test __call__."""
        self.assertAllClose(self.signal1(0.0), np.real(0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0)))
        self.assertAllClose(self.signal1(1.23), np.real(0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23)))

        self.assertAllClose(
            self.signal2(1.1), np.real(2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1))
        )
        self.assertAllClose(
            self.signal2(1.23), np.real(2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23))
        )

        self.assertAllClose(
            self.signal3(1.1),
            np.real((2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1))),
        )
        self.assertAllClose(
            self.signal3(1.23),
            np.real(
                (2 * (1.23**2) + 1j * 1.23) * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1))
            ),
        )

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(
            self.signal1(t_vals),
            np.array(
                [
                    0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1),
                    0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23),
                ]
            ).real,
        )
        self.assertAllClose(
            self.signal2(t_vals),
            np.array(
                [
                    2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1),
                    2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23),
                ]
            ).real,
        )
        self.assertAllClose(
            self.signal3(t_vals),
            np.array(
                [
                    (2 * (1.1**2) + 1j * 1.1) * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)),
                    (2 * (1.23**2) + 1j * 1.23)
                    * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1)),
                ]
            ).real,
        )

        t_vals = np.array([[1.1, 1.23], [0.1, 0.24]])
        self.assertAllClose(
            self.signal1(t_vals),
            np.array(
                [
                    [
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.1),
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 1.23),
                    ],
                    [
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.1),
                        0.25 * np.exp(1j * 2 * np.pi * 0.3 * 0.24),
                    ],
                ]
            ).real,
        )
        self.assertAllClose(
            self.signal2(t_vals),
            np.array(
                [
                    [
                        2 * (1.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.1),
                        2 * (1.23**2) * np.exp(1j * 2 * np.pi * 0.1 * 1.23),
                    ],
                    [
                        2 * (0.1**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.1),
                        2 * (0.24**2) * np.exp(1j * 2 * np.pi * 0.1 * 0.24),
                    ],
                ]
            ).real,
        )
        self.assertAllClose(
            self.signal3(t_vals),
            np.array(
                [
                    [
                        (2 * (1.1**2) + 1j * 1.1)
                        * np.exp(1j * 2 * np.pi * 0.1 * 1.1 + 1j * (-0.1)),
                        (2 * (1.23**2) + 1j * 1.23)
                        * np.exp(1j * 2 * np.pi * 0.1 * 1.23 + 1j * (-0.1)),
                    ],
                    [
                        (2 * (0.1**2) + 1j * 0.1)
                        * np.exp(1j * 2 * np.pi * 0.1 * 0.1 + 1j * (-0.1)),
                        (2 * (0.24**2) + 1j * 0.24)
                        * np.exp(1j * 2 * np.pi * 0.1 * 0.24 + 1j * (-0.1)),
                    ],
                ]
            ).real,
        )

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""

        sig3_conj = self.signal3.conjugate()

        self.assertAllClose(self.signal3.phase, -sig3_conj.phase)
        self.assertAllClose(self.signal3.carrier_freq, -sig3_conj.carrier_freq)
        self.assertAllClose(
            self.signal3.complex_value(1.231), np.conjugate(sig3_conj.complex_value(1.231))
        )
        self.assertAllClose(
            self.signal3.complex_value(1.231 * np.pi),
            np.conjugate(sig3_conj.complex_value(1.231 * np.pi)),
        )


class TestConstant(QiskitDynamicsTestCase):
    """Tests for constant signal object."""

    def setUp(self):
        self.constant1 = Signal(1.0)
        self.constant2 = Signal(3.0 + 1j * 2)

    def test_envelope(self):
        """Test envelope evaluation."""
        self.assertAllClose(self.constant1.envelope(0.0), 1.0)
        self.assertAllClose(self.constant1.envelope(1.23), 1.0)

        self.assertAllClose(self.constant2.envelope(1.1), (3.0 + 2j))
        self.assertAllClose(self.constant2.envelope(1.23), (3.0 + 2j))

    def test_envelope_vectorized(self):
        """Test vectorized evaluation of envelope."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.constant1.envelope(t_vals), np.array([1.0, 1.0]))
        self.assertAllClose(self.constant2.envelope(t_vals), (3.0 + 2j) * np.ones_like(t_vals))

        t_vals = np.array([[1.1, 1.23], [0.1, 0.24]])
        self.assertAllClose(self.constant1.envelope(t_vals), np.array([[1.0, 1.0], [1.0, 1.0]]))
        self.assertAllClose(self.constant2.envelope(t_vals), (3.0 + 2j) * np.ones_like(t_vals))

    def test_complex_value(self):
        """Test complex_value evaluation."""
        self.assertAllClose(self.constant1.complex_value(0.0), 1.0)
        self.assertAllClose(self.constant1.complex_value(1.23), 1.0)

        self.assertAllClose(self.constant2.complex_value(1.1), (3.0 + 2j))
        self.assertAllClose(self.constant2.complex_value(1.23), (3.0 + 2j))

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.constant1.complex_value(t_vals), np.array([1.0, 1.0]))
        self.assertAllClose(self.constant2.complex_value(t_vals), (3.0 + 2j) * np.ones_like(t_vals))

        t_vals = np.array([[1.1, 1.23], [0.1, 0.24]])
        self.assertAllClose(
            self.constant1.complex_value(t_vals), np.array([[1.0, 1.0], [1.0, 1.0]])
        )
        self.assertAllClose(self.constant2.complex_value(t_vals), (3.0 + 2j) * np.ones_like(t_vals))

    def test_call(self):
        """Test __call__."""
        self.assertAllClose(self.constant1(0.0), 1.0)
        self.assertAllClose(self.constant1(1.23), 1.0)

        self.assertAllClose(self.constant2(1.1), (3.0))
        self.assertAllClose(self.constant2(1.23), (3.0))

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([1.1, 1.23])
        self.assertAllClose(self.constant1(t_vals), np.array([1.0, 1.0]))
        self.assertAllClose(self.constant2(t_vals), np.array([3.0, 3.0]))

        t_vals = np.array([[1.1, 1.23], [0.1, 0.24]])
        self.assertAllClose(self.constant1(t_vals), np.array([[1.0, 1.0], [1.0, 1.0]]))
        self.assertAllClose(self.constant2(t_vals), np.array([[3.0, 3.0], [3.0, 3.0]]))

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""

        const_conj = self.constant2.conjugate()
        self.assertAllClose(const_conj(1.1), 3.0)


class TestDiscreteSignal(QiskitDynamicsTestCase):
    """Tests for DiscreteSignal object."""

    def setUp(self):
        self.discrete1 = DiscreteSignal(dt=0.5, samples=np.array([1.0, 2.0, 3.0]), carrier_freq=3.0)
        self.discrete2 = DiscreteSignal(
            dt=0.5, samples=np.array([1.0 + 2j, 2.0 + 1j, 3.0]), carrier_freq=1.0, phase=3.0
        )

    def test_envelope(self):
        """Test envelope evaluation."""
        self.assertAllClose(self.discrete1.envelope(1.5), 0.0)
        self.assertAllClose(self.discrete1.envelope(0.0), 1.0)
        self.assertAllClose(self.discrete1.envelope(1.23), 3.0)
        self.assertAllClose(self.discrete1.envelope(1.49), 3.0)

        self.assertAllClose(self.discrete2.envelope(0.1), 1.0 + 2j)
        self.assertAllClose(self.discrete2.envelope(1.23), 3.0)
        self.assertAllClose(self.discrete2.envelope(1.49), 3.0)
        self.assertAllClose(self.discrete2.envelope(1.5), 0.0)

    def test_envelope_outside(self):
        """Test envelope evaluation outside of defined start and end"""
        self.assertAllClose(self.discrete1.envelope(-1.0), 0.0)
        self.assertAllClose(self.discrete1.envelope(3.0), 0.0)

    def test_envelope_vectorized(self):
        """Test vectorized evaluation of envelope."""
        t_vals = np.array([0.1, 1.23])
        self.assertAllClose(self.discrete1.envelope(t_vals), np.array([1.0, 3.0]))
        self.assertAllClose(self.discrete2.envelope(t_vals), np.array([1.0 + 2j, 3.0]))

        t_vals = np.array([[0.8, 1.23], [0.1, 0.24]])
        self.assertAllClose(self.discrete1.envelope(t_vals), np.array([[2.0, 3.0], [1.0, 1.0]]))
        self.assertAllClose(
            self.discrete2.envelope(t_vals), np.array([[2.0 + 1j, 3.0], [1 + 2j, 1.0 + 2j]])
        )

    def test_complex_value(self):
        """Test complex_value evaluation."""
        self.assertAllClose(self.discrete1.complex_value(0.0), 1.0)
        self.assertAllClose(
            self.discrete1.complex_value(1.23), 3.0 * np.exp(1j * 2 * np.pi * 3.0 * 1.23)
        )

        self.assertAllClose(
            self.discrete2.complex_value(0.1),
            (1.0 + 2j) * np.exp(1j * 2 * np.pi * 1.0 * 0.1 + 1j * 3.0),
        )
        self.assertAllClose(
            self.discrete2.complex_value(1.23), 3.0 * np.exp(1j * 2 * np.pi * 1.0 * 1.23 + 1j * 3.0)
        )

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([0.1, 1.23])
        phases = np.exp(1j * 2 * np.pi * 3.0 * t_vals)
        self.assertAllClose(self.discrete1.complex_value(t_vals), np.array([1.0, 3.0]) * phases)
        phases = np.exp(1j * 2 * np.pi * 1.0 * t_vals + 1j * 3.0)
        self.assertAllClose(
            self.discrete2.complex_value(t_vals), np.array([1.0 + 2j, 3.0]) * phases
        )

        t_vals = np.array([[0.8, 1.23], [0.1, 0.24]])
        phases = np.exp(1j * 2 * np.pi * 3.0 * t_vals)
        self.assertAllClose(
            self.discrete1.complex_value(t_vals), np.array([[2.0, 3.0], [1.0, 1.0]]) * phases
        )
        phases = np.exp(1j * 2 * np.pi * 1.0 * t_vals + 1j * 3.0)
        self.assertAllClose(
            self.discrete2.complex_value(t_vals),
            np.array([[2.0 + 1j, 3.0], [1 + 2j, 1.0 + 2j]]) * phases,
        )

    def test_call(self):
        """Test __call__."""
        self.assertAllClose(self.discrete1(0.0), 1.0)
        self.assertAllClose(
            self.discrete1(1.23), np.real(3.0 * np.exp(1j * 2 * np.pi * 3.0 * 1.23))
        )

        self.assertAllClose(
            self.discrete2(0.1), np.real((1.0 + 2j) * np.exp(1j * 2 * np.pi * 1.0 * 0.1 + 1j * 3.0))
        )
        self.assertAllClose(
            self.discrete2(1.23), np.real(3.0 * np.exp(1j * 2 * np.pi * 1.0 * 1.23 + 1j * 3.0))
        )

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([0.1, 1.23])
        phases = np.exp(1j * 2 * np.pi * 3.0 * t_vals)
        self.assertAllClose(self.discrete1(t_vals), np.real(np.array([1.0, 3.0]) * phases))
        phases = np.exp(1j * 2 * np.pi * 1.0 * t_vals + 1j * 3.0)
        self.assertAllClose(self.discrete2(t_vals), np.real(np.array([1.0 + 2j, 3.0]) * phases))

        t_vals = np.array([[0.8, 1.23], [0.1, 0.24]])
        phases = np.exp(1j * 2 * np.pi * 3.0 * t_vals)
        self.assertAllClose(
            self.discrete1(t_vals), np.real(np.array([[2.0, 3.0], [1.0, 1.0]]) * phases)
        )
        phases = np.exp(1j * 2 * np.pi * 1.0 * t_vals + 1j * 3.0)
        self.assertAllClose(
            self.discrete2(t_vals),
            np.real(np.array([[2.0 + 1j, 3.0], [1 + 2j, 1.0 + 2j]]) * phases),
        )

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""
        discrete_conj = self.discrete2.conjugate()
        self.assertAllClose(discrete_conj.samples, np.conjugate(self.discrete2.samples))
        self.assertAllClose(discrete_conj.carrier_freq, -self.discrete2.carrier_freq)
        self.assertAllClose(discrete_conj.phase, -self.discrete2.phase)
        self.assertAllClose(discrete_conj.dt, self.discrete2.dt)

    def test_add_samples(self):
        """Verify that add_samples function works correctly"""

        discrete1 = DiscreteSignal(dt=0.5, samples=np.array([]), carrier_freq=3.0)
        discrete2 = DiscreteSignal(
            dt=0.5, samples=np.array([1.0 + 2j, 2.0 + 1j, 3.0]), carrier_freq=1.0, phase=3.0
        )
        discrete3 = DiscreteSignal(dt=0.5, samples=[], carrier_freq=3.0)

        discrete1.add_samples(0, [4.0, 3.2])
        self.assertAllClose(discrete1.samples, [4.0, 3.2])

        discrete2.add_samples(5, [1.0, 5.0 + 2j])
        self.assertAllClose(discrete2.samples, [1.0 + 2j, 2.0 + 1j, 3.0, 0.0, 0.0, 1.0, 5.0 + 2j])

        discrete3.add_samples(5, [1.0, 2.0])
        self.assertAllClose(discrete3.samples, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0])


class TestSignalSum(QiskitDynamicsTestCase):
    """Test evaluation functions for ``SignalSum``."""

    def setUp(self):
        self.signal1 = Signal(np.vectorize(lambda t: 0.25), carrier_freq=0.3)
        self.signal2 = Signal(lambda t: 2.0 * (t**2), carrier_freq=0.1)
        self.signal3 = Signal(lambda t: 2.0 * (t**2) + 1j * t, carrier_freq=0.1, phase=-0.1)

        self.sig_sum1 = self.signal1 + self.signal2
        self.sig_sum2 = self.signal2 - self.signal3
        self.double_sig_sum = self.sig_sum1 - self.sig_sum2

    def test_envelope(self):
        """Test envelope evaluation."""
        t = 0.0
        self.assertAllClose(
            self.sig_sum1.envelope(t), [self.signal1.envelope(t), self.signal2.envelope(t)]
        )
        self.assertAllClose(
            self.sig_sum2.envelope(t), [self.signal2.envelope(t), -self.signal3.envelope(t)]
        )
        self.assertAllClose(
            self.double_sig_sum.envelope(t),
            [
                self.signal1.envelope(t),
                self.signal2.envelope(t),
                -self.signal2.envelope(t),
                self.signal3.envelope(t),
            ],
        )
        t = 1.23
        self.assertAllClose(
            self.sig_sum1.envelope(t), [self.signal1.envelope(t), self.signal2.envelope(t)]
        )
        self.assertAllClose(
            self.sig_sum2.envelope(t), [self.signal2.envelope(t), -self.signal3.envelope(t)]
        )
        self.assertAllClose(
            self.double_sig_sum.envelope(t),
            [
                self.signal1.envelope(t),
                self.signal2.envelope(t),
                -self.signal2.envelope(t),
                self.signal3.envelope(t),
            ],
        )

    def test_envelope_vectorized(self):
        """Test vectorized envelope evaluation."""
        t_vals = np.array([0.0, 1.23])
        self.assertAllClose(
            self.sig_sum1.envelope(t_vals),
            [[self.signal1.envelope(t), self.signal2.envelope(t)] for t in t_vals],
        )
        self.assertAllClose(
            self.sig_sum2.envelope(t_vals),
            [[self.signal2.envelope(t), -self.signal3.envelope(t)] for t in t_vals],
        )
        self.assertAllClose(
            self.double_sig_sum.envelope(t_vals),
            [
                [
                    self.signal1.envelope(t),
                    self.signal2.envelope(t),
                    -self.signal2.envelope(t),
                    self.signal3.envelope(t),
                ]
                for t in t_vals
            ],
        )
        t_vals = np.array([[0.0, 1.23], [0.1, 2.0]])
        self.assertAllClose(
            self.sig_sum1.envelope(t_vals),
            [
                [[self.signal1.envelope(t), self.signal2.envelope(t)] for t in t_row]
                for t_row in t_vals
            ],
        )
        self.assertAllClose(
            self.sig_sum2.envelope(t_vals),
            [
                [[self.signal2.envelope(t), -self.signal3.envelope(t)] for t in t_row]
                for t_row in t_vals
            ],
        )
        self.assertAllClose(
            self.double_sig_sum.envelope(t_vals),
            [
                [
                    [
                        self.signal1.envelope(t),
                        self.signal2.envelope(t),
                        -self.signal2.envelope(t),
                        self.signal3.envelope(t),
                    ]
                    for t in t_row
                ]
                for t_row in t_vals
            ],
        )

    def test_complex_value(self):
        """Test complex_value evaluation."""
        t = 0.0
        self.assertAllClose(
            self.sig_sum1.complex_value(t),
            self.signal1.complex_value(t) + self.signal2.complex_value(t),
        )
        self.assertAllClose(
            self.sig_sum2.complex_value(t),
            self.signal2.complex_value(t) - self.signal3.complex_value(t),
        )
        self.assertAllClose(
            self.double_sig_sum.complex_value(t),
            self.signal1.complex_value(t) + self.signal3.complex_value(t),
        )
        t = 1.23
        self.assertAllClose(
            self.sig_sum1.complex_value(t),
            self.signal1.complex_value(t) + self.signal2.complex_value(t),
        )
        self.assertAllClose(
            self.sig_sum2.complex_value(t),
            self.signal2.complex_value(t) - self.signal3.complex_value(t),
        )
        self.assertAllClose(
            self.double_sig_sum.complex_value(t),
            self.signal1.complex_value(t) + self.signal3.complex_value(t),
        )

    def test_complex_value_vectorized(self):
        """Test vectorized complex_value evaluation."""
        t_vals = np.array([0.0, 1.23])
        self.assertAllClose(
            self.sig_sum1.complex_value(t_vals),
            [self.signal1.complex_value(t) + self.signal2.complex_value(t) for t in t_vals],
        )
        self.assertAllClose(
            self.sig_sum2.complex_value(t_vals),
            [self.signal2.complex_value(t) - self.signal3.complex_value(t) for t in t_vals],
        )
        self.assertAllClose(
            self.double_sig_sum.complex_value(t_vals),
            [self.signal1.complex_value(t) + self.signal3.complex_value(t) for t in t_vals],
        )
        t_vals = np.array([[0.0, 1.23], [0.1, 2.0]])
        self.assertAllClose(
            self.sig_sum1.complex_value(t_vals),
            [
                [self.signal1.complex_value(t) + self.signal2.complex_value(t) for t in t_row]
                for t_row in t_vals
            ],
        )
        self.assertAllClose(
            self.sig_sum2.complex_value(t_vals),
            [
                [self.signal2.complex_value(t) - self.signal3.complex_value(t) for t in t_row]
                for t_row in t_vals
            ],
        )
        self.assertAllClose(
            self.double_sig_sum.complex_value(t_vals),
            [
                [self.signal1.complex_value(t) + self.signal3.complex_value(t) for t in t_row]
                for t_row in t_vals
            ],
        )

    def test_call(self):
        """Test __call__."""
        t = 0.0
        self.assertAllClose(self.sig_sum1(t), self.signal1(t) + self.signal2(t))
        self.assertAllClose(self.sig_sum2(t), self.signal2(t) - self.signal3(t))
        self.assertAllClose(self.double_sig_sum(t), self.signal1(t) + self.signal3(t))
        t = 1.23
        self.assertAllClose(self.sig_sum1(t), self.signal1(t) + self.signal2(t))
        self.assertAllClose(self.sig_sum2(t), self.signal2(t) - self.signal3(t))
        self.assertAllClose(self.double_sig_sum(t), self.signal1(t) + self.signal3(t))

    def test_call_vectorized(self):
        """Test vectorized __call__."""
        t_vals = np.array([0.0, 1.23])
        self.assertAllClose(
            self.sig_sum1(t_vals), [self.signal1(t) + self.signal2(t) for t in t_vals]
        )
        self.assertAllClose(
            self.sig_sum2(t_vals), [self.signal2(t) - self.signal3(t) for t in t_vals]
        )
        self.assertAllClose(
            self.double_sig_sum(t_vals), [self.signal1(t) + self.signal3(t) for t in t_vals]
        )
        t_vals = np.array([[0.0, 1.23], [0.1, 2.0]])
        self.assertAllClose(
            self.sig_sum1(t_vals),
            [[self.signal1(t) + self.signal2(t) for t in t_row] for t_row in t_vals],
        )
        self.assertAllClose(
            self.sig_sum2(t_vals),
            [[self.signal2(t) - self.signal3(t) for t in t_row] for t_row in t_vals],
        )
        self.assertAllClose(
            self.double_sig_sum(t_vals),
            [[self.signal1(t) + self.signal3(t) for t in t_row] for t_row in t_vals],
        )

    def test_conjugate(self):
        """Verify conjugate() functioning correctly."""

        sig_sum1_conj = self.sig_sum1.conjugate()
        self.assertAllClose(
            sig_sum1_conj.complex_value(2.313), np.conjugate(self.sig_sum1.complex_value(2.313))
        )
        self.assertAllClose(
            sig_sum1_conj.complex_value(0.1232), np.conjugate(self.sig_sum1.complex_value(0.1232))
        )


class TestDiscreteSignalSum(TestSignalSum):
    """Tests for DiscreteSignalSum."""

    def setUp(self):
        self.signal1 = Signal(np.vectorize(lambda t: 0.25), carrier_freq=0.3)
        self.signal2 = Signal(lambda t: 2.0 * (t**2), carrier_freq=0.1)
        self.signal3 = Signal(lambda t: 2.0 * (t**2) + 1j * t, carrier_freq=0.1, phase=-0.1)

        self.sig_sum1 = DiscreteSignalSum.from_SignalSum(
            self.signal1 + self.signal2, dt=0.5, start_time=0, n_samples=10
        )
        self.sig_sum2 = DiscreteSignalSum.from_SignalSum(
            self.signal2 - self.signal3, dt=0.5, start_time=0, n_samples=10
        )

        self.signal1 = DiscreteSignal.from_Signal(self.signal1, dt=0.5, start_time=0, n_samples=10)
        self.signal2 = DiscreteSignal.from_Signal(self.signal2, dt=0.5, start_time=0, n_samples=10)
        self.signal3 = DiscreteSignal.from_Signal(self.signal3, dt=0.5, start_time=0, n_samples=10)

        self.double_sig_sum = self.sig_sum1 - self.sig_sum2

    def test_empty_DiscreteSignal_to_sum(self):
        """Verify empty DiscreteSignal is converted to empty DiscreteSignalSum."""

        empty_sum = to_SignalSum(DiscreteSignal(dt=1.0, samples=[]))
        self.assertTrue(isinstance(empty_sum, DiscreteSignalSum))
        self.assertTrue(empty_sum.samples.shape == (1, 0))


class TestSignalList(QiskitDynamicsTestCase):
    """Test cases for SignalList class."""

    def setUp(self):
        self.sig = Signal(lambda t: t, carrier_freq=3.0)
        self.const = Signal(5.0)
        self.discrete_sig = DiscreteSignal(
            dt=0.5, samples=[1.0, 2.0, 3.0], carrier_freq=1.0, phase=0.1
        )

        self.sig_list = SignalList(
            [self.sig + self.const, self.sig * self.discrete_sig, self.const]
        )

    def test_eval(self):
        """Test evaluation of signal sum."""

        t_vals = np.array([0.12, 0.23, 1.23])

        expected = np.array(
            [
                self.sig(t_vals) + self.const(t_vals),
                self.sig(t_vals) * self.discrete_sig(t_vals),
                self.const(t_vals),
            ]
        ).transpose(1, 0)

        self.assertAllClose(self.sig_list(t_vals), expected)

    def test_complex_value(self):
        """Test evaluation of signal sum."""

        t_vals = np.array([0.12, 0.23, 1.23])

        expected = np.array(
            [
                self.sig.complex_value(t_vals) + self.const.complex_value(t_vals),
                np.real(self.sig.complex_value(t_vals)) * self.discrete_sig.complex_value(t_vals),
                self.const.complex_value(t_vals),
            ]
        ).transpose(1, 0)

        self.assertAllClose(self.sig_list.complex_value(t_vals), expected)

    def test_drift(self):
        """Test drift evaluation."""

        expected = np.array([self.const(0.0), 0, self.const(0.0)])
        self.assertAllClose(self.sig_list.drift, expected)

    def test_construction_with_numbers(self):
        """Test construction with non-wrapped constant values."""

        sig_list = SignalList([4.0, 2.0, Signal(lambda t: t)])
        # pylint: disable=no-member
        self.assertTrue(sig_list[0][0].is_constant)
        # pylint: disable=no-member
        self.assertTrue(sig_list[1][0].is_constant)
        # pylint: disable=no-member
        self.assertFalse(sig_list[2][0].is_constant)

        self.assertAllClose(sig_list(3.0), np.array([4.0, 2.0, 3.0]))


class TestSignalCollection(QiskitDynamicsTestCase):
    """Test cases for SignalCollection functionality."""

    def setUp(self):
        self.sig1 = Signal(lambda t: t, carrier_freq=0.1)
        self.sig2 = Signal(lambda t: t + 1j * t**2, carrier_freq=3.0, phase=1.0)
        self.sig3 = Signal(lambda t: t + 1j * t**2, carrier_freq=3.0, phase=1.2)

        self.discrete_sig1 = DiscreteSignal(dt=0.5, samples=[1.0, 2.0, 3.0], carrier_freq=3.0)
        self.discrete_sig2 = DiscreteSignal(dt=0.5, samples=[2.0, 2.1, 3.4], carrier_freq=2.1)
        self.discrete_sig3 = DiscreteSignal(
            dt=0.5, samples=[1.353, 2.223, 3.2312], carrier_freq=1.1
        )

        self.sig_sum = self.sig1 + self.sig2 + self.sig3
        self.discrete_sig_sum = self.discrete_sig1 + self.discrete_sig2 + self.discrete_sig3

    def test_SignalSum_subscript(self):
        """Test subscripting of SignalSum."""

        sub02 = self.sig_sum[[0, 2]]
        self.assertTrue(len(sub02) == 2)
        t_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertAllClose(sub02(t_vals), self.sig1(t_vals) + self.sig3(t_vals))

    def test_DiscreteSignalSum_subscript(self):
        """Test subscripting of SignalSum."""

        sub02 = self.discrete_sig_sum[[0, 2]]
        self.assertTrue(len(sub02) == 2)
        t_vals = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) / 4.0
        self.assertAllClose(sub02(t_vals), self.discrete_sig1(t_vals) + self.discrete_sig3(t_vals))

    def test_SignalSum_iterator(self):
        """Test iteration of SignalSum."""

        sum_val = 0.0
        for sig in self.sig_sum:
            sum_val += sig(3.0)

        self.assertAllClose(sum_val, self.sig_sum(3.0))

    def test_DiscreteSignalSum_iterator(self):
        """Test iteration of DiscreteSignalSum."""

        sum_val = 0.0
        for sig in self.discrete_sig_sum:
            sum_val += sig(3.0)

        self.assertAllClose(sum_val, self.discrete_sig_sum(3.0))


class TestSignalJax(TestSignal, TestJaxBase):
    """Jax version of TestSignal."""


class TestConstantJax(TestSignal, TestJaxBase):
    """Jax version of TestConstant."""


class TestDiscreteSignalJax(TestDiscreteSignal, TestJaxBase):
    """Jax version of TestDiscreteSignal."""


class TestSignalSumJax(TestSignalSum, TestJaxBase):
    """Jax version of TestSignalSum."""


class TestDiscreteSignalSumJax(TestDiscreteSignalSum, TestJaxBase):
    """Jax version of TestSignalSum."""


class TestSignalListJax(TestSignalList, TestJaxBase):
    """Jax version of TestSignalList."""


class TestSignalsJaxTransformations(QiskitDynamicsTestCase, TestJaxBase):
    """Test cases for jax transformations of signals."""

    def setUp(self):
        self.signal = Signal(lambda t: t**2, carrier_freq=3.0)
        self.constant = Signal(3 * np.pi)
        self.discrete_signal = DiscreteSignal(
            dt=0.5, samples=jnp.ones(20, dtype=complex), carrier_freq=2.0
        )
        self.signal_sum = self.signal + self.discrete_signal
        self.discrete_signal_sum = DiscreteSignalSum.from_SignalSum(
            self.signal_sum, dt=0.5, n_samples=20
        )
        self.signal_list = SignalList([self.signal, self.signal_sum, self.discrete_signal])

    def test_jit_eval(self):
        """Test jit-compilation of signal evaluation."""
        self._test_jit_signal_eval(self.signal, t=2.1)
        self._test_jit_signal_eval(self.constant, t=2.1)
        self._test_jit_signal_eval(self.discrete_signal, t=2.1)
        self._test_jit_signal_eval(self.signal_sum, t=2.1)
        self._test_jit_signal_eval(self.discrete_signal_sum, t=2.1)

    def test_jit_grad_constant_construct(self):
        """Test jitting and grad through a function which constructs a constant signal."""

        def eval_const(a):
            a = Array(a)
            return Signal(a)(1.1).data

        jit_eval = jit(eval_const)
        self.assertAllClose(jit_eval(3.0), 3.0)

        jit_grad_eval = jit(grad(eval_const))
        self.assertAllClose(jit_grad_eval(3.0), 1.0)

    def test_jit_grad_carrier_freq_construct(self):
        """Test jit/gradding through a function that constructs a signal and takes carrier frequency
        as an argument.
        """

        def eval_sig(a, v, t):
            a = Array(a)
            v = Array(v)
            return Array(Signal(a, v)(t)).data

        jit_eval = jit(eval_sig)
        self.assertAllClose(jit_eval(1.0, 1.0, 1.0), 1.0)

        jit_grad_eval = jit(grad(eval_sig))
        self.assertAllClose(jit_grad_eval(1.0, 1.0, 1.0), 1.0)

    def test_signal_list_jit_eval(self):
        """Test jit-compilation of SignalList evaluation."""
        call_jit = jit(lambda t: Array(self.signal_list(t)).data)

        t_vals = np.array([0.123, 0.5324, 1.232])
        self.assertAllClose(call_jit(t_vals), self.signal_list(t_vals))

    def test_jit_grad_eval(self):
        """Test taking the gradient and then jitting signal evaluation functions."""
        t = 2.1
        self._test_grad_eval(
            self.signal,
            t=t,
            sig_deriv_val=2 * t * np.cos(2 * np.pi * 3.0 * t)
            + (t**2) * (-2 * np.pi * 3) * np.sin(2 * np.pi * 3.0 * t),
            complex_deriv_val=2 * t * np.exp(1j * 2 * np.pi * 3.0 * t)
            + (t**2) * (1j * 2 * np.pi * 3.0) * np.exp(1j * 2 * np.pi * 3.0 * t),
        )
        self._test_grad_eval(self.constant, t=t, sig_deriv_val=0.0, complex_deriv_val=0.0)
        self._test_grad_eval(
            self.discrete_signal,
            t=t,
            sig_deriv_val=np.real(self.discrete_signal.samples[5])
            * (-2 * np.pi * 2.0)
            * np.sin(2 * np.pi * 2.0 * t),
            complex_deriv_val=self.discrete_signal.samples[5]
            * (1j * 2 * np.pi * 2.0)
            * np.exp(1j * 2 * np.pi * 2.0 * t),
        )
        self._test_grad_eval(
            self.signal_sum,
            t=t,
            sig_deriv_val=2 * t * np.cos(2 * np.pi * 3.0 * t)
            + (t**2) * (-2 * np.pi * 3) * np.sin(2 * np.pi * 3.0 * t)
            + np.real(self.discrete_signal.samples[5])
            * (-2 * np.pi * 2.0)
            * np.sin(2 * np.pi * 2.0 * t),
            complex_deriv_val=2 * t * np.exp(1j * 2 * np.pi * 3.0 * t)
            + (t**2) * (1j * 2 * np.pi * 3.0) * np.exp(1j * 2 * np.pi * 3.0 * t)
            + self.discrete_signal.samples[5]
            * (1j * 2 * np.pi * 2.0)
            * np.exp(1j * 2 * np.pi * 2.0 * t),
        )
        self._test_grad_eval(
            self.discrete_signal_sum,
            t=t,
            sig_deriv_val=(2.25**2) * (-2 * np.pi * 3) * np.sin(2 * np.pi * 3.0 * t)
            + np.real(self.discrete_signal.samples[5])
            * (-2 * np.pi * 2.0)
            * np.sin(2 * np.pi * 2.0 * t),
            complex_deriv_val=(2.25**2)
            * (1j * 2 * np.pi * 3.0)
            * np.exp(1j * 2 * np.pi * 3.0 * t)
            + self.discrete_signal.samples[5]
            * (1j * 2 * np.pi * 2.0)
            * np.exp(1j * 2 * np.pi * 2.0 * t),
        )

    def _test_jit_signal_eval(self, signal, t=2.1):
        """jit compilation and evaluation of main signal functions."""
        sig_call_jit = jit(lambda t: Array(signal(t)).data)
        self.assertAllClose(sig_call_jit(t), signal(t))
        sig_envelope_jit = jit(lambda t: Array(signal.envelope(t)).data)
        self.assertAllClose(sig_envelope_jit(t), signal.envelope(t))
        sig_complex_value_jit = jit(lambda t: Array(signal.complex_value(t)).data)
        self.assertAllClose(sig_complex_value_jit(t), signal.complex_value(t))

    def _test_grad_eval(self, signal, t, sig_deriv_val, complex_deriv_val):
        """Test chained grad and jit compilation."""
        sig_call_jit = jit(grad(lambda t: Array(signal(t)).data))
        self.assertAllClose(sig_call_jit(t), sig_deriv_val)
        sig_complex_value_jit_re = jit(grad(lambda t: np.real(Array(signal.complex_value(t))).data))
        sig_complex_value_jit_imag = jit(
            grad(lambda t: np.imag(Array(signal.complex_value(t))).data)
        )
        self.assertAllClose(sig_complex_value_jit_re(t), np.real(complex_deriv_val))
        self.assertAllClose(sig_complex_value_jit_imag(t), np.imag(complex_deriv_val))
