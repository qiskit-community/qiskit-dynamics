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
"""
Tests to convert from pulse schedules to signals.
"""

from ddt import ddt, data, unpack
import numpy as np
import sympy as sym

from qiskit import pulse
from qiskit.pulse import (
    Schedule,
    DriveChannel,
    ControlChannel,
    MeasureChannel,
    Play,
    Delay,
    Drag,
    ShiftFrequency,
    SetFrequency,
    GaussianSquare,
    ShiftPhase,
    Gaussian,
    Constant,
    Waveform,
    SymbolicPulse,
)
from qiskit.pulse.transforms.canonicalization import block_to_schedule
from qiskit import QiskitError

from qiskit_dynamics.array import Array
from qiskit_dynamics.pulse import InstructionToSignals
from qiskit_dynamics.pulse.pulse_to_signals import (
    get_samples,
    _lru_cache_expr,
)
from qiskit_dynamics.signals import DiscreteSignal

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    import jax
    import jax.numpy as jnp
# pylint: disable=broad-except
except Exception:
    pass


class TestPulseToSignals(QiskitDynamicsTestCase):
    """Tests the conversion between pulse schedules and signals."""

    def setUp(self):
        """Setup the tests."""

        super().setUp()
        # Typical length of samples in units of dt in IBM real backends is 0.222.
        self._dt = 0.222

    def test_pulse_to_signals(self):
        """Generic test."""

        sched = Schedule(name="Schedule")
        sched += Play(Drag(duration=20, amp=0.5, sigma=4, beta=0.5), DriveChannel(0))
        sched += ShiftPhase(1.0, DriveChannel(0))
        sched += Play(Drag(duration=20, amp=0.5, sigma=4, beta=0.5), DriveChannel(0))
        sched += ShiftFrequency(0.5, DriveChannel(0))
        sched += Play(GaussianSquare(duration=200, amp=0.3, sigma=4, width=150), DriveChannel(0))

        test_gaussian = GaussianSquare(duration=200, amp=0.3, sigma=4, width=150)
        sched = sched.insert(0, Play(test_gaussian, DriveChannel(1)))

        converter = InstructionToSignals(dt=1, carriers=None)

        signals = converter.get_signals(sched)

        self.assertEqual(len(signals), 2)
        self.assertTrue(isinstance(signals[0], DiscreteSignal))
        self.assertTrue(isinstance(signals[0], DiscreteSignal))

        samples = test_gaussian.get_waveform().samples
        self.assertTrue(np.allclose(signals[1].samples[0 : len(samples)], samples))

    def test_shift_phase_to_signals(self):
        """Test that a shift phase gives negative envelope."""

        gaussian = Gaussian(duration=20, amp=0.5, sigma=4)

        sched = Schedule(name="Schedule")
        sched += ShiftPhase(np.pi, DriveChannel(0))
        sched += Play(gaussian, DriveChannel(0))

        converter = InstructionToSignals(dt=1, carriers=None)
        signals = converter.get_signals(sched)

        self.assertTrue(signals[0].samples[10] < 0)
        self.assertTrue(gaussian.get_waveform().samples[10] > 0)

    def test_carriers_and_dt(self):
        """Test that the carriers go into the signals."""

        sched = Schedule(name="Schedule")
        sched += Play(Gaussian(duration=20, amp=0.5, sigma=4), DriveChannel(0))

        converter = InstructionToSignals(dt=self._dt, carriers={"d0": 5.5e9})
        signals = converter.get_signals(sched)

        self.assertEqual(signals[0].carrier_freq, 5.5e9)
        # pylint: disable=protected-access
        self.assertEqual(signals[0]._dt, self._dt)

    def test_shift_frequency(self):
        """Test that the frequency is properly taken into account."""

        sched = Schedule()
        sched += ShiftFrequency(1.0, DriveChannel(0))
        sched += Play(Constant(duration=10, amp=1.0), DriveChannel(0))

        converter = InstructionToSignals(dt=self._dt, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)

        for idx in range(10):
            self.assertEqual(signals[0].samples[idx], np.exp(2.0j * idx * np.pi * 1.0 * self._dt))

    def test_set_frequency(self):
        """Test that SetFrequency is properly converted."""

        sched = Schedule()
        sched += SetFrequency(4.0, DriveChannel(0))
        sched += Play(Constant(duration=10, amp=1.0), DriveChannel(0))

        converter = InstructionToSignals(dt=self._dt, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)

        for idx in range(10):
            self.assertEqual(signals[0].samples[idx], np.exp(2.0j * idx * np.pi * -1.0 * self._dt))

    def test_set_and_shift_frequency(self):
        """Test that ShiftFrequency after SetFrequency is properly converted. It confirms
        implementation of phase accumulation is correct."""

        duration = 20
        sched = Schedule()
        sched += SetFrequency(5.5, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))
        sched += SetFrequency(6, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))
        sched += ShiftFrequency(-0.5, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))

        freq_shift = 0.5
        phase_accumulation = 0.0
        all_samples = np.exp(2j * np.pi * freq_shift * self._dt * np.arange(0, duration))

        freq_shift = 1.0
        phase_accumulation -= (6.0 - 5.5) * duration * self._dt
        all_samples = np.append(
            all_samples,
            np.exp(
                2j
                * np.pi
                * (freq_shift * self._dt * np.arange(duration, 2 * duration) + phase_accumulation)
            ),
        )

        freq_shift = 0.5
        phase_accumulation -= -0.5 * 2 * duration * self._dt
        all_samples = np.append(
            all_samples,
            np.exp(
                2j
                * np.pi
                * (
                    freq_shift * self._dt * np.arange(2 * duration, 3 * duration)
                    + phase_accumulation
                )
            ),
        )

        converter = InstructionToSignals(dt=self._dt, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)
        self.assertAllClose(signals[0].samples, all_samples)

    def test_delay(self):
        """Test that Delay is properly reflected."""

        sched = Schedule()
        sched += Play(Constant(duration=10, amp=1.0), DriveChannel(0))
        sched += Delay(10, DriveChannel(0))
        sched += Play(Constant(duration=10, amp=1.0), DriveChannel(0))

        converter = InstructionToSignals(dt=self._dt, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)
        samples_with_delay = np.array([1] * 10 + [0] * 10 + [1] * 10)
        for idx in range(30):
            self.assertEqual(signals[0].samples[idx], samples_with_delay[idx])

    def test_delay_and_shift_frequency(self):
        """Test that delay after SetFrequency is properly converted.
        It confirms implementation of phase accumulation is correct."""

        duration = 20
        sched = Schedule()
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))
        sched += ShiftFrequency(1.0, DriveChannel(0))
        sched += Delay(duration, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))

        freq_shift = 1.0
        phase_accumulation = -1.0 * duration * self._dt
        phase_accumulation = -1.0 * duration * self._dt
        all_samples = np.append(
            np.append(np.ones(duration), np.zeros(duration)),
            np.exp(
                2j
                * np.pi
                * (
                    freq_shift * self._dt * np.arange(2 * duration, 3 * duration)
                    + phase_accumulation
                )
            ),
        )

        converter = InstructionToSignals(dt=self._dt, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)
        self.assertAllClose(signals[0].samples, all_samples)

    def test_set_and_shift_frequency(self):
        """Test that ShiftFrequency after SetFrequency is properly converted. It confirms
        implementation of phase accumulation is correct."""

        duration = 20
        unit_dt = 0.222
        sched = Schedule()
        sched += SetFrequency(5.5, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))
        sched += SetFrequency(6, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))
        sched += ShiftFrequency(-0.5, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))

        freq_shift = 0.5
        phase_accumulation = 0.0
        all_samples = np.exp(2j * np.pi * freq_shift * unit_dt * np.arange(0, duration))

        freq_shift = 1.0
        phase_accumulation -= (6.0 - 5.5) * duration * unit_dt
        all_samples = np.append(
            all_samples,
            np.exp(
                2j
                * np.pi
                * (freq_shift * unit_dt * np.arange(duration, 2 * duration) + phase_accumulation)
            ),
        )

        freq_shift = 0.5
        phase_accumulation -= -0.5 * 2 * duration * unit_dt
        all_samples = np.append(
            all_samples,
            np.exp(
                2j
                * np.pi
                * (
                    freq_shift * unit_dt * np.arange(2 * duration, 3 * duration)
                    + phase_accumulation
                )
            ),
        )

        converter = InstructionToSignals(dt=unit_dt, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)
        self.assertAllClose(signals[0].samples, all_samples)

    def test_delay(self):
        """Test that Delay is properly reflected."""

        sched = Schedule()
        sched += Play(Constant(duration=10, amp=1.0), DriveChannel(0))
        sched += Delay(10, DriveChannel(0))
        sched += Play(Constant(duration=10, amp=1.0), DriveChannel(0))

        converter = InstructionToSignals(dt=0.222, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)
        samples_with_delay = np.array([1] * 10 + [0] * 10 + [1] * 10)
        for idx in range(30):
            self.assertEqual(signals[0].samples[idx], samples_with_delay[idx])

    def test_delay_and_shift_frequency(self):
        """Test that delay after SetFrequency is properly converted.
        It confirms implementation of phase accumulation is correct."""

        duration = 20
        unit_dt = 0.222
        sched = Schedule()
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))
        sched += ShiftFrequency(1.0, DriveChannel(0))
        sched += Delay(duration, DriveChannel(0))
        sched += Play(Constant(duration=duration, amp=1.0), DriveChannel(0))

        freq_shift = 1.0
        phase_accumulation = -1.0 * duration * unit_dt
        phase_accumulation = -1.0 * duration * unit_dt
        all_samples = np.append(
            np.append(np.ones(duration), np.zeros(duration)),
            np.exp(
                2j
                * np.pi
                * (
                    freq_shift * unit_dt * np.arange(2 * duration, 3 * duration)
                    + phase_accumulation
                )
            ),
        )

        converter = InstructionToSignals(dt=unit_dt, carriers={"d0": 5.0})
        signals = converter.get_signals(sched)
        self.assertAllClose(signals[0].samples, all_samples)

    def test_uneven_pulse_length(self):
        """Test conversion when length of pulses on a schedule is uneven."""

        schedule = Schedule()
        schedule |= Play(Waveform(np.ones(10)), DriveChannel(0))
        schedule += Play(Constant(20, 1), DriveChannel(1))

        converter = InstructionToSignals(dt=0.1, carriers={"d0": 2.0, "d1": 3.0})

        signals = converter.get_signals(schedule)

        self.assertTrue(signals[0].duration == 20)
        self.assertTrue(signals[1].duration == 20)

        self.assertAllClose(signals[0].samples, np.append(np.ones(10), np.zeros(10)))
        self.assertAllClose(signals[1].samples, np.ones(20))

        self.assertTrue(signals[0].carrier_freq == 2.0)
        self.assertTrue(signals[1].carrier_freq == 3.0)

    def test_different_start_times(self):
        """Test pulse schedule containing channels with first instruction at different times."""

        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(1))

        schedule = pulse.transforms.block_to_schedule(schedule)

        converter = InstructionToSignals(
            dt=0.1, channels=["d0", "d1"], carriers={"d0": 5.0, "d1": 3.1}
        )
        signals = converter.get_signals(schedule)
        # construct samples
        constant_samples = np.ones(5, dtype=float) * 0.9
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(5))
        samples1 = np.append(np.zeros(10), gauss_samples)

        # set tolerance from 1e-14 to 1e-7
        # to match the accuracy of clipping samples in class WaveForm.
        # see https://github.com/Qiskit/qiskit-terra/blob/
        # ee0b0368e72913cddf1c80ed95bc55e174c65046/qiskit/pulse/library/waveform.py#L56
        self.assertAllClose(signals[0].samples, samples0, atol=1e-7, rtol=1e-7)
        self.assertAllClose(signals[1].samples, samples1, atol=1e-7, rtol=1e-7)
        self.assertTrue(signals[0].carrier_freq == 5.0)
        self.assertTrue(signals[1].carrier_freq == 3.1)

    def test_multiple_channels_with_gaps(self):
        """Test building a schedule with multiple channels and gaps."""

        with pulse.build() as schedule:
            with pulse.align_sequential():
                pulse.play(pulse.Constant(duration=5, amp=0.9), pulse.DriveChannel(0))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(1))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(2))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(3))
                pulse.shift_phase(np.pi / 2.98, pulse.DriveChannel(0))
                pulse.play(pulse.Gaussian(duration=5, amp=0.983, sigma=2.0), pulse.DriveChannel(0))

        schedule = pulse.transforms.block_to_schedule(schedule)

        converter = InstructionToSignals(
            dt=0.1,
            channels=["d0", "d1", "d2", "d3"],
            carriers={"d0": 5.0, "d1": 3.1, "d2": 0, "d3": 4.0},
        )
        signals = converter.get_signals(schedule)

        # construct samples
        constant_samples = np.ones(5, dtype=float) * 0.9
        phase = np.exp(1j * np.pi / 2.98)
        gauss_samples = pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        samples0 = np.append(np.append(constant_samples, gauss_samples * phase), np.zeros(15))
        phase2 = np.exp(2 * 1j * np.pi / 2.98)
        samples0 = np.append(samples0, gauss_samples * phase2)
        samples1 = np.append(np.append(np.zeros(10), gauss_samples), np.zeros(15))
        samples2 = np.append(np.zeros(15), np.append(gauss_samples, np.zeros(10)))
        samples3 = np.append(np.zeros(20), np.append(gauss_samples, np.zeros(5)))

        self.assertAllClose(signals[0].samples, samples0, atol=1e-7, rtol=1e-7)
        self.assertAllClose(signals[1].samples, samples1, atol=1e-7, rtol=1e-7)
        self.assertAllClose(signals[2].samples, samples2, atol=1e-7, rtol=1e-7)
        self.assertAllClose(signals[3].samples, samples3, atol=1e-7, rtol=1e-7)
        self.assertTrue(signals[0].carrier_freq == 5.0)
        self.assertTrue(signals[1].carrier_freq == 3.1)
        self.assertTrue(signals[2].carrier_freq == 0.0)
        self.assertTrue(signals[3].carrier_freq == 4.0)

    def test_get_samples(self):
        """Test get samples of Pulse not get_waveform but get_samples function."""
        gauss_get_waveform_samples = (
            pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        )
        gauss_get_samples = get_samples(Gaussian(duration=5, amp=0.983, sigma=2.0))
        self.assertTrue(isinstance(gauss_get_samples, np.ndarray))
        self.assertAllClose(gauss_get_samples, gauss_get_waveform_samples, atol=1e-7, rtol=1e-7)

    def test_lru_cache_expr(self):
        """Test lru_cache of lru_cache_expr function."""
        gauss_envelop = Gaussian(duration=5, amp=0.983, sigma=2.0).envelope
        self.assertTrue(
            _lru_cache_expr(gauss_envelop, Array.default_backend())
            is _lru_cache_expr(gauss_envelop, Array.default_backend())
        )


class TestJaxGetSamples(QiskitDynamicsTestCase, TestJaxBase):
    """Tests get_samples function by using Jax."""

    def setUp(self):
        """Set up gaussian waveform samples for comparison."""
        self.gauss_get_waveform_samples = (
            pulse.Gaussian(duration=5, amp=0.983, sigma=2.0).get_waveform().samples
        )
        self.constant_get_waveform_samples = (
            pulse.Constant(duration=5, amp=0.1).get_waveform().samples
        )

    def test_get_samples(self):
        """Test get samples of Pulse not get_waveform but get_samples function in Jax case."""
        gauss_get_samples = get_samples(Gaussian(duration=5, amp=0.983, sigma=2.0))
        self.assertTrue(isinstance(gauss_get_samples, jnp.ndarray))
        self.assertAllClose(
            gauss_get_samples, self.gauss_get_waveform_samples, atol=1e-7, rtol=1e-7
        )

    def test_jit_get_samples(self):
        """Test compiling to get samples of Pulse."""

        def jit_func(amp):
            parameters = {"amp": amp}
            _time, _amp, _duration = sym.symbols("t, amp, duration")
            envelope_expr = _amp * sym.Piecewise(
                (1, sym.And(_time >= 0, _time <= _duration)), (0, True)
            )
            valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0
            # we can use only SymbolicPulse when jax-jitting
            # bacause jax-jitting doesn't correspond to validate_parameters in qiskit.pulse.
            instance = SymbolicPulse(
                pulse_type="Constant",
                duration=5,
                parameters=parameters,
                envelope=envelope_expr,
                valid_amp_conditions=valid_amp_conditions_expr,
            )
            return get_samples(instance)

        self.jit_wrap(jit_func)(0.1)
        self.jit_grad_wrap(jit_func)(0.1)
        jit_samples = jax.jit(jit_func)(0.1)
        self.assertAllClose(jit_samples, self.constant_get_waveform_samples, atol=1e-7, rtol=1e-7)


@ddt
class TestPulseToSignalsFiltering(QiskitDynamicsTestCase):
    """Test the extraction of signals when specifying channels."""

    def setUp(self):
        """Setup the tests."""

        super().setUp()

        # Typical length of samples in units of dt in IBM real backends is 0.222.
        self._dt = 0.222

        # Drags on all qubits, then two CRs, then readout all qubits.
        with pulse.build(name="test schedule") as schedule:
            with pulse.align_sequential():
                with pulse.align_left():
                    for chan_idx in [0, 1, 2, 3]:
                        pulse.play(Drag(160, 0.5, 40, 0.1), DriveChannel(chan_idx))

                with pulse.align_sequential():
                    for chan_idx in [0, 1]:
                        pulse.play(GaussianSquare(660, 0.2, 40, 500), ControlChannel(chan_idx))

                with pulse.align_left():
                    for chan_idx in [0, 1, 2, 3]:
                        pulse.play(GaussianSquare(660, 0.2, 40, 500), MeasureChannel(chan_idx))

        self._schedule = block_to_schedule(schedule)

    @unpack
    @data(
        ({"d0": 5.0, "d2": 5.1, "u0": 5.0, "u1": 5.1}, ["d0", "d2", "u0", "u1"]),
        ({"m0": 5.0, "m1": 5.1, "m2": 5.0, "m3": 5.1}, ["m0", "m1", "m2", "m3"]),
        ({"m0": 5.0, "m1": 5.1, "d0": 5.0, "d1": 5.1}, ["m0", "m1", "d0", "d1"]),
        ({"d1": 5.0}, ["d1"]),
        ({"d123": 5.0}, ["d123"]),
    )
    def test_channel_combinations(self, carriers, channels):
        """Test that we can filter out channels in the right order and number."""

        converter = InstructionToSignals(dt=self._dt, carriers=carriers, channels=channels)

        signals = converter.get_signals(self._schedule)

        self.assertEqual(len(signals), len(channels))
        for idx, chan_name in enumerate(channels):
            self.assertEqual(signals[idx].name, chan_name)

    def test_empty_signal(self):
        """Test that requesting a channel that is not in the schedule gives and empty signal."""

        converter = InstructionToSignals(dt=self._dt, carriers={"d123": 1.0}, channels=["d123"])

        signals = converter.get_signals(self._schedule)

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].duration, 0)

    @data("123", "s123", "", "d")
    def test_get_channel_raise(self, channel_name):
        """Test that getting channel instances works well."""

        converter = InstructionToSignals(dt=self._dt)

        with self.assertRaisesRegex(QiskitError, f"channel name {channel_name}"):
            converter._get_channel(channel_name)
