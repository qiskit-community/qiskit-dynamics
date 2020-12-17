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
Pulse schedule to Signals converter.
"""

from typing import List
import numpy as np

from qiskit.pulse import Schedule, Play, ShiftPhase, SetPhase, ShiftFrequency, SetFrequency
from qiskit import QiskitError
from qiskit_ode.signals import PiecewiseConstant


class InstructionToSignals:
    """Converts pulse instructions to Signals for the Aer simulator."""

    def __init__(self, dt: float, carriers: List[float] = None):
        """

        Args:
            dt: length of the samples.
            carriers: a list of carrier frequencies. If it is not None there
                must be at least as many carrier frequencies as there are
                channels in the schedules that will be converted.
        """

        self._dt = dt
        self._carriers = carriers

    def get_signals(self, schedule: Schedule) -> List[PiecewiseConstant]:
        """
        Args:
            schedule: The schedule to represent in terms of signals.

        Returns:
            a list of piecewise constant signals.

        Raises:
            qiskit.QiskitError: if not enough frequencies supplied
        """

        if self._carriers and len(self._carriers) < len(schedule.channels):
            raise QiskitError('Not enough carrier frequencies supplied.')

        signals, phases, frequency_shifts = {}, {}, {}

        for idx, chan in enumerate(schedule.channels):
            if self._carriers:
                carrier_freq = self._carriers[idx]
            else:
                carrier_freq = 0.

            phases[chan.name] = 0.
            frequency_shifts[chan.name] = 0.
            signals[chan.name] = PiecewiseConstant(samples=[],
                                                   dt=self._dt,
                                                   name=chan.name,
                                                   carrier_freq=carrier_freq)

        for start_sample, inst in schedule.instructions:
            chan = inst.channel.name
            phi = phases[chan]
            freq = frequency_shifts[chan]

            if isinstance(inst, Play):
                samples = []
                start_idx = len(signals[chan].samples)
                for idx, sample in enumerate(inst.pulse.get_waveform().samples):
                    time = self._dt * (idx + start_idx)
                    samples.append(sample * np.exp(2.0j * np.pi * freq * time
                                                   + 1.0j * phi))

                signals[chan].add_samples(start_sample, samples)

            if isinstance(inst, ShiftPhase):
                phases[chan] += inst.phase

            if isinstance(inst, ShiftFrequency):
                frequency_shifts[chan] += inst.frequency

            if isinstance(inst, SetPhase):
                phases[chan] = inst.phase

            if isinstance(inst, SetFrequency):
                frequency_shifts[chan] = (inst.frequency -
                                          signals[chan].carrier_freq)

        return list(signals.values())
