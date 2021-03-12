# -*- coding: utf-8 -*-

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

"""
Module for representation of model coefficients.
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional

import numpy as np
from matplotlib import pyplot as plt

from qiskit import QiskitError
from qiskit_ode.dispatch import Array


class BaseSignal(ABC):
    """Base class for a time-dependent mixed signal."""

    def __init__(self, name: str = None):
        """Init function."""
        self._name = name
        self._carrier_freq = Array(0.0)
        self._phase = Array(0.0)

    @property
    def carrier_freq(self):
        """Return the carrier frequency of the signal."""
        return self._carrier_freq

    @property
    def phase(self):
        """Return the phase of the Signal."""
        return self._phase

    @abstractmethod
    def conjugate(self):
        """Return a new signal that is the complex conjugate of self."""

    @abstractmethod
    def envelope_value(self, t: float = 0.0) -> complex:
        """Evaluates the envelope at time t."""

    @abstractmethod
    def value(self, t: float = 0.0) -> complex:
        """Return the value of the signal at time t."""

    def to_pwc(self, dt: float, n_samples: int, start_time: float = 0.0) -> "PiecewiseConstant":
        """
        Converts a signal to a piecewise constant signal.

        Args:
            dt: Time increment to use.
            n_samples: number of steps to resample with.
            start_time: start time from which to resample.

        Returns:
            A piecewiseConstant signal.
        """
        samples = [self.envelope_value(dt * idx + start_time) for idx in range(n_samples)]

        return PiecewiseConstant(
            dt, samples, start_time=start_time, carrier_freq=self.carrier_freq, phase=self.phase
        )

    def __mul__(self, other):
        return signal_multiply(self, other)

    def __rmul__(self, other):
        return signal_multiply(self, other)

    def __add__(self, other):
        return signal_add(self, other)

    def __radd__(self, other):
        return signal_add(self, other)

    def plot(self, t0: float, tf: float, n: int):
        """Plot the mixed signal over an interval.

        Args:
            t0: initial time
            tf: final time
            n: number of points to sample in interval.
        """
        x_vals = np.linspace(t0, tf, n)

        sig_vals = []
        for x in x_vals:
            sig_vals.append(self.value(x))

        plt.plot(x_vals, np.real(sig_vals))
        plt.plot(x_vals, np.imag(sig_vals))

    def plot_envelope(self, t0: float, tf: float, n: int):
        """Plot the envelope over an interval.

        Args:
            t0: initial time
            tf: final time
            n: number of points to sample in interval.
        """
        x_vals = np.linspace(t0, tf, n)

        sig_vals = []
        for x in x_vals:
            sig_vals.append(self.envelope_value(x))

        plt.plot(x_vals, np.real(sig_vals))
        plt.plot(x_vals, np.imag(sig_vals))


class Signal(BaseSignal):
    """The most general mixed signal type, represented by a callable
    envelope function and a carrier frequency.
    """

    def __init__(
        self,
        envelope: Union[Callable, complex, float, int],
        carrier_freq: float = 0.0,
        phase: float = 0.0,
        name: str = None,
    ):
        """
        Initializes a signal given by an envelop and an optional carrier.

        Args:
            envelope: Envelope function of the signal.
            carrier_freq: Frequency of the carrier.
            phase: The phase of the carrier.
            name: name of signal.
        """
        super().__init__(name)

        if isinstance(envelope, (float, int)):
            envelope = complex(envelope)

        if isinstance(envelope, complex):
            self.envelope = lambda t: envelope
        else:
            self.envelope = envelope

        self._carrier_freq = Array(carrier_freq)
        self._phase = Array(phase)

    def envelope_value(self, t: float = 0.0) -> Array:
        """Evaluates the envelope at time t."""
        return Array(self.envelope(t))

    def value(self, t: float = 0.0) -> Array:
        """Return the value of the signal at time t."""
        arg = 1j * 2 * np.pi * self.carrier_freq * t + 1j * self.phase
        return self.envelope_value(t) * np.exp(arg)

    def conjugate(self):
        """Return a new signal that is the complex conjugate of this one"""
        return Signal(lambda t: self.envelope_value(t).conjugate(), -self.carrier_freq, -self.phase)


class Constant(BaseSignal):
    """
    Constant signal that has no carrier and may appear in a model.
    """

    def __init__(self, value: complex, name: str = None):
        """Initialize a constant signal.

        Args:
            value: the constant.
            name: name of the constant.
        """
        self._value = value
        super().__init__(name)

    def envelope_value(self, t: float = 0.0) -> complex:
        return self._value

    def value(self, t: float = 0.0) -> complex:
        return self._value

    def conjugate(self):
        return Constant(self._value.conjugate())

    def __repr__(self):
        return "Constant(" + repr(self._value) + ")"


class PiecewiseConstant(BaseSignal):
    """A piecewise constant signal implemented as an array of samples."""

    def __init__(
        self,
        dt: float,
        samples: Union[Array, List],
        start_time: float = 0.0,
        duration: int = None,
        carrier_freq: float = 0.0,
        phase: float = 0.0,
        name: str = None,
    ):
        """Initialize a piecewise constant signal.

        Args:
            dt: The duration of each sample.
            samples: The array of samples.
            start_time: The time at which the signal starts.
            duration: The duration of the signal in samples.
            carrier_freq: The frequency of the carrier.
            phase: The phase of the carrier.
            name: name of the signal.
        """
        super().__init__(name)

        self._dt = dt

        if samples is not None:
            self._samples = Array(samples)
        else:
            self._samples = Array([0.0] * duration)

        self._start_time = start_time

        self._carrier_freq = Array(carrier_freq)
        self._phase = Array(phase)

    @property
    def duration(self) -> int:
        """
        Returns:
            duration: The duration of the signal in samples.
        """
        return len(self._samples)

    @property
    def dt(self) -> float:
        """
        Returns:
             dt: the duration of each sample.
        """
        return self._dt

    @property
    def samples(self) -> Array:
        """
        Returns:
            samples: the samples of the piecewise constant signal.
        """
        return Array(self._samples)

    @property
    def start_time(self) -> float:
        """
        Returns:
            start_time: The time at which the list of samples start.
        """
        return self._start_time

    def envelope_value(self, t: float = 0.0) -> complex:

        idx = Array((t - self._start_time) // self._dt, dtype=int)

        return Array(self._samples.data[idx.data])

    def value(self, t: float = 0.0) -> Array:
        """Return the value of the signal at time t."""
        arg = 1j * 2 * np.pi * self.carrier_freq * t + 1j * self.phase
        return self.envelope_value(t) * np.exp(arg)

    def conjugate(self):
        return PiecewiseConstant(
            dt=self._dt,
            samples=np.conjugate(self._samples),
            start_time=self._start_time,
            duration=self.duration,
            carrier_freq=-self.carrier_freq,
            phase=-self.phase,
        )

    def add_samples(self, start_sample: int, samples: List):
        """
        Appends samples to the pulse starting at start_sample.
        If start_sample is larger than the number of samples currently
        in the signal the signal is padded with zeros.

        Args:
            start_sample: number of the sample at which the new samples
                should be appended.
            samples: list of samples to append.

        Raises:
            QiskitError: if start_sample is invalid.
        """
        if start_sample < len(self._samples):
            raise QiskitError()

        while len(self._samples) < start_sample:
            self._samples = np.append(self._samples, [0.0])

        self._samples = np.append(self._samples, samples)


def signal_multiply(
    sig1: Union[BaseSignal, float, int, complex], sig2: Union[BaseSignal, float, int, complex]
) -> Signal:
    r"""Implements mathematical multiplication between two signals.
    Since a signal is represented by

    .. math::

        \Omega(t)*exp(2\pi i \nu t)

    multiplication of two signals implements

    .. math::

        \Omega_1(t)*\Omega_2(t)*exp(2\pi i (\nu_1+\nu_2) t)


    Args:
        sig1: A child of base signal or a constant.
        sig2: A child of base signal or a constant.

    Returns:
        signal: The type will depend on the given base class.
    """

    # ensure both arguments are signals
    if isinstance(sig1, (int, float, complex)):
        sig1 = Constant(sig1)

    if isinstance(sig2, (int, float, complex)):
        sig2 = Constant(sig2)

    return Signal(
        lambda t: sig1.envelope_value(t) * sig2.envelope_value(t),
        sig1.carrier_freq + sig2.carrier_freq,
        sig1.phase + sig2.phase,
    )


def signal_add(
    sig1: Union[BaseSignal, float, int, complex], sig2: Union[BaseSignal, float, int, complex]
) -> BaseSignal:
    r"""Implements mathematical addition between two signals.
    Since a signal is represented by
    .. math::

        \Omega(t)*exp(2\pi i \nu t)

    addition of two signals implements
    .. math::

        \Omega_1(t)*exp(2\pi i \nu_1 t) + \Omega_2(t)*exp(2\pi i \nu_2 t)


    Args:
        sig1: A child of base signal or a constant.
        sig2: A child of base signal or a constant.

    Returns:
        signal: The type will depend on the given base class.

    Raises:
        Exception: if signals cannot be added.
    """

    # ensure both arguments are signals
    if isinstance(sig1, (int, float, complex)):
        sig1 = Constant(sig1)
    if isinstance(sig2, (int, float, complex)):
        sig2 = Constant(sig2)

    avg_freq = (sig1.carrier_freq + sig2.carrier_freq) / 2

    return Signal(
        lambda t: (sig1.value(t) + sig2.value(t)) * np.exp(-2.0j * np.pi * t * avg_freq),
        avg_freq,
        0,
    )


class VectorSignal:
    """The vector version of the Signal class - the envelope is an array-valued
    function, and carrier_freqs is an array of carrier frequencies.

    In addition, a drift_array is set to correspond to the value of the
    VectorSignal when all "time-dependent terms" are off. E.g. if it is
    composed of a list of Signal objects, this corresponds to the output when
    all non-Constant signal objects are zero.
    """

    def __init__(
        self,
        envelope: Callable,
        carrier_freqs: Array,
        phases: Array,
        drift_array: Optional[Array] = None,
    ):
        """Initialize with vector-valued envelope, carrier frequencies for
        each entry, and a drift_array, which corresponds to the value of the
        signal when the "time-dependent terms" are "off".

        Args:
            envelope: function of a single float returning an array.
            carrier_freqs: list of carrier frequencies for each component of the
                           envelope.
            phases: list of carrier phases for each component of the envelope.
            drift_array: a default array meant to be the value of the envelope
                         when all "time-dependent terms" are off.
        """
        carrier_freqs = Array(carrier_freqs)
        phases = Array(phases)

        self.envelope = envelope
        self.carrier_freqs = carrier_freqs
        self.phases = phases

        self._im_angular_freqs = 1j * 2 * np.pi * carrier_freqs

        # if not supplied nothing is assumed, constant array is taken as all
        # zeros
        if drift_array is None:
            self.drift_array = Array(np.zeros(len(self.carrier_freqs)))
        else:
            self.drift_array = Array(drift_array)

    @classmethod
    def from_signal_list(cls, signal_list: List[BaseSignal]):
        """Instantiate from a list of Signal objects. The drift_array will
        correspond to the Constant objects.

        Args:
            signal_list: list of Signal objects.

        Returns:
            VectorSignal: that evaluates the signal list
        """

        # define the envelope as iteratively evaluating the envelopes
        def env_func(t):
            return Array([Array(sig.envelope_value(t)).data for sig in signal_list])

        # construct carrier frequency and phase list
        # if signal doesn't have a carrier, set to 0.
        carrier_freqs = Array(
            [Array(getattr(sig, "carrier_freq", 0.0)).data for sig in signal_list]
        )

        phases = Array([Array(getattr(sig, "phase", 0.0)).data for sig in signal_list])

        # construct drift_array
        drift_array = []
        for sig in signal_list:
            if isinstance(sig, Constant):
                drift_array.append(sig.value())
            else:
                drift_array.append(0.0)

        return cls(
            envelope=env_func,
            carrier_freqs=carrier_freqs,
            phases=phases,
            drift_array=Array(drift_array),
        )

    def envelope_value(self, t: float) -> Array:
        """Evaluate the envelope.

        Args:
            t: time

        Returns:
            Array: the signal envelope at time t
        """
        return self.envelope(t)

    def value(self, t: float) -> Array:
        """Evaluate the full value of the VectorSignal.

        Args:
            t (float): time

        Returns:
            Array: the value of the signal (including carrier frequencies)
                      at time t
        """
        carrier_val = np.exp(t * self._im_angular_freqs + 1.0j * self.phases)
        return self.envelope_value(t) * carrier_val

    def conjugate(self):
        """Return a new VectorSignal that is the complex conjugate of self.

        Returns:
            VectorSignal: the complex conjugate of self
        """
        return VectorSignal(
            lambda t: np.conjugate(self.envelope_value(t)),
            -self.carrier_freqs,
            -self.phases,
            np.conjugate(self.drift_array),
        )
