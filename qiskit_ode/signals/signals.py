# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name, unidiomatic-typecheck, super-init-not-called

"""
Module for representation of model coefficients.
"""

from typing import List, Callable, Union, Optional, Tuple
import itertools
import operator

import numpy as np
from matplotlib import pyplot as plt

from qiskit import QiskitError
from qiskit_ode.dispatch import Array


class Signal:
    r"""General signal class representing a function of the form:

    .. math::
        Re[f(t)e^{i (2 \pi \nu t + \phi)}]
                = Re[f(t)]\cos(2 \pi \nu t + \phi) - Im[f(t)]\sin(2 \pi \nu t + \phi),

    where

    - :math:`f(t)` is the envelope function.
    - :math:`\nu` is the carrier frequency.
    - :math:`\phi` is the phase.

    The envelope function can be complex-valued, and the frequency and phase must be real.

    Note: this class assumes that the envelope function is vectorized. If it isn't, it can
    be vectorized automatically by calling ``numpy.vectorize``, or, if using JAX, by
    calling ``jax.numpy.vectorize``.
    """

    def __init__(
        self,
        envelope: Union[Callable, complex, float, int],
        carrier_freq: float = 0.0,
        phase: float = 0.0,
        name: Optional[str] = None,
    ):
        """
        Initializes a signal given by an envelope and a carrier.

        Args:
            envelope: Envelope function of the signal, must be vectorized.
            carrier_freq: Frequency of the carrier.
            phase: The phase of the carrier.
            name: Name of the signal.
        """
        self._name = name

        if isinstance(envelope, (float, int)):
            envelope = Array(complex(envelope))

        if isinstance(envelope, Array):
            self._envelope = lambda t: envelope

        if callable(envelope):
            self._envelope = envelope

        # set carrier and phase
        self.carrier_freq = carrier_freq
        self.phase = phase

    @property
    def name(self) -> str:
        """Return the name of the signal."""
        return self._name

    @property
    def carrier_freq(self) -> Array:
        """The carrier frequency of the signal."""
        return self._carrier_freq

    @carrier_freq.setter
    def carrier_freq(self, carrier_freq: Union[float, list, Array]):
        """Carrier frequency setter. List handling is to support subclasses storing a
        list of frequencies."""
        if type(carrier_freq) == list:
            carrier_freq = [Array(entry).data for entry in carrier_freq]
        self._carrier_freq = Array(carrier_freq)
        self._carrier_arg = 1j * 2 * np.pi * self._carrier_freq

    @property
    def phase(self) -> Array:
        """The phase of the signal."""
        return self._phase

    @phase.setter
    def phase(self, phase: Union[float, list, Array]):
        """Phase setter. List handling is to support subclasses storing a
        list of phases."""
        if type(phase) == list:
            phase = [Array(entry).data for entry in phase]
        self._phase = Array(phase)
        self._phase_arg = 1j * self._phase

    def envelope(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        """Vectorized evaluation of the envelope at time t."""
        return self._envelope(t)

    def complex_value(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        """Vectorized evaluation of the complex value at time t."""
        arg = self._carrier_arg * t + self._phase_arg
        return self.envelope(t) * np.exp(arg)

    def __call__(self, t: Union[float, np.array, Array]) -> Array:
        """Vectorized evaluation of the signal at time t."""
        return np.real(self.complex_value(t))

    def __str__(self) -> str:
        """Return string representation."""
        if self.name is not None:
            return str(self.name)

        return "Signal(carrier_freq={freq}, phase={phase})".format(
            freq=str(self.carrier_freq), phase=str(self.phase)
        )

    def __add__(self, other: "Signal") -> "SignalSum":
        return signal_add(self, other)

    def __radd__(self, other: "Signal") -> "SignalSum":
        return self.__add__(other)

    def __mul__(self, other: "Signal") -> "SignalSum":
        return signal_multiply(self, other)

    def __rmul__(self, other: "Signal") -> "SignalSum":
        return self.__mul__(other)

    def __neg__(self) -> "SignalSum":
        return -1 * self

    def __sub__(self, other: "Signal") -> "SignalSum":
        return self + (-other)

    def __rsub__(self, other: "Signal") -> "SignalSum":
        return other + (-self)

    def conjugate(self):
        """Return a new signal obtained via complex conjugation of the envelope and phase."""

        def conj_env(t):
            return np.conjugate(self.envelope(t))

        return Signal(conj_env, self.carrier_freq, -self.phase)

    def draw(
        self,
        t0: float,
        tf: float,
        n: int,
        function: Optional[str] = "signal",
        axis: Optional[plt.axis] = None,
    ):
        """Plot the signal over an interval. The `function` arg specifies which function to
        plot:
            - `function == 'signal'` plots the full signal.
            - `function == 'envelope'` plots the complex envelope.
            - `function == 'complex_value'` plots the `complex_value`.

        Args:
            t0: Initial time.
            tf: Final time.
            n: Number of points to sample in interval.
            function: Which function to plot.
            axis: The axis to use for plotting.
        """

        t_vals = np.linspace(t0, tf, n)

        y_vals = None
        data_type = "real"
        if function == "signal":
            y_vals = self(t_vals)
        elif function == "envelope":
            y_vals = self.envelope(t_vals)
            data_type = "complex"
        elif function == "complex_value":
            y_vals = self.complex_value(t_vals)
            data_type = "complex"

        if data_type == "complex":
            if axis:
                axis.plot(t_vals, np.real(y_vals))
                axis.plot(t_vals, np.imag(y_vals))
            else:
                plt.plot(t_vals, np.real(y_vals))
                plt.plot(t_vals, np.imag(y_vals))
        else:
            if axis:
                axis.plot(t_vals, y_vals)
            else:
                plt.plot(t_vals, y_vals)


class Constant(Signal):
    """:class:`Signal` representing a constant value."""

    def __init__(self, value: complex, name: str = None):
        """Initialize a constant signal.

        Args:
            value: the constant.
            name: name of the constant.
        """
        self._name = name
        self._value = Array(value)
        self.phase = 0.0
        self.carrier_freq = 0.0

    def envelope(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        return self._value * np.ones(np.shape(t), dtype=complex)

    def conjugate(self):
        return Constant(np.conjugate(self._value))

    def __str__(self) -> str:
        if self.name is not None:
            return str(self.name)

        return "Constant({})".format(str(self._value))


class DiscreteSignal(Signal):
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
        self._name = name

        self._dt = dt

        if samples is not None:
            self._samples = Array(samples)
        else:
            self._samples = Array([0.0] * duration)

        self._start_time = start_time

        # initialize internally stored carrier/phase information
        self._carrier_freq = None
        self._phase = None
        self._carrier_arg = None
        self._phase_arg = None

        # set carrier and phase
        self.carrier_freq = carrier_freq
        self.phase = phase

    @classmethod
    def from_Signal(
        cls,
        signal: Signal,
        dt: float,
        n_samples: int,
        start_time: Optional[float] = 0.0,
        sample_carrier: Optional[bool] = False,
    ):
        """Constructs a ``DiscreteSignal`` object by sampling a ``Signal``.

        The optional argument ``sample_carrier`` controls whether or not to include the carrier
        in the sampling. I.e.:

            - If ``sample_carrier == False``, a ``DiscreteSignal`` is constructed with:
                - ``samples`` obtained by sampling ``signal.envelope``.
                - ``carrier_freq = signal.carrier_freq``.
                - ``phase = signal.phase``.

            - If ``sample_carrier == True``, a ``DiscreteSignal`` is constructed with:
                - ``samples`` obtained by sampling ``signal`` (as a ``callable``)
                - ``carrier_freq = 0``.
                - ``phase = signal.phase``.

        In either case, samples are obtained from the midpoint of each interval.

        Args:
            signal: Signal to sample.
            dt: Time increment to use.
            n_samples: Number of steps to resample with.
            start_time: Start time from which to resample.
            sample_carrier: Whether or not to include the carrier in the sampling.

        Returns:
            DiscreteSignal: The discretized ``Signal``.
        """

        times = start_time + (np.arange(n_samples) + 0.5) * dt
        freq = signal.carrier_freq
        samples = None

        if sample_carrier:
            freq = 0.0
            samples = signal(times)
        else:
            samples = signal.envelope(times)

        return DiscreteSignal(
            dt, samples, start_time=start_time, carrier_freq=freq, phase=signal.phase
        )

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

    def envelope(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        """Envelope. If ``t`` is before (resp. after) the start (resp. end) of the definition of
        the ``DiscreteSignal```, this will return the start value (resp. end value).
        """
        idx = np.clip(
            Array((t - self._start_time) // self._dt, dtype=int), 0, len(self._samples) - 1
        )
        return self._samples[idx.data]

    def complex_value(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        """Return the value of the signal at time t."""
        arg = self._carrier_arg * t + self._phase_arg
        return self.envelope(t) * np.exp(arg)

    def conjugate(self):
        return DiscreteSignal(
            dt=self._dt,
            samples=np.conjugate(self._samples),
            start_time=self._start_time,
            duration=self.duration,
            carrier_freq=self.carrier_freq,
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

        if len(self._samples) < start_sample:
            self._samples = np.append(
                self._samples, np.zeros(start_sample - len(self._samples), dtype=complex)
            )

        self._samples = np.append(self._samples, samples)

    def __str__(self) -> str:
        """Return string representation."""
        if self.name is not None:
            return str(self.name)

        return "DiscreteSignal(dt={dt}, carrier_freq={freq}, phase={phase})".format(
            dt=self.dt, freq=str(self.carrier_freq), phase=str(self.phase)
        )


class SignalSum(Signal):
    r"""Represents a sum of ``Signal`` objects:

    .. math::
        s_1(t) + \dots + s_k(t)

    For basic evaluation, this class behaves in the same manner as ``Signal``:
    - ``__call__`` evaluates the sum.
    - ``complex_value`` evaluates the sum of the complex values of the individual summands.

    Attributes ``carrier_freq`` and ``phase`` here correspond to an ``Array`` of
    frequencies/phases for each term in the sum, and the ``envelope`` method returns an
    ``Array`` of the envelopes for each summand.

    Internally, the signals are stored as a list in the ``components`` attribute, which can
    be accessed via direct subscripting of the object.
    """

    def __init__(self, *signals, name: Optional[str] = None):
        """Initialize with a list of Signal objects through ``args``.

        Args:
            signals: ``Signal`` subclass objects.
            name: Name of the sum.

        Raises:
            QiskitError: If ``signals`` are not subclasses of ``Signal``.
        """
        self._name = name

        self.components = []
        for sig in signals:
            if isinstance(sig, SignalSum):
                self.components += sig.components
            elif isinstance(sig, Signal):
                self.components.append(sig)
            else:
                raise QiskitError(
                    "Components of a SignalSum must be instances of a Signal subclass."
                )

        self._envelopes = [sig.envelope for sig in self.components]

        carrier_freqs = []
        for sig in self.components:
            carrier_freqs.append(sig.carrier_freq)

        phases = []
        for sig in self.components:
            phases.append(sig.phase)

        # set carrier and phase
        self.carrier_freq = carrier_freqs
        self.phase = phases

    def envelope(self, t: Union[float, np.array, Array]) -> Array:
        """Return an array of the envelope values of each component."""
        # to do: jax version
        # not sure what the right way to do this is, here we actually need
        # to get it to use np/jnp
        return np.moveaxis(Array([env(t) for env in self._envelopes]), 0, -1)

    def complex_value(self, t: Union[float, np.array, Array]) -> Array:
        """Return the sum of the complex values of each component."""
        exp_phases = np.exp(np.expand_dims(Array(t), -1) * self._carrier_arg + self._phase_arg)
        return np.sum(self.envelope(t) * exp_phases, axis=-1)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, idx: Union[int, List, np.array, slice]) -> Signal:
        """Enables numpy-style subscripting, as if this class were a 1d array."""

        if type(idx) == np.ndarray and idx.ndim > 0:
            idx = list(idx)

        sublist = None
        # get a list of the subcomponents
        if type(idx) == list:
            # handle lists
            sublist = operator.itemgetter(*idx)(self.components)

            # output will either be a single signal or a tuple of Signals
            # convert to list if tuple
            if type(sublist) == tuple:
                sublist = list(sublist)
        else:
            # handle slices or singletons
            sublist = operator.itemgetter(idx)(self.components)

        # sublist should either be a single Signal, or a list of Signals
        if isinstance(sublist, Signal):
            return sublist
        else:
            return SignalSum(*sublist)

    def __str__(self):
        if self.name is not None:
            return str(self.name)

        if len(self) == 0:
            return "SignalSum()"

        default_str = str(self[0])
        for sig in self.components[1:]:
            default_str += " + {}".format(str(sig))

        return default_str

    def flatten(self) -> Signal:
        """Merge into a single ``Signal``. The output frequency is given by the
        average.
        """

        if len(self) == 0:
            return Constant(0.0)
        elif len(self) == 1:
            return self.components[0]

        ave_freq = np.sum(self.carrier_freq) / len(self)
        shifted_arg = self._carrier_arg - (1j * 2 * np.pi * ave_freq)

        def merged_env(t):
            exp_phases = np.exp(np.expand_dims(Array(t), -1) * shifted_arg + self._phase_arg)
            return np.sum(self.envelope(t) * exp_phases, axis=-1)

        return Signal(envelope=merged_env, carrier_freq=ave_freq, name=str(self))


class DiscreteSignalSum(DiscreteSignal, SignalSum):
    """Represents a sum of discretized signals, all with the same
    time parameters: dt, number of samples, and start time.
    """

    def __init__(
        self,
        dt: float,
        samples: Union[Array, List],
        start_time: float = 0.0,
        duration: int = None,
        carrier_freqs: float = None,
        phases: float = None,
        name: str = None,
    ):
        """Samples array has 0th axis corresponding to a signal, 1st axis corresponding to samples
        for each signal."""
        self._name = name
        self._dt = dt
        self._samples = Array(samples)
        self._start_time = start_time

        if carrier_freqs is None:
            carrier_freqs = np.zeros(len(samples), dtype=float)

        if phases is None:
            phases = np.zeros(len(samples), dtype=float)

        # construct individual components so they can be accessed as in SignalSum
        components = []
        for sample_row, freq, phase in zip(self.samples.transpose(), carrier_freqs, phases):
            components.append(
                DiscreteSignal(
                    dt=self.dt,
                    samples=sample_row,
                    start_time=self.start_time,
                    duration=self.duration,
                    carrier_freq=freq,
                    phase=phase,
                )
            )

        self.components = components
        self.carrier_freq = carrier_freqs
        self.phase = phases

    @classmethod
    def from_SignalSum(
        cls,
        signal_sum: SignalSum,
        dt: float,
        n_samples: int,
        start_time: Optional[float] = 0.0,
        sample_carrier: Optional[bool] = False,
    ):
        """Constructs a ``DiscreteSignalSum`` object by sampling a ``SignalSum``.

        The optional argument ``sample_carrier`` controls whether or not to include the carrier
        in the sampling. I.e.:

            - If ``sample_carrier == False``, a ``DiscreteSignalSum`` is constructed with:
                - ``samples`` obtained by sampling ``signal_sum.envelope``.
                - ``carrier_freq = signal_sum.carrier_freq``.
                - ``phase = signal_sum.phase``.

            - If ``sample_carrier == True``, a ``DiscreteSignal`` is constructed with:
                - ``samples`` obtained by sampling ``signal_sum`` (as a ``callable``)
                - ``carrier_freq = 0``.
                - ``phase = signal_sum.phase``.

        In either case, samples are obtained from the midpoint of each interval.

        Args:
            signal: Signal to sample.
            dt: Time increment to use.
            n_samples: Number of steps to resample with.
            start_time: Start time from which to resample.
            sample_carrier: Whether or not to include the carrier in the sampling.

        Returns:
            DiscreteSignalSum: The discretized ``SignalSum``.
        """

        times = start_time + (np.arange(n_samples) + 0.5) * dt

        freq = signal_sum.carrier_freq
        samples = None

        if sample_carrier:
            freq = 0.0 * freq
            exp_phases = np.exp(np.expand_dims(Array(times), -1) * signal_sum._carrier_arg)
            samples = signal_sum.envelope(times) * exp_phases
        else:
            samples = signal_sum.envelope(times)

        return DiscreteSignalSum(
            dt, samples, start_time=start_time, carrier_freqs=freq, phases=signal_sum.phase
        )

    def complex_value(self, t: Union[float, np.array, Array]) -> Array:
        exp_phases = np.exp(np.expand_dims(Array(t), -1) * self._carrier_arg + self._phase_arg)
        return np.sum(self.envelope(t) * exp_phases, axis=-1)

    def __str__(self):
        if self.name is not None:
            return str(self.name)

        if len(self) == 0:
            return "DiscreteSignalSignalSum()"

        default_str = str(self[0])
        for sig in self.components[1:]:
            default_str += " + {}".format(str(sig))

        return default_str

    def __getitem__(self, idx: Union[int, List, np.array, slice]) -> Signal:
        """Enables numpy-style subscripting, as if this class were a 1d array."""

        samples = self.samples[idx]
        carrier_freqs = self.carrier_freq[idx]
        phases = self.phase[idx]

        if samples.ndim == 1:
            samples = Array([samples])

        if carrier_freqs.ndim == 0:
            carrier_freqs = Array([carrier_freqs])

        if phases.ndim == 0:
            phases = Array([phases])

        if len(samples) == 1:
            return DiscreteSignal(
                dt=self.dt,
                samples=samples[0],
                start_time=self.start_time,
                carrier_freq=carrier_freqs[0],
                phase=phases[0],
            )

        return DiscreteSignalSum(
            dt=self.dt,
            samples=samples,
            start_time=self.start_time,
            carrier_freqs=carrier_freqs,
            phases=phases,
        )


class SignalList:
    """A list of ``Signal``s, with functionality for simultaneous evaluation."""

    def __init__(self, signal_list: List[Signal]):
        self.components = signal_list

    def complex_value(self, t: Union[float, np.array, Array]) -> Array:
        """Vectorized evaluation of complex value of components."""
        return np.moveaxis(
            Array([Array(sig.complex_value(t)).data for sig in self.components]), 0, -1
        )

    def __call__(self, t: Union[float, np.array, Array]) -> Array:
        """Vectorized evaluation of all components."""
        return np.moveaxis(Array([Array(sig(t)).data for sig in self.components]), 0, -1)

    def flatten(self) -> "SignalList":
        """Return a ``SignalList`` with each component flattened."""
        flattened_list = []
        for sig in self.components:
            if isinstance(sig, SignalSum):
                flattened_list.append(sig.flatten())
            else:
                flattened_list.append(sig)

        return SignalList(flattened_list)

    @property
    def drift(self) -> Array:
        """Return the 'drift' Array, i.e. the constant part of the ``SignalList``."""

        drift_array = []

        for sig_entry in self.components:
            val = 0.0

            if not isinstance(sig_entry, SignalSum):
                sig_entry = SignalSum(sig_entry)

            for term in sig_entry:
                if isinstance(term, Constant):
                    val += term._value.data

            drift_array.append(val)

        return Array(drift_array)

    def __getitem__(self, idx: Union[int, List, np.array, slice]) -> Signal:
        """Enables numpy-style subscripting, as if this class were a 1d array."""

        if type(idx) == np.ndarray and idx.ndim > 0:
            idx = list(idx)

        sublist = None

        # get a list of the subcomponents
        if type(idx) == list:
            # handle lists
            sublist = operator.itemgetter(*idx)(self.components)

            # output will either be a single signal or a tuple of Signals
            # convert to list if tuple
            if type(sublist) == tuple:
                sublist = list(sublist)
        else:
            # handle slices or singletons
            sublist = operator.itemgetter(idx)(self.components)

        # sublist should either be a single Signal, or a list of Signals
        if isinstance(sublist, Signal):
            return sublist
        else:
            return SignalList(*sublist)

    def __len__(self):
        return len(self.components)


def signal_add(sig1: Signal, sig2: Signal) -> SignalSum:
    """Add two signals."""

    # generic routine
    # convert to SignalSum instances
    try:
        wrapped_sig1 = to_SignalSum(sig1)
        wrapped_sig2 = to_SignalSum(sig2)
    except QiskitError as e:
        raise QiskitError("Only a number or a Signal instance can be added to a Signal.") from e

    sig_sum = SignalSum(*(wrapped_sig1.components + wrapped_sig2.components))

    # if they were originally DiscreteSignalSum objects with compatible structure,
    # convert back
    if isinstance(sig1, DiscreteSignal) and isinstance(sig2, DiscreteSignal):
        if (
            sig1.dt == sig2.dt
            and sig1.start_time == sig2.start_time
            and sig1.duration == sig2.duration
        ):
            sig_sum = DiscreteSignalSum.from_SignalSum(
                sig_sum, dt=sig2.dt, start_time=sig2.start_time, n_samples=sig2.duration
            )

    return sig_sum


def signal_multiply(sig1: Signal, sig2: Signal) -> SignalSum:
    r"""Multiply two ``Signal``s. For a pair of elementary (non-``SignalSum``) ``Signal``s,
    expands the product of two signals into a ``SignalSum`` via the formula:

    .. math::
        Re[f(t)e^{i(2 \pi \nu t + \phi)}] \times Re[g(t)e^{i(2 \pi \omega t + \psi)}]
         = Re[\frac{1}{2} f(t)g(t)e^{i(2\pi (\omega + \nu)t + (\phi + \psi))} ]
          + Re[\frac{1}{2} f(t)\overline{g(t)}e^{i(2\pi (\omega - \nu)t + (\phi - \psi))} ]

    If either (or both) of ``sig1`` or ``sig2`` are ``SignalSum``s, the multiplication is
    distributed over addition.
    """

    # convert to SignalSum instances
    try:
        wrapped_sig1 = to_SignalSum(sig1)
        wrapped_sig2 = to_SignalSum(sig2)
    except QiskitError as e:
        raise QiskitError("Only a number or a Signal instance can multiply a Signal.") from e

    # initialize to empty sum
    product = SignalSum()

    # loop through every pair of components and multiply
    for comp1, comp2 in itertools.product(wrapped_sig1.components, wrapped_sig2.components):
        product += base_signal_multiply(comp1, comp2)

    # if they were originally DiscreteSignalSum objects with compatible structure,
    # convert back
    if isinstance(sig1, DiscreteSignalSum) and isinstance(sig2, DiscreteSignalSum):
        if (
            sig1.dt == sig2.dt
            and sig1.start_time == sig2.start_time
            and sig1.duration == sig2.duration
        ):
            product = DiscreteSignalSum.from_SignalSum(
                product, dt=sig1.dt, start_time=sig1.start_time, n_samples=sig1.duration
            )

    return product


def base_signal_multiply(sig1: Signal, sig2: Signal) -> Signal:
    r"""Utility function for multiplying two elementary (non ``SignalSum``) signals.
    This function assumes ``sig1`` and ``sig2`` are legitimate instances of ``Signal``
    subclasses.

    Special cases:

        - Multiplication of two ``Constant``s returns a ``Constant``.
        - Multiplication of a ``Constant`` and a ``DiscreteSignal`` returns a ``DiscreteSignal``.
        - If two ``DiscreteSignal``s have compatible parameters, the resulting signals are
        ``DiscreteSignal``, with the multiplication being implemented by array multiplication of
        the samples.
        - Lastly, if no special rules apply, the two ``Signal``s are multiplied generically via
        multiplication of the envelopes as functions.

    Args:
        sig1: First signal.
        sig2: Second signal.

    Returns:
        SignalSum: Representing the RHS of the formula when two Signals are multiplied.
    """

    # ensure signals are ordered from most to least specialized
    sig1, sig2 = sort_signals(sig1, sig2)

    if type(sig1) is Constant and type(sig2) is Constant:
        return Constant(sig1(0.0) * sig2(0.0))
    elif type(sig1) is Constant and type(sig2) is DiscreteSignal:
        # multiply the samples by the constant
        return DiscreteSignal(
            dt=sig2.dt,
            samples=sig1(0.0) * sig2.samples,
            start_time=sig2.start_time,
            duration=sig2.duration,
            carrier_freq=sig2.carrier_freq,
            phase=sig2.phase,
        )
    elif type(sig1) is Constant and type(sig2) is Signal:
        const = sig1(0.0)

        def new_env(t):
            return const * sig2.envelope(t)

        return Signal(envelope=new_env, carrier_freq=sig2.carrier_freq, phase=sig2.phase)
    elif type(sig1) is DiscreteSignal and type(sig2) is DiscreteSignal:
        # verify compatible parameters before applying special rule
        if (
            sig1.start_time == sig2.start_time
            and sig1.dt == sig2.dt
            and len(sig1.samples) == len(sig2.samples)
        ):
            pwc1 = DiscreteSignal(
                dt=sig2.dt,
                samples=0.5 * sig1.samples * sig2.samples,
                start_time=sig2.start_time,
                duration=sig2.duration,
                carrier_freq=sig1.carrier_freq + sig2.carrier_freq,
                phase=sig1.phase + sig2.phase,
            )
            pwc2 = DiscreteSignal(
                dt=sig2.dt,
                samples=0.5 * sig1.samples * np.conjugate(sig2.samples),
                start_time=sig2.start_time,
                duration=sig2.duration,
                carrier_freq=sig1.carrier_freq - sig2.carrier_freq,
                phase=sig1.phase - sig2.phase,
            )
            return pwc1 + pwc2

    # if no special cases apply, implement generic rule
    def new_env1(t):
        return 0.5 * sig1.envelope(t) * sig2.envelope(t)

    def new_env2(t):
        return 0.5 * sig1.envelope(t) * np.conjugate(sig2.envelope(t))

    prod1 = Signal(
        envelope=new_env1,
        carrier_freq=sig1.carrier_freq + sig2.carrier_freq,
        phase=sig1.phase + sig2.phase,
    )
    prod2 = Signal(
        envelope=new_env2,
        carrier_freq=sig1.carrier_freq - sig2.carrier_freq,
        phase=sig1.phase - sig2.phase,
    )
    return prod1 + prod2


def sort_signals(sig1: Signal, sig2: Signal) -> Tuple[Signal, Signal]:
    """Utility function for ordering a pair of ``Signal``s according to the partial order:
    ``sig1 <= sig2`` if and only if:
        - ``type(sig1)`` precedes ``type(sig2)`` in the list
        ``[Constant, DiscreteSignal, Signal, SignalSum, DiscreteSignalSum]``.
    """
    if isinstance(sig1, Constant):
        pass
    elif isinstance(sig2, Constant):
        sig1, sig2 = sig2, sig1
    elif isinstance(sig1, DiscreteSignal):
        pass
    elif isinstance(sig2, DiscreteSignal):
        sig2, sig1 = sig1, sig2
    elif isinstance(sig1, Signal):
        pass
    elif isinstance(sig2, Signal):
        sig2, sig1 = sig1, sig2
    elif isinstance(sig1, SignalSum):
        pass
    elif isinstance(sig2, SignalSum):
        sig2, sig1 = sig1, sig2

    return sig1, sig2


def to_SignalSum(sig: Union[int, float, complex, Array, Signal]) -> SignalSum:
    """Convert the input to a SignalSum according to:

    - If it is already a SignalSum, do nothing.
    - If it is a Signal but not a SignalSum, wrap in a SignalSum.
    - If it is a number, wrap in Constant in a SignalSum.
    - Otherwise, raise an error.

    Args:
        sig: A SignalSum compatible input.

    Returns:
        SignalSum

    Raises:
        QiskitError: If the input type is incompatible with SignalSum.
    """

    if isinstance(sig, (int, float, complex)) or (isinstance(sig, Array) and sig.ndim == 0):
        sig = Constant(sig)

    if isinstance(sig, Signal) and not isinstance(sig, SignalSum):
        sig = SignalSum(sig)

    if not isinstance(sig, SignalSum):
        raise QiskitError("Input type incompatible with SignalSum.")

    return sig
