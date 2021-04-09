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
# pylint: disable=invalid-name

"""
Module for representation of model coefficients.
"""

from abc import ABC, abstractmethod
from typing import List, Callable, Union, Optional, Tuple
import itertools

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
    """

    def __init__(
        self,
        envelope: Union[Callable, complex, float, int],
        carrier_freq: float = 0.0,
        phase: float = 0.0,
        name: Optional[str] = None,
        vectorize_envelope: Optional[bool] = False,
    ):
        """
        Initializes a signal given by an envelope and a carrier.

        Args:
            envelope: Envelope function of the signal, must be vectorized.
            carrier_freq: Frequency of the carrier.
            phase: The phase of the carrier.
            name: Name of the signal.
            vectorize_envelope: Whether or not automatically vectorize the envelope function.
        """
        self.name = name

        if isinstance(envelope, (float, int)):
            envelope = Array(complex(envelope))

        if isinstance(envelope, Array):
            self._envelope = lambda t: envelope

        if isinstance(envelope, Callable):
            # to do: put jax version
            if vectorize_envelope:
                envelope = np.vectorize(envelope)

            self._envelope = envelope

        # initialize internally stored carrier/phase information
        self._carrier_freq = None
        self._phase = None
        self._carrier_arg = None
        self._phase_arg = None

        # set carrier and phase
        self.carrier_freq = carrier_freq
        self.phase = phase

    @property
    def carrier_freq(self) -> Array:
        return self._carrier_freq

    @carrier_freq.setter
    def carrier_freq(self, carrier_freq: float):
        self._carrier_freq = Array(carrier_freq)
        self._carrier_arg = 1j * 2 * np.pi * self._carrier_freq

    @property
    def phase(self) -> Array:
        return self._phase

    @phase.setter
    def phase(self, phase: float):
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

        return 'Signal(carrier_freq={freq}, phase={phase})'.format(freq=str(self.carrier_freq), phase=str(self.phase))

    def __add__(self, other: 'Signal') -> 'SignalSum':
        return signal_add(self, other)

    def __radd__(self, other: 'Signal') -> 'SignalSum':
        return self.__add__(other)

    def __mul__(self, other: 'Signal') -> 'SignalSum':
        return signal_multiply(self, other)

    def __rmul__(self, other: 'Signal') -> 'SignalSum':
        return self.__mul__(other)

    def __neg__(self) -> 'SignalSum':
        return -1 * self

    def __sub__(self, other: 'Signal') -> 'SignalSum':
        return self + (-other)

    def __rsub__(self, other: 'Signal') -> 'SignalSum':
        return other + (-self)

    def conjugate(self):
        """Return a new signal obtained via complex conjugation of the envelope and phase."""
        def conj_env(t):
            return np.conjugate(self.envelope(t))

        return Signal(conj_env, self.carrier_freq, -self.phase)

    def plot(self, t0: float, tf: float, n: int, axis=None):
        """Plot the signal over an interval.

        Args:
            t0: Initial time.
            tf: Final time.
            n: Number of points to sample in interval.
            axis: The axis to use for plotting.
        """
        t_vals = np.linspace(t0, tf, n)
        sig_vals = self(t_vals)

        if axis:
            axis.plot(t_vals, sig_vals)
        else:
            plt.plot(t_vals, sig_vals)

    def plot_envelope(self, t0: float, tf: float, n: int, axis=None):
        """Plot the signal over an interval.

        Args:
            t0: Initial time.
            tf: Final time.
            n: Number of points to sample in interval.
            axis: The axis to use for plotting.
        """
        t_vals = np.linspace(t0, tf, n)
        env_vals = self.envelope(t_vals)

        if axis:
            axis.plot(t_vals, np.real(env_vals))
            axis.plot(t_vals, np.imag(env_vals))
        else:
            plt.plot(t_vals, np.real(env_vals))
            plt.plot(t_vals, np.imag(env_vals))

    def plot_complex_value(self, t0: float, tf: float, n: int, axis=None):
        """Plot the complex value over an interval.

        Args:
            t0: Initial time.
            tf: Final time.
            n: Number of points to sample in interval.
            axis: The axis to use for plotting.
        """
        t_vals = np.linspace(t0, tf, n)
        sig_vals = self.complex_value(t_vals)

        if axis:
            axis.plot(t_vals, np.real(sig_vals))
            axis.plot(t_vals, np.imag(sig_vals))
        else:
            plt.plot(t_vals, np.real(sig_vals))
            plt.plot(t_vals, np.imag(sig_vals))


class Constant(Signal):
    """:class:`Signal` representing a constant value."""

    def __init__(self, value: complex, name: str = None):
        """Initialize a constant signal.

        Args:
            value: the constant.
            name: name of the constant.
        """
        self.name = name
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

        return 'Constant({})'.format(str(self._value))


class PiecewiseConstant(Signal):
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
        self.name = name

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
        idx = Array((t - self._start_time) // self._dt, dtype=int)
        return Array(self._samples.data[idx.data])

    def complex_value(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        """Return the value of the signal at time t."""
        arg = self._carrier_arg * t + self._phase_arg
        return self.envelope(t) * np.exp(arg)

    def conjugate(self):
        return PiecewiseConstant(
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

        return 'PiecewiseConstant(dt={dt}, carrier_freq={freq}, phase={phase})'.format(dt=self.dt, freq=str(self.carrier_freq), phase=str(self.phase))



class SignalSum(Signal):
    """Class representing a sum of ``Signal`` objects."""

    def __init__(self, *args, name: Optional[str] = None):
        """Initialize with a list of Signal objects through ``args``.

        Args:
            args: ``Signal`` subclass objects.
            name: Name of the sum.
        """
        self.name = name

        self.components = []
        for sig in args:
            if isinstance(sig, SignalSum):
                self.components += sig.components
            elif isinstance(sig, Signal):
                self.components.append(sig)
            else:
                raise QiskitError('Components of a SignalSum must be instances of a Signal subclass.')

        self._envelopes = [sig.envelope for sig in self.components]

        # initialize internally stored carrier/phase information
        self._carrier_freq = None
        self._phase = None
        self._carrier_arg = None
        self._phase_arg = None

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
        """Evaluate envelopes of each component. For vectorized operation,
        last axis indexes the envelope, and all proceeding axes are the
        same as the ``t`` arg.
        """
        # to do: jax version
        # not sure what the right way to do this is, here we actually need
        # to get it to use np/jnp
        return np.moveaxis(Array([env(t) for env in self._envelopes]), 0, -1)

    def complex_value(self, t: Union[float, np.array, Array]) -> Array:
        exp_phases = np.exp(np.expand_dims(Array(t), -1) * self._carrier_arg + self._phase_arg)
        return np.sum(self.envelope(t) * exp_phases, axis=-1)

    def __len__(self):
        return len(self.components)

    def __getitem__(self, idx):
        return self.components[idx]

    def __str__(self):
        if self.name is not None:
            return str(self.name)

        if len(self) == 0:
            return 'SignalSum()'

        default_str = str(self[0])
        for sig in self.components[1:]:
            default_str += ' + {}'.format(str(sig))

        return default_str

    def simplify(self, combine_types: Optional[bool] = False) -> 'SignalSum':
        """Combine similar terms in the sum. If ``combine_types == False`` terms will only
        be combined if their types can be preserved. If
        ``combine_types == True``, all terms with compatible frequencies will combined,
        regardless of type.
        """
        if len(self) == 0:
            return SignalSum()
        elif len(self) == 1:
            return SignalSum(self[0])

        new_sigs = [self[0]]
        for sig1 in self.components[1:]:
            merge = False
            for idx, sig2 in enumerate(new_sigs):
                # iterate through cases deciding if to merge
                if sig1.carrier_freq == sig2.carrier_freq or -sig1.carrier_freq == sig2.carrier_freq:
                    if combine_types:
                        merge = True

                    # if not combine types, verify same type and further compatibility
                    # requirements before merging
                    elif type(sig1) == type(sig2):
                        if type(sig1) == PiecewiseConstant:
                            # for piecewise constants, only merge if compatible parameters
                            if sig1.start_time == sig2.start_time and sig1.dt == sig2.dt and len(sig1.samples) == len(sig2.samples):
                                merge = True
                        else:
                            merge = True

                # if merge
                if merge:
                    new_sigs[idx] = merge_freq_compatible_pair(sig1, sig2)
                    break

            # if it wasn't merged, add it to the list as a new signal
            if not merge:
                new_sigs.append(sig1)

        return SignalSum(*new_sigs, name=self.name)

    def collapse(self) -> Signal:
        """Merge into a single ``Signal``. The output frequency is given by the
        average.
        """

        if len(self) == 0:
            return Constant(0.)
        elif len(self) == 1:
            return self.components[0]

        ave_freq = np.sum(self.carrier_freq) / len(self)
        shifted_arg = self._carrier_arg - (1j * 2 * np.pi * ave_freq)

        def merged_env(t):
            exp_phases = np.exp(np.expand_dims(Array(t), -1) * shifted_arg + self._phase_arg)
            return np.sum(self.envelope(t) * exp_phases, axis=-1)

        return Signal(envelope=merged_env, carrier_freq=ave_freq, name=str(self))


class PiecewiseConstantSignalSum(PiecewiseConstant, SignalSum):
    """Represents a sum of piecewise constant signals, all with the same
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
        """samples array assumed to have 0th axis corresponding to a signal, 1st axis the samples
        of that signal."""
        self.name=name
        self._dt = dt
        self._samples = Array(samples)
        self._start_time = start_time

        if carrier_freqs is None:
            carrier_freqs = np.zeros(len(samples), dtype=float)

        if phases is None:
            phases = np.zeros(len(samples), dtype=float)

        # construct individual components so they can be accessed as in SignalSum
        components = []
        for sample_row, freq, phase in zip(self.samples, carrier_freqs, phases):
            components.append(PiecewiseConstant(
                                                dt=self.dt,
                                                samples=sample_row,
                                                start_time=self.start_time,
                                                duration=self.duration,
                                                carrier_freq=freq,
                                                phase=phase,
                                                )
                                )

        self.components = components

        # initialize internally stored carrier/phase information
        self._carrier_freq = None
        self._phase = None
        self._carrier_arg = None
        self._phase_arg = None

        self.carrier_freq = carrier_freqs
        self.phase = phases


    @property
    def duration(self) -> int:
        """
        Returns:
            duration: The duration of the signal in samples.
        """
        return self._samples.shape[1]

    def envelope(self, t: Union[float, np.array, Array]) -> Array:
        """Evaluate envelopes of each component. For vectorized operation,
        last axis indexes the envelope, and all proceeding axes are the
        same as the ``t`` arg.
        """
        # to do: jax version
        # not sure what the right way to do this is, here we actually need
        # to get it to use np/jnp
        idx = Array((t - self.start_time) // self.dt, dtype=int)

        return np.moveaxis(Array(self.samples[:, idx]), 0, -1)

    def complex_value(self, t: Union[float, np.array, Array]) -> Array:
        exp_phases = np.exp(np.expand_dims(Array(t), -1) * self._carrier_arg + self._phase_arg)
        return np.sum(self.envelope(t) * exp_phases, axis=-1)


def merge_freq_compatible_pair(sig1: Signal, sig2: Signal) -> Signal:
    """Helper function for merging two non-``SignalSum`` ``Signal``s that satisfy
    ``sig1.carrier_freq == sig2.carrier_freq or sig1.carrier_freq == - sig2.carrier_freq``,
    with the ``opposite`` argument indicating which of the two cases holds.

    The returned ``Signal`` has the same carrier frequency as ``sig1`` in either case.
    """

    sig1, sig2 = sort_signals(sig1, sig2)

    equal_sign = sig1.carrier_freq == sig2.carrier_freq

    if type(sig1) == Constant and type(sig2) == Constant:
        return Constant(sig1(0.0) + sig2(0.0))
    elif type(sig1) == Constant and type(sig2) == PiecewiseConstant:
        # return a PiecewiseConstant shifted by sig1
        return PiecewiseConstant(
            dt=sig2.dt,
            samples=sig1(0.0) * np.exp(-1j * sig2.phase) + sig2.samples,
            start_time=sig2.start_time,
            duration=sig2.duration,
            carrier_freq=0.0,
            phase=sig2.phase,
        )
    elif type(sig1) == Constant and type(sig2) == Signal:
        const = sig1(0.0) * np.exp(-1j * sig2.phase)
        def new_env(t):
            return const + sig2.envelope(t)

        return Signal(envelope=new_env,
                      carrier_freq=0.0,
                      phase=sig2.phase)
    elif type(sig1) == PiecewiseConstant and type(sig2) == PiecewiseConstant:
        if sig1.start_time == sig2.start_time and sig1.dt == sig2.dt and len(sig1.samples) == len(sig2.samples):
            if not equal_sign:
                sig2 = sig2.conjugate()

            new_samples = (sig1.samples * np.exp(1j * 0.5 * (sig1.phase - sig2.phase)) +
                           sig2.samples * np.exp(1j * 0.5 * (sig2.phase - sig1.phase)))
            new_phase = 0.5 * (sig1.phase + sig2.phase)

            return PiecewiseConstant(
                dt=sig1.dt,
                samples=new_samples,
                start_time=sig1.start_time,
                duration=sig1.duration,
                carrier_freq=sig1.carrier_freq,
                phase=new_phase,
            )

    # if no special cases apply, do generic signal addition
    # average phases
    new_phase = 0.5 * (sig1.phase + sig2.phase)
    sig1_phase_shift = np.exp(1j * 0.5 * (sig1.phase - sig2.phase))
    sig2_phase_shift = np.exp(1j * 0.5 * (sig2.phase - sig1.phase))

    def new_env(t):
        return sig1.envelope(t) * sig1_phase_shift + sig2.envelope(t) * sig2_phase_shift

    return Signal(envelope=new_env,
                  carrier_freq=sig1.carrier_freq,
                  phase=new_phase)

def signal_add(sig1: Signal, sig2: Signal) -> SignalSum:
    """Add two signals."""

    # convert to SignalSum instances
    try:
        sig1 = to_SignalSum(sig1)
        sig2 = to_SignalSum(sig2)
    except:
        raise QiskitError('Only a number or a Signal instance can be added to a Signal.')

    # join component lists and define new sum
    return SignalSum(*(sig1.components + sig2.components))


def signal_multiply(sig1: Signal, sig2: Signal) -> SignalSum:
    """Multiply two signals."""

    # convert to SignalSum instances
    try:
        sig1 = to_SignalSum(sig1)
        sig2 = to_SignalSum(sig2)
    except:
        raise QiskitError('Only a number or a Signal instance can multiply a Signal.')

    # initialize to empty sum
    product = SignalSum()

    # loop through every pair of components and multiply
    for comp1, comp2 in itertools.product(sig1.components, sig2.components):
        product += base_signal_multiply(comp1, comp2)

    return product

def base_signal_multiply(sig1: Signal, sig2: Signal) -> Signal:
    r"""Utility function for multiplying two elementary (non ``SignalSum``) signals.
    This function assumes ``sig1`` and ``sig2`` are legitimate instances of ``Signal``
    subclasses.

    Mathematically, this implements the multiplication:

    .. math::
        Re[f(t)e^{i(2 \pi \nu t + \phi)}] \times Re[g(t)e^{i(2 \pi \omega t + \psi)}]
         = Re[\frac{1}{2} f(t)g(t)e^{i(2\pi (\omega + \nu)t + (\phi + \psi))} ]
          + Re[\frac{1}{2} f(t)\overline{g(t)}e^{i(2\pi (\omega - \nu)t + (\phi - \psi))} ]

    i.e. a ``SignalSum`` representing the RHS is returned. Special cases in which the above
    formula can be simplified are handled:

    - Multiplication of two ``Constant``s returns a ``Constant``.
    - Multiplication of a ``Constant`` and a ``PiecewiseConstant`` returns a ``PiecewiseConstant``.
    - If two ``PiecewiseConstant``s have compatible parameters, the resulting signals are
    ``PiecewiseConstant``, with the multiplication being implemented by array multiplication of
    the samples.
    - Beyond this, if one of the two signals has zero frequency, then the RHS is simplified to
    only return a single ``Signal`` (in both the compatible ``PiecewiseConstant`` and
    general ``Signal`` cases.)
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

    if isinstance(sig1, Constant) and isinstance(sig2, Constant):
        return Constant(sig1(0.0) * sig2(0.0))
    elif isinstance(sig1, Constant) and isinstance(sig2, PiecewiseConstant):
        # multiply the samples by the constant
        return PiecewiseConstant(
            dt=sig2.dt,
            samples=sig1(0.0) * sig2.samples,
            start_time=sig2.start_time,
            duration=sig2.duration,
            carrier_freq=sig2.carrier_freq,
            phase=sig2.phase,
        )
    elif isinstance(sig1, Constant) and type(sig2) == Signal:
        const = sig1(0.0)
        def new_env(t):
            return const * sig2.envelope(t)
        return Signal(envelope=new_env,
                      carrier_freq=sig2.carrier_freq,
                      phase=sig2.phase)
    elif isinstance(sig1, PiecewiseConstant) and isinstance(sig2, PiecewiseConstant):
        # verify compatible parameters before applying special rule
        if sig1.start_time == sig2.start_time and sig1.dt == sig2.dt and len(sig1.samples) == len(sig2.samples):
            if sig1.carrier_freq == 0:
                # absorb phase into envelope for signal 1 and multiply real part
                real_sig_samples = np.real(sig1.samples * np.exp(1j * sig1.phase))
                return PiecewiseConstant(
                            dt=sig2.dt,
                            samples=real_sig_samples * sig2.samples,
                            start_time=sig2.start_time,
                            duration=sig2.duration,
                            carrier_freq=sig2.carrier_freq,
                            phase=sig2.phase,
                        )
            else:
                pwc1 = PiecewiseConstant(
                            dt=sig2.dt,
                            samples=0.5 * sig1.samples * sig2.samples,
                            start_time=sig2.start_time,
                            duration=sig2.duration,
                            carrier_freq=sig1.carrier_freq + sig2.carrier_freq,
                            phase=sig1.phase + sig2.phase,
                        )
                pwc2 = PiecewiseConstant(
                            dt=sig2.dt,
                            samples=0.5 * sig1.samples * np.conjugate(sig2.samples),
                            start_time=sig2.start_time,
                            duration=sig2.duration,
                            carrier_freq=sig1.carrier_freq - sig2.carrier_freq,
                            phase=sig1.phase - sig2.phase,
                        )
                return pwc1 + pwc2
    elif sig1.carrier_freq == 0.0:
        # rule for generic signals if one has zero carrier frequency
        phase_factor = np.exp(1j * sig1.phase)
        def new_env(t):
            return np.real(sig1.envelope(t) * phase_factor) * sig2.envelope(t)

        return Signal(envelope=new_env,
                      carrier_freq=sig2.carrier_freq,
                      phase=sig2.phase)


    # if no special cases apply, implement generic rule
    def new_env1(t):
        return 0.5 * sig1.envelope(t) * sig2.envelope(t)

    def new_env2(t):
        return 0.5 * sig1.envelope(t) * np.conjugate(sig2.envelope(t))

    prod1 = Signal(envelope=new_env1,
                   carrier_freq=sig1.carrier_freq + sig2.carrier_freq,
                   phase=sig1.phase + sig2.phase)
    prod2 = Signal(envelope=new_env1,
                   carrier_freq=sig1.carrier_freq - sig2.carrier_freq,
                   phase=sig1.phase - sig2.phase)
    return prod1 + prod2


def sort_signals(sig1: Signal, sig2: Signal) -> Tuple[Signal, Signal]:
    """Sorts signals according to the partial order: ``sig1 <= sig2`` if and only if:
    - The ``type(sig1)`` preceds ``type(sig2)`` in the list
    ``[Constant, PiecewiseConstant, Signal]``, or
    - If ``type(sig1) == type(sig2)``, ``sig1.carrier_freq == 0``.

    This is a utility function useful for binary operations between ``Signal``s, in which
    special cases can be implemented for signals of "simpler" types. For example,
    once sorting, checking if one signal is ``Constant`` and the other is ``PiecewiseConstant``
    only requires checking the single case:
    ``type(sig1) == Constant and type(sig2) == PiecewiseConstant``.
    """
    # if sig2 has zero carrier frequency, swap them
    if sig2.carrier_freq == 0.0:
        sig1, sig2 = sig2, sig1

    if isinstance(sig1, Constant):
        return sig1, sig2
    elif isinstance(sig2, Constant):
        return sig2, sig1
    elif isinstance(sig1, PiecewiseConstant):
        return sig1, sig2
    elif isinstance(sig2, PiecewiseConstant):
        return sig2, sig1

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
        raise QiskitError('Input type incompatible with SignalSum.')

    return sig
