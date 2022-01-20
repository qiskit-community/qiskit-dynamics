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

try:
    import jax.numpy as jnp
except ImportError:
    pass

from qiskit import QiskitError
from qiskit_dynamics.array import Array


class Signal:
    r"""General signal class.

    Represents a function of the form:

    .. math::
        Re[f(t)e^{i (2 \pi \nu t + \phi)}]
                = Re[f(t)]\cos(2 \pi \nu t + \phi) - Im[f(t)]\sin(2 \pi \nu t + \phi),

    where

    - :math:`f(t)` is the envelope function.
    - :math:`\nu` is the carrier frequency.
    - :math:`\phi` is the phase.

    The envelope function can be specified either as a constant numeric value
    (indicating a constant function), or as a complex-valued callable,
    and the frequency and phase must be real.


    .. note::

        :class:`~qiskit_dynamics.signals.Signal` assumes the envelope ``f`` is
        *array vectorized* in the sense that ``f`` can operate on arrays of arbitrary shape
        and satisfy ``f(x)[idx] == f(x[idx])`` for a multidimensional index ``idx``. This
        can be ensured either by writing ``f`` to be vectorized, or by using the ``vectorize``
        function in ``numpy`` or ``jax.numpy``.

        For example, for an unvectorized envelope function ``f``:

        .. code-block:: python

            import numpy as np
            vectorized_f = np.vectorize(f)
            signal = Signal(envelope=vectorized_f, carrier_freq=2.)
    """

    def __init__(
        self,
        envelope: Union[Callable, complex, float, int, Array],
        carrier_freq: Union[float, List, Array] = 0.0,
        phase: Union[float, List, Array] = 0.0,
        name: Optional[str] = None,
    ):
        """
        Initializes a signal given by an envelope and a carrier.

        Args:
            envelope: Envelope function of the signal, must be vectorized.
            carrier_freq: Frequency of the carrier. Subclasses such as SignalSums
                          represent the carriers of each signal in an array.
            phase: The phase of the carrier. Subclasses such as SignalSums
                   represent the phase of each signal in an array.
            name: Name of the signal.
        """
        self._name = name
        self._is_constant = False

        if isinstance(envelope, (complex, float, int)):
            envelope = Array(complex(envelope))

        if isinstance(envelope, Array):

            # if envelope is constant and the carrier is zero, this is a constant signal
            if carrier_freq == 0.0:
                self._is_constant = True

            if envelope.backend == "jax":
                self._envelope = lambda t: envelope * jnp.ones_like(t)
            else:
                self._envelope = lambda t: envelope * np.ones_like(t)
        elif callable(envelope):
            if Array.default_backend() == "jax":
                self._envelope = lambda t: Array(envelope(t))
            else:
                self._envelope = envelope

        # set carrier and phase
        self.carrier_freq = carrier_freq
        self.phase = phase

    @property
    def name(self) -> str:
        """Return the name of the signal."""
        return self._name

    @property
    def is_constant(self) -> bool:
        """Whether or not the signal is constant."""
        return self._is_constant

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

    def __call__(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        """Vectorized evaluation of the signal at time(s) t."""
        return np.real(self.complex_value(t))

    def __str__(self) -> str:
        """Return string representation."""
        if self.name is not None:
            return str(self.name)

        if self.is_constant:
            return "Constant({})".format(str(self(0.0)))

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
        """Return a new signal whose complex value is the complex conjugate of this one."""

        def conj_env(t):
            return np.conjugate(self.envelope(t))

        return Signal(conj_env, -self.carrier_freq, -self.phase)

    def draw(
        self,
        t0: float,
        tf: float,
        n: int,
        function: Optional[str] = "signal",
        axis: Optional[plt.axis] = None,
        title: Optional[str] = None,
    ):
        """Plot the signal over an interval.

        The ``function`` arg specifies which function to
        plot:

            - ``function == 'signal'`` plots the full signal.
            - ``function == 'envelope'`` plots the complex envelope.
            - ``function == 'complex_value'`` plots the ``complex_value``.

        Args:
            t0: Initial time.
            tf: Final time.
            n: Number of points to sample in interval.
            function: Which function to plot.
            axis: The axis to use for plotting.
            title: Title of plot.
        """

        if axis is None:
            plotter = plt
            plotter.title(title)
        else:
            plotter = axis
            plotter.set_title(title)

        t_vals = np.linspace(t0, tf, n)

        y_vals = None
        data_type = "real"
        if function == "signal":
            y_vals = self(t_vals)
            title = title or "Value of " + str(self)
        elif function == "envelope":
            y_vals = self.envelope(t_vals)
            data_type = "complex"
            title = title or "Envelope of " + str(self)
        elif function == "complex_value":
            y_vals = self.complex_value(t_vals)
            data_type = "complex"
            title = title or "Complex value of " + str(self)

        legend = False
        if data_type == "complex":
            plotter.plot(t_vals, np.real(y_vals), label="Real")
            plotter.plot(t_vals, np.imag(y_vals), label="Imag")
            legend = True
        else:
            plotter.plot(t_vals, y_vals)

        if legend:
            plotter.legend()


class DiscreteSignal(Signal):
    r"""Piecewise constant signal implemented as an array of samples.

    The envelope is specified by an array of samples ``s = [s_0, ..., s_k]``, sample width ``dt``,
    and a start time ``t_0``, with the envelope being evaluated as
    :math:`f(t) =` ``s[floor((t - t0)/dt)]``.
    By default a :class:`~qiskit_dynamics.signals.DiscreteSignal` is defined to start at
    :math:`t=0` but a custom start time can be set via the ``start_time`` kwarg.
    """

    def __init__(
        self,
        dt: float,
        samples: Union[Array, List],
        start_time: float = 0.0,
        carrier_freq: Union[float, List, Array] = 0.0,
        phase: Union[float, List, Array] = 0.0,
        name: str = None,
    ):
        """Initialize a piecewise constant signal.

        Args:
            dt: The duration of each sample.
            samples: The array of samples.
            start_time: The time at which the signal starts.
            carrier_freq: Frequency of the carrier. Subclasses such as SignalSums
                          represent the carriers of each signal in an array.
            phase: The phase of the carrier. Subclasses such as SignalSums
                   represent the phase of each signal in an array.
            name: name of the signal.
        """
        self._dt = dt
        self._samples = Array(samples)
        self._start_time = start_time

        # define internal envelope function
        if self._samples.backend == "jax":

            def envelope(t):
                t = Array(t).data
                idx = jnp.clip(
                    jnp.array((t - self._start_time) // self._dt, dtype=int),
                    0,
                    len(self._samples) - 1,
                )
                return self._samples[idx]

        else:

            def envelope(t):
                idx = np.clip(
                    np.array((t - self._start_time) // self._dt, dtype=int),
                    0,
                    len(self._samples) - 1,
                )
                return self._samples[idx]

        Signal.__init__(self, envelope=envelope, carrier_freq=carrier_freq, phase=phase, name=name)

    @classmethod
    def from_Signal(
        cls,
        signal: Signal,
        dt: float,
        n_samples: int,
        start_time: Optional[float] = 0.0,
        sample_carrier: Optional[bool] = False,
    ):
        r"""Constructs a ``DiscreteSignal`` object by sampling a ``Signal``\.

        The optional argument ``sample_carrier`` controls whether or not to include the carrier
        in the sampling. I.e.:

            - If ``sample_carrier == False``\, a ``DiscreteSignal`` is constructed with:
                - ``samples`` obtained by sampling ``signal.envelope``\.
                - ``carrier_freq = signal.carrier_freq``\.
                - ``phase = signal.phase``\.

            - If ``sample_carrier == True``\, a ``DiscreteSignal`` is constructed with:
                - ``samples`` obtained by sampling ``signal`` (as a ``callable``\)
                - ``carrier_freq = 0``\.
                - ``phase = signal.phase``\.

        In either case, samples are obtained from the midpoint of each interval.

        Args:
            signal: Signal to sample.
            dt: Time increment to use.
            n_samples: Number of steps to resample with.
            start_time: Start time from which to resample.
            sample_carrier: Whether or not to include the carrier in the sampling.

        Returns:
            DiscreteSignal: The discretized ``Signal``\.
        """

        times = start_time + (np.arange(n_samples) + 0.5) * dt
        freq = signal.carrier_freq

        if sample_carrier:
            freq = 0.0
            samples = signal(times)
        else:
            samples = signal.envelope(times)

        return DiscreteSignal(
            dt,
            samples,
            start_time=start_time,
            carrier_freq=freq,
            phase=signal.phase,
            name=signal.name,
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
        return self._samples

    @property
    def start_time(self) -> float:
        """
        Returns:
            start_time: The time at which the list of samples start.
        """
        return self._start_time

    def conjugate(self):
        return self.__class__(
            dt=self._dt,
            samples=np.conjugate(self._samples),
            start_time=self._start_time,
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


class SignalCollection:
    """Base class for a list-like collection of signals."""

    def __init__(self, signal_list: List[Signal]):
        """Initialize by storing list of signals.

        Args:
            signal_list: List of signals.
        """
        self._is_constant = False
        self._components = signal_list

    @property
    def components(self) -> List[Signal]:
        """The list of components."""
        return self._components

    def __len__(self):
        """Number of components."""
        return len(self.components)

    def __getitem__(
        self, idx: Union[int, List, np.array, slice]
    ) -> Union[Signal, "SignalCollection"]:
        """Get item with Numpy-style subscripting, as if this class were a 1d array."""

        if type(idx) == np.ndarray and idx.ndim > 0:
            idx = list(idx)

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

        # at this point sublist should either be a single Signal, or a list of Signals
        if isinstance(sublist, list):
            return self.__class__(sublist)
        else:
            return sublist

    def __iter__(self):
        """Return iterator over component list."""
        return self.components.__iter__()

    def conjugate(self) -> "SignalCollection":
        """Return the conjugation of this collection."""
        return self.__class__([sig.conjugate() for sig in self.components])


class SignalSum(SignalCollection, Signal):
    r"""Represents a sum of signals.

    I.e. a sum of ``Signal`` objects:

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
        r"""Initialize with a list of Signal objects through ``args``\.

        Args:
            signals: ``Signal`` subclass objects.
            name: Name of the sum.

        Raises:
            QiskitError: If ``signals`` are not subclasses of ``Signal``\.
        """
        self._name = name

        components = []
        for sig in signals:
            if isinstance(sig, list):
                sig = SignalSum(*sig)

            if isinstance(sig, SignalSum):
                components += sig.components
            elif isinstance(sig, Signal):
                components.append(sig)
            elif isinstance(sig, (int, float, complex)) or (
                isinstance(sig, Array) and sig.ndim == 0
            ):
                components.append(Signal(sig))
            else:
                raise QiskitError(
                    "Components of a SignalSum must be instances of a Signal subclass."
                )

        SignalCollection.__init__(self, components)

        # set up routine for evaluating envelopes if jax
        if Array.default_backend() == "jax":
            jax_arraylist_eval = array_funclist_evaluate([sig.envelope for sig in self.components])

            def envelope(t):
                return np.moveaxis(jax_arraylist_eval(t), 0, -1)

        else:

            def envelope(t):
                return np.moveaxis([sig.envelope(t) for sig in self.components], 0, -1)

        carrier_freqs = []
        for sig in self.components:
            carrier_freqs.append(sig.carrier_freq)

        phases = []
        for sig in self.components:
            phases.append(sig.phase)

        Signal.__init__(
            self, envelope=envelope, carrier_freq=carrier_freqs, phase=phases, name=name
        )

    def complex_value(self, t: Union[float, np.array, Array]) -> Union[complex, np.array, Array]:
        """Return the sum of the complex values of each component."""
        if Array.default_backend() == "jax":
            t = Array(t)
        exp_phases = np.exp(np.expand_dims(t, -1) * self._carrier_arg + self._phase_arg)
        return np.sum(self.envelope(t) * exp_phases, axis=-1)

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
        r"""Merge into a single ``Signal``\. The output frequency is given by the
        average.
        """

        if len(self) == 0:
            return Signal(0.0)
        elif len(self) == 1:
            return self.components[0]

        ave_freq = np.sum(self.carrier_freq) / len(self)
        shifted_arg = self._carrier_arg - (1j * 2 * np.pi * ave_freq)

        def merged_env(t):
            exp_phases = np.exp(np.expand_dims(Array(t), -1) * shifted_arg + self._phase_arg)
            return np.sum(self.envelope(t) * exp_phases, axis=-1)

        return Signal(envelope=merged_env, carrier_freq=ave_freq, name=str(self))


class DiscreteSignalSum(DiscreteSignal, SignalSum):
    """Represents a sum of piecewise constant signals, all with the same
    time parameters: dt, number of samples, and start time.
    """

    def __init__(
        self,
        dt: float,
        samples: Union[List, Array],
        start_time: float = 0.0,
        carrier_freq: Union[List, np.array, Array] = None,
        phase: Union[List, np.array, Array] = None,
        name: str = None,
    ):
        r"""Directly initialize a ``DiscreteSignalSum``\. Samples of all terms in the
        sum are specified as a 2d array, with 0th axis indicating time, and 1st axis
        indicating a term in the sum.

        Args:
            dt: The duration of each sample.
            samples: The 2d array representing a list whose elements are all envelope values
                     at a given time.
            start_time: The time at which the signal starts.
            carrier_freq: Array with the carrier frequencies of each term in the sum.
            phase: Array with the phases of each term in the sum.
            name: name of the signal.
        """

        if carrier_freq is None:
            carrier_freq = np.zeros(samples.shape[-1], dtype=float)

        if phase is None:
            phase = np.zeros(samples.shape[-1], dtype=float)

        DiscreteSignal.__init__(
            self,
            dt=dt,
            samples=samples,
            start_time=start_time,
            carrier_freq=carrier_freq,
            phase=phase,
            name=name,
        )

        # construct individual components so they can be accessed as in SignalSum
        components = []
        for sample_row, freq, phi in zip(self.samples.transpose(), carrier_freq, phase):
            components.append(
                DiscreteSignal(
                    dt=self.dt,
                    samples=sample_row,
                    start_time=self.start_time,
                    carrier_freq=freq,
                    phase=phi,
                )
            )

        self._components = components

    @classmethod
    def from_SignalSum(
        cls,
        signal_sum: SignalSum,
        dt: float,
        n_samples: int,
        start_time: Optional[float] = 0.0,
        sample_carrier: Optional[bool] = False,
    ):
        r"""Constructs a ``DiscreteSignalSum`` object by sampling a ``SignalSum``\.

        The optional argument ``sample_carrier`` controls whether or not to include the carrier
        in the sampling. I.e.:

            - If ``sample_carrier == False``, a ``DiscreteSignalSum`` is constructed with:
                - ``samples`` obtained by sampling ``signal_sum.envelope``\.
                - ``carrier_freq = signal_sum.carrier_freq``\.
                - ``phase = signal_sum.phase``\.

            - If ``sample_carrier == True``\, a ``DiscreteSignal`` is constructed with:
                - ``samples`` obtained by sampling ``signal_sum`` (as a ``callable``\)
                - ``carrier_freq = 0``\.
                - ``phase = signal_sum.phase``\.

        In either case, samples are obtained from the midpoint of each interval.

        Args:
            signal_sum: SignalSum to sample.
            dt: Time increment to use.
            n_samples: Number of steps to resample with.
            start_time: Start time from which to resample.
            sample_carrier: Whether or not to include the carrier in the sampling.

        Returns:
            DiscreteSignalSum: The discretized ``SignalSum``\.
        """

        times = start_time + (np.arange(n_samples) + 0.5) * dt

        freq = signal_sum.carrier_freq

        if sample_carrier:
            freq = 0.0 * freq
            exp_phases = np.exp(np.expand_dims(Array(times), -1) * signal_sum._carrier_arg)
            samples = signal_sum.envelope(times) * exp_phases
        else:
            samples = signal_sum.envelope(times)

        return DiscreteSignalSum(
            dt,
            samples,
            start_time=start_time,
            carrier_freq=freq,
            phase=signal_sum.phase,
            name=signal_sum.name,
        )

    def __str__(self):
        """Get the string rep."""
        if self.name is not None:
            return str(self.name)

        if len(self) == 0:
            return "DiscreteSignalSum()"

        default_str = str(self[0])
        for sig in self.components[1:]:
            default_str += " + {}".format(str(sig))

        return default_str

    def __getitem__(self, idx: Union[int, List, np.array, slice]) -> Signal:
        """Enables numpy-style subscripting, as if this class were a 1d array."""

        if type(idx) == int and idx >= len(self):
            raise IndexError("index out of range for DiscreteSignalSum of length " + str(len(self)))

        samples = self.samples[:, idx]
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
            carrier_freq=carrier_freqs,
            phase=phases,
        )


class SignalList(SignalCollection):
    r"""A list of signals with functionality for simultaneous evaluation.

    The passed list is stored in the ``components`` attribute.
    """

    def __init__(self, signal_list: List[Signal]):

        signal_list = [to_SignalSum(signal) for signal in signal_list]

        super().__init__(signal_list)

        # setup complex value and full signal evaluation
        if Array.default_backend() == "jax":
            self._eval_complex_value = array_funclist_evaluate(
                [sig.complex_value for sig in self.components]
            )
            self._eval_signals = array_funclist_evaluate(self.components)
        else:
            self._eval_complex_value = lambda t: [sig.complex_value(t) for sig in self.components]
            self._eval_signals = lambda t: [sig(t) for sig in self.components]

    def complex_value(self, t: Union[float, np.array, Array]) -> Union[np.array, Array]:
        """Vectorized evaluation of complex value of components."""
        return np.moveaxis(self._eval_complex_value(t), 0, -1)

    def __call__(self, t: Union[float, np.array, Array]) -> Union[np.array, Array]:
        """Vectorized evaluation of all components."""
        return np.moveaxis(self._eval_signals(t), 0, -1)

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
        r"""Return the drift ``Array``\, i.e. return an ``Array`` whose entries are the sum
        of the constant parts of the corresponding component of this ``SignalList``\.
        """

        drift_array = []
        for sig_entry in self.components:
            val = 0.0

            if not isinstance(sig_entry, SignalSum):
                sig_entry = SignalSum(sig_entry)

            for term in sig_entry:
                if term.is_constant:
                    val += Array(term(0.0)).data

            drift_array.append(val)

        return Array(drift_array)


def signal_add(sig1: Signal, sig2: Signal) -> SignalSum:
    """Add two signals."""

    # generic routine
    # convert to SignalSum instances
    try:
        sig1 = to_SignalSum(sig1)
        sig2 = to_SignalSum(sig2)
    except QiskitError as qe:
        raise QiskitError("Only a number or a Signal instance can be added to a Signal.") from qe

    # if both are DiscreteSignalSum objects with compatible structure, append data together
    if isinstance(sig1, DiscreteSignalSum) and isinstance(sig2, DiscreteSignalSum):
        if (
            sig1.dt == sig2.dt
            and sig1.start_time == sig2.start_time
            and sig1.duration == sig2.duration
        ):
            samples = np.append(sig1.samples, sig2.samples, axis=1)
            carrier_freq = np.append(sig1.carrier_freq, sig2.carrier_freq)
            phase = np.append(sig1.phase, sig2.phase)
            return DiscreteSignalSum(
                dt=sig1.dt,
                samples=samples,
                start_time=sig1.start_time,
                carrier_freq=carrier_freq,
                phase=phase,
            )

    sig_sum = SignalSum(*(sig1.components + sig2.components))

    return sig_sum


def signal_multiply(sig1: Signal, sig2: Signal) -> SignalSum:
    r"""Multiply two ``Signal``\s. For a pair of elementary (non-``SignalSum``\) ``Signal``\s,
    expands the product of two signals into a ``SignalSum`` via the formula:

    .. math::
        Re[f(t)e^{i(2 \pi \nu t + \phi)}] \times Re[g(t)e^{i(2 \pi \omega t + \psi)}]
         = Re[\frac{1}{2} f(t)g(t)e^{i(2\pi (\omega + \nu)t + (\phi + \psi))} ]
          + Re[\frac{1}{2} f(t)\overline{g(t)}e^{i(2\pi (\omega - \nu)t + (\phi - \psi))} ]

    If either (or both) of ``sig1`` or ``sig2`` are ``SignalSum``\s, the multiplication is
    distributed over addition.
    """

    # convert to SignalSum instances
    try:
        sig1 = to_SignalSum(sig1)
        sig2 = to_SignalSum(sig2)
    except QiskitError as qe:
        raise QiskitError("Only a number or a Signal instance can multiply a Signal.") from qe

    sig1, sig2 = sort_signals(sig1, sig2)

    # if sig1 contains only a constant and sig2 is a DiscreteSignalSum
    if len(sig1) == 1 and sig1[0].is_constant and isinstance(sig2, DiscreteSignalSum):
        return DiscreteSignalSum(
            dt=sig2.dt,
            samples=sig1(0.0) * sig2.samples,
            start_time=sig2.start_time,
            carrier_freq=sig2.carrier_freq,
            phase=sig2.phase,
        )
    # if both are DiscreteSignalSum objects with compatible structure, append data together
    elif isinstance(sig1, DiscreteSignalSum) and isinstance(sig2, DiscreteSignalSum):
        if (
            sig1.dt == sig2.dt
            and sig1.start_time == sig2.start_time
            and sig1.duration == sig2.duration
        ):
            # this vectorized operation produces a 2d array whose columns are the products of
            # the original columns
            new_samples = Array(
                0.5
                * (sig1.samples[:, :, None] * sig2.samples[:, None, :]).reshape(
                    (sig1.samples.shape[0], sig1.samples.shape[1] * sig2.samples.shape[1]),
                    order="C",
                )
            )
            new_samples_conj = Array(
                0.5
                * (sig1.samples[:, :, None] * sig2.samples[:, None, :].conj()).reshape(
                    (sig1.samples.shape[0], sig1.samples.shape[1] * sig2.samples.shape[1]),
                    order="C",
                )
            )
            samples = np.append(new_samples, new_samples_conj, axis=1)

            new_freqs = sig1.carrier_freq + sig2.carrier_freq
            new_freqs_conj = sig1.carrier_freq - sig2.carrier_freq
            freqs = np.append(Array(new_freqs), Array(new_freqs_conj))

            new_phases = sig1.phase + sig2.phase
            new_phases_conj = sig1.phase - sig2.phase
            phases = np.append(Array(new_phases), Array(new_phases_conj))

            return DiscreteSignalSum(
                dt=sig1.dt,
                samples=samples,
                start_time=sig1.start_time,
                carrier_freq=freqs,
                phase=phases,
            )

    # initialize to empty sum
    product = SignalSum()

    # loop through every pair of components and multiply
    for comp1, comp2 in itertools.product(sig1.components, sig2.components):
        product += base_signal_multiply(comp1, comp2)

    return product


def base_signal_multiply(sig1: Signal, sig2: Signal) -> Signal:
    r"""Utility function for multiplying two elementary (non ``SignalSum``\) signals.
    This function assumes ``sig1`` and ``sig2`` are legitimate instances of ``Signal``
    subclasses.

    Special cases:

        - Multiplication of two constant ``Signal``\s returns a constant ``Signal``\.
        - Multiplication of a constant ``Signal`` and a ``DiscreteSignal`` returns
        a ``DiscreteSignal``\.
        - If two ``DiscreteSignal``\s have compatible parameters, the resulting signals are
        ``DiscreteSignal``\, with the multiplication being implemented by array multiplication of
        the samples.
        - Lastly, if no special rules apply, the two ``Signal``\s are multiplied generically via
        multiplication of the envelopes as functions.

    When a sum with two signals is produced, the carrier frequency (phase) of each component are,
    respectively, the sum and difference of the two frequencies (phases). For special cases
    involving constant ``Signal``\s and a non-constant ``Signal``, the carrier frequency and phase
    are preserved as that of the non-constant piece.

    Args:
        sig1: First signal.
        sig2: Second signal.

    Returns:
        SignalSum: Representing the RHS of the formula when two Signals are multiplied.
    """

    # ensure signals are ordered from most to least specialized
    sig1, sig2 = sort_signals(sig1, sig2)

    if sig1.is_constant and sig2.is_constant:
        return Signal(sig1(0.0) * sig2(0.0))
    elif sig1.is_constant and type(sig2) is DiscreteSignal:
        # multiply the samples by the constant
        return DiscreteSignal(
            dt=sig2.dt,
            samples=sig1(0.0) * sig2.samples,
            start_time=sig2.start_time,
            carrier_freq=sig2.carrier_freq,
            phase=sig2.phase,
        )
    elif sig1.is_constant and type(sig2) is Signal:
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
                carrier_freq=sig1.carrier_freq + sig2.carrier_freq,
                phase=sig1.phase + sig2.phase,
            )
            pwc2 = DiscreteSignal(
                dt=sig2.dt,
                samples=0.5 * sig1.samples * np.conjugate(sig2.samples),
                start_time=sig2.start_time,
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
    r"""Utility function for ordering a pair of ``Signal``\s according to the partial order:
    ``sig1 <= sig2`` if and only if:
        - ``type(sig1)`` precedes ``type(sig2)`` in the list
        ``[constant, DiscreteSignal, Signal, SignalSum, DiscreteSignalSum]``\.
    """
    if sig1.is_constant:
        pass
    elif sig2.is_constant:
        sig1, sig2 = sig2, sig1
    elif isinstance(sig1, DiscreteSignal) and not isinstance(sig1, DiscreteSignalSum):
        pass
    elif isinstance(sig2, DiscreteSignal) and not isinstance(sig2, DiscreteSignalSum):
        sig2, sig1 = sig1, sig2
    elif isinstance(sig1, Signal) and not isinstance(sig1, SignalSum):
        pass
    elif isinstance(sig2, Signal) and not isinstance(sig2, SignalSum):
        sig2, sig1 = sig1, sig2
    elif isinstance(sig1, SignalSum) and not isinstance(sig1, DiscreteSignalSum):
        pass
    elif isinstance(sig2, SignalSum) and not isinstance(sig2, DiscreteSignalSum):
        sig2, sig1 = sig1, sig2
    elif isinstance(sig1, DiscreteSignalSum):
        pass
    elif isinstance(sig2, DiscreteSignalSum):
        sig2, sig1 = sig1, sig2

    return sig1, sig2


def to_SignalSum(sig: Union[int, float, complex, Array, Signal]) -> SignalSum:
    r"""Convert the input to a SignalSum according to:

        - If it is already a ``SignalSum``\, do nothing.
        - If it is a Signal but not a ``SignalSum``\, wrap in a ``SignalSum``\.
        - If it is a number, wrap in constant ``Signal`` in a ``SignalSum``\.
        - Otherwise, raise an error.

    Args:
        sig: A SignalSum compatible input.

    Returns:
        SignalSum

    Raises:
        QiskitError: If the input type is incompatible with SignalSum.
    """

    if isinstance(sig, (int, float, complex)) or (isinstance(sig, Array) and sig.ndim == 0):
        return SignalSum(Signal(sig))
    elif isinstance(sig, DiscreteSignal) and not isinstance(sig, DiscreteSignalSum):
        return DiscreteSignalSum(
            dt=sig.dt,
            samples=Array([sig.samples.data]).transpose(1, 0),
            start_time=sig.start_time,
            carrier_freq=Array([sig.carrier_freq.data]),
            phase=Array([sig.phase.data]),
        )
    elif isinstance(sig, Signal) and not isinstance(sig, SignalSum):
        return SignalSum(sig)
    elif isinstance(sig, SignalSum):
        return sig

    raise QiskitError("Input type incompatible with SignalSum.")


def array_funclist_evaluate(func_list: List[Callable]) -> Callable:
    """Utility for evaluating a list of functions in a way that respects Arrays.
    Currently relevant for JAX evaluation.
    """

    def eval_func(t):
        return Array([Array(func(t)).data for func in func_list])

    return eval_func
