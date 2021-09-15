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

r"""
========================================
Signals (:mod:`qiskit_dynamics.signals`)
========================================

.. currentmodule:: qiskit_dynamics.signals

Signal objects represent a class of real-valued functions of the form:

.. math::
    s(t) = \textnormal{Re}[f(t)e^{i(2 \pi \nu t + \phi)}],

where

    * :math:`f` is a complex-valued function called the *envelope*,
    * :math:`\nu \in \mathbb{R}` is the *carrier frequency*, and
    * :math:`\phi \in \mathbb{R}` is the *phase*.

There are two main signal classes:

    #. :class:`~qiskit_dynamics.signals.Signal`: which
       is instantiated with an *arbitrary* python callable function for the envelope, and
    #. :class:`~qiskit_dynamics.signals.DiscreteSignal`: in which the envelope
       is piecewise constant, specified as a list of samples with a sample width and start time.

:class:`~qiskit_dynamics.signals.Signal` allows for complete generality,
while :class:`~qiskit_dynamics.signals.DiscreteSignal` is internally implemented
with array-based operations and geared more towards performance.


Signal
^^^^^^

A :class:`~qiskit_dynamics.signals.Signal` is instantiated with an envelope.

.. code-block:: python

    f = lambda t: t**2
    signal = Signal(envelope=f, carrier_freq=2.)

.. note::

    :class:`~qiskit_dynamics.signals.Signal` assumes the envelope ``f`` is
    *array vectorized* in the sense that ``f`` can operate on arrays of arbitrary shape
    and satisfy ``f(x)[idx] == f(x[idx])`` for a multidimensional index ``idx``. This
    can be ensured either by writing ``f`` to be vectorized, or by using the ``vectorize``
    function in ``numpy`` or ``jax.numpy``, e.g.:

.. code-block:: python

    import numpy as np
    vectorized_f = np.vectorize(f)
    signal = Signal(envelope=vectorized_f, carrier_freq=2.)

Once instantiated, various functions associated with the siganl can be evaluated:

.. code-block:: python

    t = 0.5
    # evaluate the full expression
    signal(t)
    # evaluate the envelope
    signal.envelope(t)
    # evaluate the complex value
    signal.complex_value(t)

Signals can also be visualized over a time interval by using the :meth:`~Signal.draw` method.

.. jupyter-execute::

    from qiskit_dynamics.signals import Signal
    signal = Signal(envelope=lambda t: t**2, carrier_freq=2.)

    # draw with start point endpoint, and number of samples
    signal.draw(t0=0., tf=2., n=100)


Constant Signal
^^^^^^^^^^^^^^^

:class:`~qiskit_dynamics.signals.Signal` supports specification of a *constant signal*:

.. code-block:: python

    const = Signal(2.)

This initializes the object to always return the constant ``2.``, and allows constants to be
treated on the same footing as arbitrary :class:`~qiskit_dynamics.signals.Signal` instances.
A :class:`~qiskit_dynamics.signals.Signal` operating in constant-mode can be checked via the
boolean attribute ``const.is_constant``.


DiscreteSignal
^^^^^^^^^^^^^^

A :class:`~qiskit_dynamics.signals.DiscreteSignal`  is a signal specified
as an array of samples with a sample width and an optional start point (defaults to ``0.``).
:class:`~qiskit_dynamics.signals.DiscreteSignal` inherits the same API as
:class:`~qiskit_dynamics.signals.Signal`, and once instantiated can be interacted with
in the same way.

.. jupyter-execute::

    from qiskit_dynamics.signals import DiscreteSignal
    from matplotlib import pyplot as plt

    discrete_signal = DiscreteSignal(dt=0.5, samples=[1., 2., 3.], carrier_freq=1.)

    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    discrete_signal.draw(t0=0., tf=1.5, n=100,
                         axis=axs[0], title='Signal')
    discrete_signal.draw(t0=0., tf=1.5, n=100,
                         function='complex_value', axis=axs[1], title='Complex value')

Note that the carrier wave is still analog for :class:`~qiskit_dynamics.signals.DiscreteSignal`.

Algebraic Operations and composite signals
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Signals support algebraic operations. Adding two signals results in a
:class:`~qiskit_dynamics.signals.SignalSum` object:

.. code-block:: python

    signal_sum = signal1 + signal2


:class:`~qiskit_dynamics.signals.SignalSum` again implements the same API
as :class:`~qiskit_dynamics.signals.Signal`, so for many purposes it can be
interacted with in the same manner as :class:`~qiskit_dynamics.signals.Signal`
and :class:`~qiskit_dynamics.signals.DiscreteSignal`.

The components of a sum are stored in the ``components`` attribute as a list, but may
also be accessed via subscripting, e.g. ``signal_sum[0]`` will produce ``signal1`` in
the codeblock above.

Signals of any type can be added together in this way. However, in the special case that
:class:`~qiskit_dynamics.signals.DiscreteSignal`\s with compatible sample structure
(same number of samples, ``dt``, and start time) are added together,
a :class:`~qiskit_dynamics.signals.DiscreteSignalSum` is produced.
:class:`~qiskit_dynamics.signals.DiscreteSignalSum` stores a sum of compatible
:class:`~qiskit_dynamics.signals.DiscreteSignal`\s by joining the underlying arrays,
so that the sum can be evaluated using purely array-based operations.

Signal multiplication is also supported via :class:`~qiskit_dynamics.signals.SignalSum`
and :class:`~qiskit_dynamics.signals.DiscreteSignalSum` using the identity:

.. math::

    Re[f(t)e^{i(2 \pi \nu t + \phi)}] \times &Re[g(t)e^{i(2 \pi \omega t + \psi)}]
         \\&= Re[\frac{1}{2} f(t)g(t)e^{i(2\pi (\omega + \nu)t + (\phi + \psi))} ]
          + Re[\frac{1}{2} f(t)\overline{g(t)}e^{i(2\pi (\omega - \nu)t + (\phi - \psi))} ].

I.e. multiplication produces a :class:`~qiskit_dynamics.signals.SignalSum` (or a
:class:`~qiskit_dynamics.signals.DiscreteSignalSum` for
:class:`~qiskit_dynamics.signals.DiscreteSignal`\s with compatible structure)
with two elements, whose envelopes, frequencies, and phases are as given by the above formula.

Sampling
^^^^^^^^

Both :class:`~qiskit_dynamics.signals.DiscreteSignal` and
:class:`~qiskit_dynamics.signals.DiscreteSignalSum` have constructors which build an
instance by sampling a :class:`~qiskit_dynamics.signals.Signal` or
:class:`~qiskit_dynamics.signals.SignalSum`. These constructors have the
option to just sample the envelope (and keep the carrier analog), or to also
sample the carrier.

.. jupyter-execute::

    from qiskit_dynamics.signals import Signal, DiscreteSignal
    from matplotlib import pyplot as plt

    # discretize a signal with and without samplying the carrier
    signal = Signal(lambda t: t, carrier_freq=2.)
    discrete_signal = DiscreteSignal.from_Signal(signal, dt=0.1, start_time=0.,
                                                 n_samples=10, sample_carrier=True)
    discrete_signal2 = DiscreteSignal.from_Signal(signal, dt=0.1, start_time=0., n_samples=10)

    # plot the signal against each discretization
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))
    signal.draw(t0=0., tf=1., n=100, axis=axs[0])
    discrete_signal.draw(t0=0., tf=1., n=100, axis=axs[0],
                         title='Signal v.s. Sampled envelope and carrier')
    signal.draw(t0=0., tf=1., n=100, axis=axs[1])
    discrete_signal2.draw(t0=0., tf=1., n=100, axis=axs[1],
                          title='Signal v.s. Sampled envelope')

Analagous functionality is available in the
:meth:`DiscreteSignalSum.from_SignalSum` method.


Signal Classes
==============

.. autosummary::
   :toctree: ../stubs/

   Signal
   DiscreteSignal
   SignalSum
   DiscreteSignalSum
   SignalList

Transfer Functions
==================

.. autosummary::
   :toctree: ../stubs/

   Convolution
   IQMixer
"""

from .signals import Signal, DiscreteSignal, SignalSum, DiscreteSignalSum, SignalList
from .transfer_functions import Convolution, Sampler, IQMixer
