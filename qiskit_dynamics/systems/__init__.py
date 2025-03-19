# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
========================================
Systems (:mod:`qiskit_dynamics.systems`)
========================================

.. currentmodule:: qiskit_dynamics.systems

This module provides high level interfaces for building and solving models of quantum systems. Where
the :mod:`.solvers` and :mod:`.models` modules provide interfaces for defining and solving systems
in terms of user-defined arrays, this module provides tools for building descriptions of systems in
terms of tensor-factor subsystems, an algebraic system for defining operators on subsystems, a high
level class representing an abstract dynamical model of a quantum system, and tools for analysing
results. The ultimate purpose of the module is to minimize the need for a user to work explicitly
with building and manipulating arrays and array indexing, which can be time consuming and prone to
error. See the :ref:`Systems Modelling Tutorial <systems modelling tutorial>` and the :ref:`How-to
use advanced system modelling functionality <systems modelling userguide>` for detailed examples.

The core building block of a model is a :class:`.Subsystem`, which represents a single
finite-dimensional complex vector space on which to define the model of a quantum system. A single
model may be defined on multiple subsystems, in which each subsystem represents a tensor factor in a
tensor-product space.

.. code-block:: python

    Q0 = Subsystem(name="Q0", dim=2)
    Q1 = Subsystem(name="Q0", dim=2)


Abstract operators acting on these subsystems can be defined as follows:

.. code-block:: python

    X0 = X(Q0)
    Y1 = Y(Q1)

Using algebraic operations, new operators may be defined. For example, the tensor product of ``X``
on ``Q0`` and ``Y`` on ``Q1`` can be constructed through matrix multiplication:

.. code-block:: python

    X0 @ Y1

Similarly, the sum of these operators can be constructed through addition ``X0 + Y1``. To facilitate
working with operators on subsystems without needing to always specify the full context of all
subsystems in a given model, operators are always assumed to act as the identity on all unspecified
subsystems, similar to the common mathematical notation in which an operator :math:`a_2` means "the
operator :math:`a` acting on subsystem :math:`2` and the identity on all others".

The matrix of an abstract operator can be built by calling the ``matrix`` method. The specific
ordering of the tensor factors desired can be supplied, e.g.:

.. code-block:: python

    (X0 @ Y1).matrix(ordered_subsystems=[Q0, Q1])

If no explicitly ordering is supplied, the default internal ordering built during the construction
of the operator will be used. In addition to a set of pre-defined operators, users can instantiate a
:class:`.SubsystemOperator` instance with an arbitrary concrete matrix which acts on an arbitrary
list of :class:`.Subsystem` instances.

Operators can be assumbled into models of quantum systems using the :class:`.QuantumSystemModel`
class. For example, a model of a standard qubit can be built as follows:

.. code-block:: python

    q0_model = QuantumSystemModel(
        static_hamiltonian=2 * np.pi * 5. * N(Q0),
        drive_hamiltonians=[2 * np.pi * 0.1 * X(Q0)],
        drive_hamiltonian_coefficients=["d0"]
    )

This model can now be solved with a single call:

.. code-block:: python

    results = q0_model.solve(
        signals={"d0": Signal(1., carrier_freq=5.)},
        t_span=t_span,
        t_eval=t_eval,
        y0=y0
    )

with ``results`` being the standard ``OdeResult`` object returned by Qiskit Dynamics solvers.

In addition to the functionality above, this module contains the :class:`SubsystemMapping` class for
defining linear maps between tensor factor spaces given as lists of :class:`Subsystem` instances. As
shown in the :ref:`How-to use advanced system modelling functionality <systems modelling userguide>`
userguide entry, this class can be used to define injections of subspaces into larger spaces, or to
restrict a model to a subspace of interest.

Furthermore, the :class:`ONBasis` and :class:`DressedBasis` classes represent bases for subspaces on
tensor product spaces represented by lists of :class:`Subsystem` instances.

System modelling classes
========================

.. autosummary::
   :toctree: ../stubs/

   Subsystem
   SubsystemOperator
   FunctionOperator
   ONBasis
   DressedBasis
   SubsystemMapping
   QuantumSystemModel
   IdealQubit
   DuffingOscillator
   ExchangeInteraction

Pre-defined operators
=====================

.. autosummary::
   :toctree: ../stubs/

   I
   X
   Y
   Z
   N
   A
   Adag
"""

from .subsystem import Subsystem
from .subsystem_operators import *
from .abstract_subsystem_operators import FunctionOperator
from .orthonormal_basis import ONBasis, DressedBasis
from .subsystem_mapping import SubsystemMapping
from .quantum_system_model import *
