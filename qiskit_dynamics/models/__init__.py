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
======================================
Models (:mod:`qiskit_dynamics.models`)
======================================

.. currentmodule:: qiskit_dynamics.models

This module contains classes for constructing the right-hand side of an ordinary differential equations. In this package, a "model of a quantum system" means a description of a differential
equation used to model a physical quantum system, in this case either the
Schrodinger equation:

.. math::
    \dot{y}(t) = -iH(t)y(t),

where :math:`H(t)` is the Hamiltonian, or Lindblad equation:

.. math::
    \dot{\rho}(t) = -i[H(t), \rho(t)] +
                \sum_j g_j(t) \left(L_j\rho(t)L_j^\dagger -
                \frac{1}{2}\{L_j^\dagger L_j, \rho(t)\}\right),

where the sum term is called the *dissipation* term, each :math:`L_j` is called a
*dissipator*, and :math:`[\cdot, \cdot]` and :math:`\{\cdot, \cdot\}` are, respectively, the
matrix commutator and anti-commutator. Model classes primarily
serve a *computational* purpose, and expose functions for evaluating
model expressions, such as :math:`t \mapsto H(t)` or :math:`t,y \mapsto -iH(t)y` in the
case of a Hamiltonian.

The Schrodinger equation is represented via the
:class:`~qiskit_dynamics.models.HamiltonianModel` class, which represents
the decomposition:

.. math::
    H(t) = H_d + \sum_j s_j(t) H_j,

where both :math:`H_j` and the *drift* :math:`H_d` are Hermitian operators, and
:math:`s_j(t)` are either :class:`~qiskit_dynamics.signals.Signal` objects or
are numerical constants. Constructing a :class:`~qiskit_dynamics.models.HamiltonianModel`
requires specifying the above decomposition, e.g.:

.. code-block:: python

    hamiltonian = HamiltonianModel(operators, signals, drift)

with ``operators`` being a specification of the :math:`H_j`, ``signals`` a specification of
the :math:`s_j`, and ``drift`` a specification of :math:`H_d`.

Similarly, the :class:`~qiskit_dynamics.models.LindbladModel` class represents
the Lindblad equation as written above, with the Hamiltonian decomposed as in
:class:`~qiskit_dynamics.models.HamiltonianModel`. It may be instantiated as

.. code-block:: python

    lindblad_model = LindbladModel(hamiltonian_operators,
                                   hamiltonian_signals,
                                   drift,
                                   dissipator_operators,
                                   dissipator_signals)

where the arguments ``hamiltonian_operators``, ``hamiltonian_signals``, and ``drift`` are for
the Hamiltonian decomposition as in :class:`~qiskit_dynamics.models.HamiltonianModel`,
and the ``dissipator_operators`` correspond to the :math:`L_j`, and the ``dissipator_signals``
the :math:`g_j(t)`, which default to the constant ``1.``.

Once constructed, model classes enable *evaluation* of certain functions, e.g. for a
:class:`~qiskit_dynamics.models.HamiltonianModel`, ``hamiltonian.evaluate(t)`` returns
:math:`H(t)`, and ``hamiltonian.evaluate_rhs(t, y)`` evaluates :math:`-iH(t)y`. Similar
behaviour applies to :class:`~qiskit_dynamics.models.LindbladModel`, however the
:meth:`~qiskit_dynamics.models.LindbladModel.evaluate` method will raise an error unless
a vectorized evaluation mode is set (see below).

Rotating Frames
^^^^^^^^^^^^^^^

"Entering a rotating frame" is a common transformation on models of quantum systems.
For example, for a Hamiltonian, this corresponds to the transformation

.. math::
    H(t) \mapsto e^{-tF}(H(t) - F)e^{tF}

for an anti-Hermitian operator :math:`F`, called the *frame operator*,
commonly expressed as a Hermitian operator via the association
:math:`F = -iH`.

Any model class can be transformed into a rotating frame by setting the
``rotating_frame`` property:

.. code-block:: python

    model.rotating_frame = frame_operator

where ``frame_operator`` is specification of :math:`F=-iH`
(either giving :math:`F` or :math:`H`) of valid type. Setting this property modifies
the behaviour of the evaluation functions, e.g.
for a :class:`~qiskit_dynamics.models.HamiltonianModel`, ``hamiltonian.evaluate(t)``
will now evaluate to :math:`e^{-tF}(H(t) - F)e^{tF}`, and
``hamiltonian.evaluate_rhs(t, y)`` evaluates to :math:`e^{-tF}(H(t) - F)e^{tF}y`.
:class:`~qiskit_dynamics.models.LindbladModel` has similar behaviour.

Internally, the model classes make use of the :class:`~qiskit_dynamics.models.RotatingFrame`
class, which is constructed only with the frame operator :math:`F=-iH`, and contains helper
functions for transforming various objects into and out of the rotating frame.

Rotating wave approximation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The rotating wave approximation (RWA) is a transformation in which "high-frequency"
components, above a given cutoff frequency, are removed from a model.
This transformation is implemented in
:meth:`~qiskit_dynamics.models.rotating_wave_approximation`, see its documentation for
details.


Numerical methods and evaluation modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

All model classes offer different underlying numerical implementations that a user can choose
using the ``set_evaluation_mode`` method. For example,
:class:`~qiskit_dynamics.models.HamiltonianModel` can internally use
either sparse or dense arrays to compute :math:`H(t)` or a product :math:`-iH(t)y`.
The default is dense arrays, and a model can be set to use sparse arrays via:

.. code-block:: python

    model.set_evaluation_mode('sparse')

See ``set_evaluation_mode`` for each model class for available modes.

**Important**: When setting a rotating frame, models internally store their operators
in the basis in which the frame operator is diagonal. In general, sparsity of an operator
is not perserved by basis transformations. Hence, preserving internal sparsity with
rotating frames requires more restrictive choice of frames. For example, diagonal frame
operators exactly preserve sparsity.

The different modes are organized via a series of objects called ``OperatorCollection`` s,
which abstract the computational details of the particular equation out of
the model classes.


Model classes
=============

.. autosummary::
   :toctree: ../stubs/

   HamiltonianModel
   LindbladModel
   GeneratorModel

Model transformations
=====================

.. autosummary::
   :toctree: ../stubs/

   RotatingFrame
   rotating_wave_approximation

Operator Collections
====================

.. autosummary::
   :toctree: ../stubs/

   BaseOperatorCollection
   DenseOperatorCollection
   SparseOperatorCollection
   DenseLindbladCollection
   DenseVectorizedLindbladCollection
   SparseLindbladCollection
"""

from .rotating_frame import RotatingFrame
from .generator_models import BaseGeneratorModel, GeneratorModel, CallableGenerator
from .hamiltonian_models import HamiltonianModel
from .lindblad_models import LindbladModel
from .operator_collections import (
    BaseOperatorCollection,
    DenseOperatorCollection,
    SparseOperatorCollection,
    DenseLindbladCollection,
    DenseVectorizedLindbladCollection,
    SparseLindbladCollection,
)
from .rotating_wave_approximation import rotating_wave_approximation
