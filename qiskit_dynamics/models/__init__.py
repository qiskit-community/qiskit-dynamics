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

"""
======================================
Models (:mod:`qiskit_dynamics.models`)
======================================

.. currentmodule:: qiskit_dynamics.models

Tools for constructing models of differential equations.

Model classes
=============

Models for quantum systems.

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
Calculation objects used to implement multiple evaluation modes.

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
