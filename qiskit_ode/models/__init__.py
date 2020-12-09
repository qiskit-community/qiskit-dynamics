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
=================================
Models (:mod:`qiskit_ode.models`)
=================================

.. currentmodule:: qiskit_ode.models

Models
======

.. autosummary::
   :toctree: ../stubs/

   BaseFrame
   Frame
   OperatorModel
   HamiltonianModel
   LindbladModel

Signals
=======

.. autosummary::
   :toctree: ../stubs/

   BaseSignal
   Signal
   PiecewiseConstant
   Constant
"""

from .frame import BaseFrame, Frame
from .operator_models import OperatorModel
from .quantum_models import HamiltonianModel, LindbladModel
from .signals import BaseSignal, Signal, PiecewiseConstant, Constant
from .transfer_functions import Convolution
