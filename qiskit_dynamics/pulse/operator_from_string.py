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

"""Generate operators from string."""

import numpy as np


def operator_from_string(op_label: str, subsystem_index: int, subsystem_dims: dict) -> np.ndarray:
    """ Generates an operator acting on a single subsystem, tensoring identities for remaining
    subsystems.

    inputs:
        - op_label: label for a single-subsystem operator
        - subsystem_index: index of the subsystem that the operator applies to
        - subsystem_dims: dimensions of all subsystems.

    returns:
        np.ndarray corresponding to the specified operator.
    """
    pass


# functions for
def a(dim):
