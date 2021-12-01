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
"""Register numpy library for Dispatch"""


def register_numpy(dispatcher) -> bool:
    """Register default implementation of numpy if installed"""
    try:
        import numpy

        # Register numpy ndarray
        dispatcher.register_type(numpy.ndarray)

        # Register numpy modules
        dispatcher.register_module(numpy)
        dispatcher.register_module(numpy.linalg)
        dispatcher.register_module(numpy.random)

        return True

    except ModuleNotFoundError:
        return False
