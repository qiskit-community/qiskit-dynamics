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

try:
    import numpy
    from qiskit_dynamics.dispatch.register import register_module, register_type

    __all__ = []

    # Register numpy ndarray
    register_type(numpy.ndarray)

    # Register numpy modules
    register_module(numpy)
    register_module(numpy.linalg)
    register_module(numpy.random)

except ModuleNotFoundError:
    pass
