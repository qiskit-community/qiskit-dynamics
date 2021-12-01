# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Default dispatcher"""

from qiskit_dynamics.dispatch.dispatcher import Dispatcher
from qiskit_dynamics.dispatch.default_libraries import register_numpy, register_jax


def default_dispatcher() -> Dispatcher:
    """REturn a dispatcher with installed default libraries registered.

    Libraries which will be registered if installed in the current
    Python environment are

    * Numpy
    * JAX

    Returns:
        A dispatcher with installed libraries registered.
    """
    dispatcher = Dispatcher()
    register_numpy(dispatcher)
    register_jax(dispatcher)
    return dispatcher
