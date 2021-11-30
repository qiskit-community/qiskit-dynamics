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
"""Register jax library for Dispatch"""
# pylint: disable=import-error


try:
    import jax
    from jax.interpreters.xla import DeviceArray
    from jax.core import Tracer
    from jax.interpreters.ad import JVPTracer
    from jax.interpreters.partial_eval import JaxprTracer

    JAX_TYPES = (DeviceArray, Tracer, JaxprTracer, JVPTracer)

    try:
        # This class is not in older versions of Jax
        from jax.interpreters.partial_eval import DynamicJaxprTracer

        JAX_TYPES += (DynamicJaxprTracer,)
    except ImportError:
        pass

    from qiskit_dynamics.dispatch.default_dispatcher import default_dispatcher as DISPATCHER

    __all__ = []

    # Register jax types
    for atype in JAX_TYPES:
        DISPATCHER.register_type(atype, "jax")

    # Register jax numpy modules
    DISPATCHER.register_module(jax.numpy, "jax")
    DISPATCHER.register_module(jax.numpy.linalg, "jax")

    # Jax doesn't implement a copy method, so we add one using the
    # jax numpy.array constructor which implicitly copies
    @DISPATCHER.register_function(name="copy", lib="jax")
    def _copy(array, order="K"):
        return jax.numpy.array(array, copy=True, order=order)


except ModuleNotFoundError:
    pass
