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

    from ..register import register_function, register_module, register_type
    import numpy as np
    from .numpy import array_repr as numpy_repr

    __all__ = []

    # Register jax types
    for atype in JAX_TYPES:
        register_type(atype, "jax")

    # Register modules
    register_module(jax.numpy, "jax")
    register_module(jax.numpy.linalg, "jax")

    # Register custom functions
    @register_function("jax", "repr")
    def array_repr(array, prefix="", suffix=""):
        """Wrapper for showing Numpy array in custom class"""
        if hasattr(array, "_value"):
            return numpy_repr(array._value, prefix=prefix, suffix=suffix)
        return prefix + repr(array) + suffix

    # Jax doesn't implement a copy method, so we add one using the
    # jax numpy.array constructor which implicitly copies
    @register_function("jax", np.copy)
    def _copy(array, order="K"):
        return jax.numpy.array(array, copy=True, order=order)


except ModuleNotFoundError:
    pass
