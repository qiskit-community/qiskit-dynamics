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

import numpy
from ..register import register_function, register_module, register_type

__all__ = []

# Register jax types
register_type(numpy.ndarray)

# Register modules
register_module(numpy)
register_module(numpy.linalg)
register_module(numpy.random)


@register_function(name="repr", lib="numpy")
def array_repr(array, prefix="", suffix=""):
    """Wrapper for showing Numpy array in custom class"""
    max_line_width = numpy.get_printoptions()["linewidth"]
    array_str = numpy.array2string(
        array,
        separator=", ",
        prefix=prefix,
        suffix="," if suffix else None,
        max_line_width=max_line_width,
    )
    sep = ""
    if len(suffix) > 1:
        last_line_width = len(array_str) - array_str.rfind("\n") + 1
        if last_line_width + len(suffix) + 1 > max_line_width:
            sep = ",\n" + " " * len(prefix)
        else:
            sep = ", "
    return prefix + array_str + sep + suffix
