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
"""Deprecated import path for array Class"""

import warnings


# pylint: disable = invalid-name
def Array(*args, **kwargs):
    """Deprecated import path wrapper for initializing an Array."""
    warnings.warn(
        "Importing Array from `qiskit_dynamics.dispatch` is deprecated and will "
        "be removed next release. Import from `qiskit_dynamics.array` instead.",
        DeprecationWarning,
    )
    from qiskit_dynamics.array.array import Array as _Array

    return _Array(*args, **kwargs)
