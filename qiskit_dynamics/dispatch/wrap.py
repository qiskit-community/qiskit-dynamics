# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Deprecated import path for wrap"""

import warnings


def wrap(*args, **kwargs):
    """Deprecated import path wrapper for wrap function."""
    warnings.warn(
        "Importing `wrap` from `qiskit_dynamics.dispatch` is deprecated and will "
        "be removed next release. Import from `qiskit_dynamics.array` instead.",
        DeprecationWarning,
    )
    from qiskit_dynamics.array.wrap import wrap as _wrap

    return _wrap(*args, **kwargs)
