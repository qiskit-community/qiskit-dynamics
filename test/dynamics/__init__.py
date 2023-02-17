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
Qiskit Dynamics tests
"""

# temporarily disable a change in JAX 0.4.4 that introduced a bug. Must be run before importing JAX
import os

os.environ["JAX_JIT_PJIT_API_MERGE"] = "0"
