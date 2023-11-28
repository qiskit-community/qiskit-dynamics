# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,no-member

"""
Test global alias instances.
"""

from functools import partial
import unittest

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix

from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS
from qiskit_dynamics import DYNAMICS_NUMPY as unp
from qiskit_dynamics import DYNAMICS_SCIPY as usp
from qiskit_dynamics.arraylias.alias import _preferred_lib, _numpy_multi_dispatch

from ..common import test_array_backends, JAXTestBase

try:
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO
except:
    pass


@partial(test_array_backends, array_libraries=["numpy", "jax", "array_numpy", "array_jax"])
class TestDynamicsNumpy:
    """Test cases for global numpy configuration."""

    def test_simple_case(self):
        """Validate correct type and output."""
        a = self.asarray([1.0, 2.0, 3.0])
        output = unp.exp(a)
        self.assertArrayType(output)

        expected = np.exp(np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(output, expected)


@test_array_backends
class TestDynamicsScipy:
    """Test cases for global scipy configuration."""

    def test_simple_case(self):
        """Validate correct type and output."""
        a = self.asarray([1.0, 2.0, 3.0])
        output = usp.fft.dct(a)
        self.assertArrayType(output)

        expected = sp.fft.dct(np.array([1.0, 2.0, 3.0]))
        self.assertAllClose(output, expected)


class TestDynamicsNumpyAliasType(unittest.TestCase):
    """Test cases for which types are registered in dynamics_numpy_alias."""

    def test_spmatrix_type(self):
        """Test spmatrix is registered as scipy_sparse."""
        sp_matrix = csr_matrix([[0.0, 1.0], [1.0, 0.0]])
        registered_type_name = "scipy_sparse"
        self.assertTrue(registered_type_name in DYNAMICS_NUMPY_ALIAS.infer_libs(sp_matrix))

    def test_bcoo_type(self):
        """Test bcoo is registered."""
        try:
            from jax.experimental.sparse import BCOO

            bcoo = BCOO.fromdense([[0.0, 1.0], [1.0, 0.0]])
            registered_type_name = "jax_sparse"
            self.assertTrue(registered_type_name in DYNAMICS_NUMPY_ALIAS.infer_libs(bcoo)[0])
        except ImportError as err:
            raise unittest.SkipTest("Skipping jax tests.") from err


@partial(test_array_backends, array_libraries=["numpy", "jax", "jax_sparse"])
class Test_linear_combo:

    def test_linear_combo(self):
        mats = self.asarray([[[0., 1.], [1., 0.]], [[1j, 0.], [0., -1j]]])
        coeffs = np.array([1., 2.])
        out = _numpy_multi_dispatch(coeffs, mats, path="linear_combo")
        self.assertArrayType(out)
        
        if "sparse" in self.array_library():
            out = out.todense()
        self.assertAllClose(out, np.array([[2j, 1.], [1., -2j]]))


class Test_preferred_lib(JAXTestBase):
    """Test class for preferred_lib functions. Inherits from JAXTestBase as this functionality
    is primarily to facilitate JAX types.
    """

    def test_defaults_to_numpy(self):
        """Test that it defaults to numpy."""
        self.assertEqual(_preferred_lib(None), "numpy")
    
    def test_prefers_jax_over_numpy(self):
        """Test that it chooses jax over numpy."""
        self.assertEqual(_preferred_lib(1.), "numpy")
        self.assertEqual(_preferred_lib(jnp.array(1.), 1.), "jax")

    def test_prefers_jax_sparse_over_numpy(self):
        """Test that it prefers jax_sparse over jax."""
        self.assertEqual(_preferred_lib(np.array(1.)), "numpy")
        self.assertEqual(_preferred_lib(np.array(1.), BCOO.fromdense([1., 2.])), "jax_sparse")

    def test_prefers_jax_sparse_over_jax(self):
        """Test that it prefers jax_sparse over jax."""
        self.assertEqual(_preferred_lib(jnp.array(1.)), "jax")
        self.assertEqual(_preferred_lib(jnp.array(1.), BCOO.fromdense([1., 2.])), "jax_sparse")