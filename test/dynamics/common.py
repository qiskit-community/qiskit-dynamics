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
# pylint: disable=invalid-name,isinstance-second-argument-not-valid-type

"""
General infrastructure for unit tests.

``QiskitDynamicsTestCase`` adds general functionality used across tests.

The ``test_array_backends`` decorator, along with the classes with names ``<array_library>TestBase``
enable writing test cases that are agnostic to the underlying array library. The general setup is:
- Each ``<array_library>TestBase`` implements an ``array_library`` class method returning a string,
  an ``asarray`` method for specifically defining arrays for that library, and an
  ``assertArrayType`` method for validating that an array is from that library. These classes can
  also implement any required setup and teardown methods for working with that library (e.g.
  ``JAXTestBase`` skips tests if running on windows).
- Each ``<array_library>TestBase`` subclasses ``QiskitDynamicsTestCase``.
- When used on a given ``test_class``, the decorator ``test_array_backends`` creates a series of
  subclasses inheriting from ``test_class`` and a desired list of ``<array_library>TestBase``. The
  desired list is specified via the ``array_libraries`` argument, which are matched against the
  output of ``<array_library>TestBase.array_library()``. Note that the actual format of the class
  name ``<array_library>TestBase`` is irrelevant: ``test_array_backends`` automatically detects
  classes in this file with ``array_library`` class methods.
Using the above infrastructure, a new array library can be added to the testing infrastructure by
adding a new ``<array_library>TestBase`` class implementing the required methods, and possibly
adding the output of ``<array_library>TestBase.array_library()`` to the default list of array
libraries in ``test_array_backends``.
"""

from typing import Type, Optional, List, Callable, Iterable
import warnings
import unittest
import inspect

import numpy as np
from scipy.sparse import csr_matrix, issparse

try:
    from jax import jit, grad
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

except ImportError:
    pass

from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS

from qiskit_dynamics.array import Array, wrap


def _is_sparse_object_array(A):
    return isinstance(A, np.ndarray) and A.ndim > 0 and issparse(A[0])


class QiskitDynamicsTestCase(unittest.TestCase):
    """Helper class that contains common functionality."""

    def assertAllClose(self, A, B, rtol=1e-8, atol=1e-8):
        """Call np.allclose and assert true."""
        if any(
            "sparse" in x for x in DYNAMICS_NUMPY_ALIAS.infer_libs(A)
        ) or _is_sparse_object_array(A):
            if isinstance(A, list) or _is_sparse_object_array(A):
                A = [x.todense() for x in A]
            else:
                A = A.todense()
        if any(
            "sparse" in x for x in DYNAMICS_NUMPY_ALIAS.infer_libs(B)
        ) or _is_sparse_object_array(B):
            if isinstance(B, list) or _is_sparse_object_array(B):
                B = [x.todense() for x in B]
            else:
                B = B.todense()
        A = np.array(A)
        B = np.array(B)

        self.assertTrue(np.allclose(A, B, rtol=rtol, atol=atol))

    def assertAllCloseSparse(self, A, B, rtol=1e-8, atol=1e-8):
        """Call np.allclose and assert true. Converts A and B to arrays and then calls np.allclose.
        Assumes that A and B are either sparse matrices or lists of sparse matrices"""

        if issparse(A):
            A = A.toarray()
            B = B.toarray()
        elif isinstance(A, Iterable) and issparse(A[0]):
            A = [item.toarray() for item in A]
            B = [item.toarray() for item in B]

        self.assertTrue(np.allclose(A, B, rtol=rtol, atol=atol))


class NumpyTestBase(QiskitDynamicsTestCase):
    """Base class for tests working with numpy arrays."""

    @classmethod
    def array_library(cls):
        """Library method."""
        return "numpy"

    def asarray(self, a):
        """Array generation method."""
        return np.array(a)

    def assertArrayType(self, a):
        """Assert the correct array type."""
        return isinstance(a, np.ndarray)


class JAXTestBase(QiskitDynamicsTestCase):
    """Base class for tests working with JAX arrays."""

    @classmethod
    def setUpClass(cls):
        try:
            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

    @classmethod
    def array_library(cls):
        """Library method."""
        return "jax"

    def asarray(self, a):
        """Array generation method."""
        return jnp.array(a)

    def assertArrayType(self, a):
        """Assert the correct array type."""
        return isinstance(a, jnp.ndarray)

    def jit_grad(self, func_to_test: Callable) -> Callable:
        """Tests whether a function can be graded. Converts
        all functions to scalar, real functions if they are not
        already.
        Args:
            func_to_test: The function whose gradient will be graded.
        Returns:
            JIT-compiled gradient of function.
        """
        return jit(grad(lambda *args: np.sum(func_to_test(*args)).real))


class ScipySparseTestBase(QiskitDynamicsTestCase):
    """Base class for tests working with scipy_sparse arrays."""

    @classmethod
    def array_library(cls):
        """Library method."""
        return "scipy_sparse"

    def asarray(self, a):
        """Array generation method."""
        return csr_matrix(a)

    def assertArrayType(self, a):
        """Assert the correct array type."""
        return issparse(a)


class JAXSparseTestBase(QiskitDynamicsTestCase):
    """Base class for tests working with jax_sparse arrays."""

    @classmethod
    def setUpClass(cls):
        try:
            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

    @classmethod
    def array_library(cls):
        """Library method."""
        return "jax_sparse"

    def asarray(self, a):
        """Array generation method."""
        return BCOO.fromdense(a)

    def assertArrayType(self, a):
        """Assert the correct array type."""
        return type(a).__name__ == "BCOO"

    def jit_grad(self, func_to_test: Callable) -> Callable:
        """Tests whether a function can be graded. Converts
        all functions to scalar, real functions if they are not
        already.
        Args:
            func_to_test: The function whose gradient will be graded.
        Returns:
            JIT-compiled gradient of function.
        """
        return jit(grad(lambda *args: np.sum(func_to_test(*args)).real))


class ArrayNumpyTestBase(QiskitDynamicsTestCase):
    """Base class for tests working with qiskit_dynamics Arrays with numpy backend."""

    @classmethod
    def array_library(cls):
        """Library method."""
        return "array_numpy"

    def asarray(self, a):
        """Array generation method."""
        return Array(a)

    def assertArrayType(self, a):
        """Assert the correct array type."""
        return isinstance(a, Array) and a.backend == "numpy"


class ArrayJaxTestBase(QiskitDynamicsTestCase):
    """Base class for tests working with qiskit_dynamics Arrays with jax backend."""

    @classmethod
    def setUpClass(cls):
        try:
            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

        Array.set_default_backend("jax")

    @classmethod
    def tearDownClass(cls):
        """Set numpy back to the default backend."""
        Array.set_default_backend("numpy")

    @classmethod
    def array_library(cls):
        """Library method."""
        return "array_jax"

    def asarray(self, a):
        """Array generation method."""
        return Array(a)

    def assertArrayType(self, a):
        """Assert the correct array type."""
        return isinstance(a, Array) and a.backend == "jax"


def test_array_backends(test_class: Type, array_libraries: Optional[List[str]] = None):
    """Test class decorator for different array backends.

    Creates subclasses of ``test_class`` with any class in this file implementing an
    ``array_library`` class method whose output matches an entry of ``array_libraries``. These
    classes are added to the calling module, and the original ``test_class`` is deleted. The classes
    in this file implementing ``array_library`` are assumed to be a subclass of
    ``QiskitDynamicsTestCase``, and hence ``test_class`` should not already be a subclass of
    ``QiskitDynamicsTestCase`` or ``unittest.TestCase``.

    Read the file doc string for the intended usage.

    Args:
        test_class: The class to create subclasses from.
        array_libraries: The list of outputs to ``cls.array_library()`` to match when creating
            subclasses.
    Returns:
        None
    """
    if array_libraries is None:
        array_libraries = ["numpy", "jax"]

    # reference to module that called this function
    module = inspect.getmodule(inspect.stack()[1][0])

    # list of classes in this module implementing the array_library method
    classes = inspect.getmembers(inspect.getmodule(inspect.currentframe()), inspect.isclass)
    base_classes = [cls[1] for cls in classes if hasattr(cls[1], "array_library")]

    # create subclasses
    for base_class in base_classes:
        lib = base_class.array_library()
        if lib in array_libraries:
            class_name = f"{test_class.__name__}_{lib}"
            setattr(module, class_name, type(class_name, (test_class, base_class), {}))

    del test_class


class DiffraxTestBase(unittest.TestCase):
    """Base class with setUpClass and tearDownClass for importing diffrax solvers

    Test cases that inherit from this class will automatically work with diffrax solvers
    backend.
    """

    @classmethod
    def setUpClass(cls):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import diffrax  # pylint: disable=import-outside-toplevel,unused-import

            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping diffrax tests.") from err


class QutipTestBase(QiskitDynamicsTestCase):
    """Base class for tests that utilize Qutip."""

    @classmethod
    def setUpClass(cls):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import qutip  # pylint: disable=import-outside-toplevel,unused-import
        except Exception as err:
            raise unittest.SkipTest("Skipping qutip tests.") from err


# to be removed for 0.5.0
class TestJaxBase(unittest.TestCase):
    """Base class with setUpClass and tearDownClass for setting jax as the
    default backend.

    Test cases that inherit from this class will automatically work with jax
    backend.
    """

    @classmethod
    def setUpClass(cls):
        try:
            # pylint: disable=import-outside-toplevel
            import jax

            jax.config.update("jax_enable_x64", True)
            jax.config.update("jax_platform_name", "cpu")
        except Exception as err:
            raise unittest.SkipTest("Skipping jax tests.") from err

        Array.set_default_backend("jax")

    @classmethod
    def tearDownClass(cls):
        """Set numpy back to the default backend."""
        Array.set_default_backend("numpy")

    def jit_wrap(self, func_to_test: Callable) -> Callable:
        """Wraps and jits func_to_test.
        Args:
            func_to_test: The function to be jited.
        Returns:
            Wrapped and jitted function."""
        wf = wrap(jit, decorator=True)
        return wf(wrap(func_to_test))

    def jit_grad_wrap(self, func_to_test: Callable) -> Callable:
        """Tests whether a function can be graded. Converts
        all functions to scalar, real functions if they are not
        already.
        Args:
            func_to_test: The function whose gradient will be graded.
        Returns:
            JIT-compiled gradient of function.
        """
        return wrap(lambda f: jit(grad(f)), decorator=True)(
            lambda *args: np.sum(func_to_test(*args)).real
        )
