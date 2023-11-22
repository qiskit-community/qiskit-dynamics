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
# pylint: disable=invalid-name

"""tests for rotating_frame.py"""

from functools import partial
import unittest
import warnings

import numpy as np

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from scipy.sparse import csr_matrix
from qiskit_dynamics.models.rotating_frame import RotatingFrame
from qiskit_dynamics.arraylias import DYNAMICS_NUMPY_ALIAS
from qiskit_dynamics.arraylias import DYNAMICS_NUMPY as unp
from ..common import JAXTestBase, NumpyTestBase, test_array_backends

try:
    from jax import jit
    import jax.numpy as jnp
    from jax.experimental.sparse import BCOO

except ImportError:
    pass


# Classes that don't explicitly inherit QiskitDynamicsTestCase get no-member errors
# pylint: disable=no-member


@partial(test_array_backends, array_libraries=["numpy", "jax"])
class TestRotatingFrame:
    """Tests for RotatingFrame."""

    def setUp(self):
        """Setup Pauli operators"""
        self.X = self.asarray(Operator.from_label("X"))
        self.Y = self.asarray(Operator.from_label("Y"))
        self.Z = self.asarray(Operator.from_label("Z"))

    def test_state_out_of_frame_basis(self):
        """Test state_out_of_frame_basis."""

        rng = np.random.default_rng(10933)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = self.asarray(rand_op - rand_op.conj().transpose())
        rotating_frame = RotatingFrame(frame_op)

        _, U = unp.linalg.eigh(1j * frame_op)

        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        val = rotating_frame.state_into_frame_basis(y0)
        expected = U.conj().transpose() @ y0
        self.assertAllClose(val, expected)

        val = rotating_frame.state_out_of_frame_basis(y0)
        expected = U @ y0
        self.assertAllClose(val, expected)

    def test_operator_into_frame_basis(self):
        """Test state_into_frame_basis."""

        rng = np.random.default_rng(98747)
        rand_op = rng.uniform(low=-10, high=10, size=(10, 10)) + 1j * rng.uniform(
            low=-10, high=10, size=(10, 10)
        )

        frame_op = self.asarray(rand_op - rand_op.conj().transpose())
        rotating_frame = RotatingFrame(frame_op)

        _, U = np.linalg.eigh(1j * frame_op)
        Uadj = U.conj().transpose()

        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(10, 10))
            + 1j * rng.uniform(low=-10, high=10, size=(10, 10))
        )

        val = rotating_frame.operator_into_frame_basis(y0)
        expected = U.conj().transpose() @ y0 @ U
        self.assertAllClose(val, expected)

        val = rotating_frame.operator_out_of_frame_basis(y0)
        expected = U @ y0 @ Uadj
        self.assertAllClose(val, expected)

    def test_operator_into_frame_basis_sparse_list(self):
        """Test state_into_frame_basis for a list of sparse arrays."""

        ops = [csr_matrix([[0.0, 1.0], [1.0, 0.0]]), csr_matrix([[1.0, 0.0], [0.0, -1.0]])]
        rotating_frame = RotatingFrame(self.asarray([[0.0, 1.0], [1.0, 0.0]]))
        val = rotating_frame.operator_into_frame_basis(ops)
        U = rotating_frame.frame_basis
        Uadj = rotating_frame.frame_basis_adjoint
        expected = [U @ (op @ Uadj) for op in ops]
        self.assertAllClose(val, expected)

    def test_operator_out_of_frame_basis_sparse_list(self):
        """Test state_out_of_frame_basis for a list of sparse arrays."""

        ops = [csr_matrix([[0.0, 1.0], [1.0, 0.0]]), csr_matrix([[1.0, 0.0], [0.0, -1.0]])]
        rotating_frame = RotatingFrame(self.asarray([[0.0, 1.0], [1.0, 0.0]]))

        val = rotating_frame.operator_out_of_frame_basis(ops)
        U = rotating_frame.frame_basis
        Uadj = rotating_frame.frame_basis_adjoint
        expected = [Uadj @ (op @ U) for op in ops]
        self.assertAllClose(val, expected)

    def test_state_transformations_no_frame(self):
        """Test frame transformations with no frame."""

        rotating_frame = RotatingFrame(self.asarray(np.zeros(2)))

        t = 0.123
        y = self.asarray([1.0, 1j])
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)

        t = 100.12498
        y = self.asarray(np.eye(2))
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)

    def test_state_into_frame_2_level(self):
        """Test state_into_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (self.X + 0.1 * self.Y + 12.0 * self.Z)
        t = 1312.132
        y0 = self.asarray([[1.0, 2.0], [3.0, 4.0]])

        self._test_state_into_frame(t, frame_op, y0)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    def test_state_into_frame_pseudo_random(self):
        """Test state_into_frame with pseudo-random matrices."""
        rng = np.random.default_rng(30493)
        rand_op = rng.uniform(low=-10, high=10, size=(5, 5)) + 1j * rng.uniform(
            low=-10, high=10, size=(5, 5)
        )

        frame_op = self.asarray(rand_op - rand_op.conj().transpose())

        t = 1312.132
        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(5, 5))
            + 1j * rng.uniform(low=-10, high=10, size=(5, 5))
        )

        self._test_state_into_frame(t, frame_op, y0)
        self._test_state_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_state_into_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        evals, U = unp.linalg.eigh(1j * frame_op)
        evals = -1j * evals

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.state_into_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = U.conj().transpose() @ expected

        expected = unp.diag(unp.exp(-t * self.asarray(evals))) @ expected

        if not return_in_frame_basis:
            expected = U @ expected

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_state_out_of_frame_2_level(self):
        """Test state_out_of_frame with a non-trival frame."""
        frame_op = -1j * np.pi * (3.1 * self.X + 1.1 * self.Y + 12.0 * self.Z)
        t = 122.132
        y0 = self.asarray([[1.0, 2.0], [3.0, 4.0]])
        self._test_state_out_of_frame(t, frame_op, y0)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    def test_state_out_of_frame_pseudo_random(self):
        """Test state_out_of_frame with pseudo-random matrices."""
        rng = np.random.default_rng(1382)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = self.asarray(rand_op - rand_op.conj().transpose())

        t = rng.uniform(low=-100, high=100)
        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_state_out_of_frame(t, frame_op, y0)
        self._test_state_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_state_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_state_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_state_out_of_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        evals, U = unp.linalg.eigh(1j * frame_op)
        evals = -1j * self.asarray(evals)

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.state_out_of_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = U.conj().transpose() @ expected

        expected = unp.diag(unp.exp(t * evals)) @ expected

        if not return_in_frame_basis:
            expected = U @ expected

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_operator_into_frame(self):
        """Test operator_into_frame."""
        rng = np.random.default_rng(94994)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = self.asarray(rand_op - rand_op.conj().transpose())

        t = rng.uniform(low=-100, high=100)
        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_operator_into_frame(t, frame_op, y0)
        self._test_operator_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_operator_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_operator_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_operator_into_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        evals, U = unp.linalg.eigh(1j * frame_op)
        evals = -1j * self.asarray(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.operator_into_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = unp.diag(unp.exp(-t * evals)) @ expected @ unp.diag(unp.exp(t * evals))

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_operator_out_of_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(37164093)
        rand_op = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_operator_out_of_frame(t, frame_op, y0)
        self._test_operator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_operator_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_operator_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_operator_out_of_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        evals, U = unp.linalg.eigh(1j * frame_op)
        evals = -1j * self.asarray(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.operator_out_of_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = unp.diag(unp.exp(t * evals)) @ expected @ unp.diag(unp.exp(-t * evals))

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_generator_into_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(111)
        rand_op = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        frame_op = rand_op - rand_op.conj().transpose()

        t = rng.uniform(low=-100, high=100)
        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_generator_into_frame(t, frame_op, y0)
        self._test_generator_into_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_generator_into_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_generator_into_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_generator_into_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        """Helper function for testing generator_into_frame."""
        evals, U = unp.linalg.eigh(1j * frame_op)
        evals = -1j * self.asarray(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.generator_into_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = unp.diag(unp.exp(-t * evals)) @ expected @ unp.diag(unp.exp(t * evals))
        expected = expected - np.diag(evals)

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_generator_out_of_frame(self):
        """Test operator_out_of_frame."""
        rng = np.random.default_rng(111)
        rand_op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        frame_op = self.asarray(rand_op - rand_op.conj().transpose())

        t = rng.uniform(low=-100, high=100)
        y0 = self.asarray(
            rng.uniform(low=-10, high=10, size=(6, 6))
            + 1j * rng.uniform(low=-10, high=10, size=(6, 6))
        )

        self._test_generator_out_of_frame(t, frame_op, y0)
        self._test_generator_out_of_frame(t, frame_op, y0, y_in_frame_basis=True)
        self._test_generator_out_of_frame(t, frame_op, y0, return_in_frame_basis=True)
        self._test_generator_out_of_frame(
            t, frame_op, y0, y_in_frame_basis=True, return_in_frame_basis=True
        )

    # pylint: disable=too-many-arguments
    def _test_generator_out_of_frame(
        self, t, frame_op, y, y_in_frame_basis=False, return_in_frame_basis=False
    ):
        """Helper function for testing generator_into_frame."""
        evals, U = unp.linalg.eigh(1j * frame_op)
        evals = -1j * self.asarray(evals)
        Uadj = U.conj().transpose()

        rotating_frame = RotatingFrame(frame_op)

        value = rotating_frame.generator_out_of_frame(t, y, y_in_frame_basis, return_in_frame_basis)
        expected = y
        if not y_in_frame_basis:
            expected = Uadj @ expected @ U

        expected = unp.diag(unp.exp(t * evals)) @ expected @ unp.diag(unp.exp(-t * evals))
        expected = expected + np.diag(evals)

        if not return_in_frame_basis:
            expected = U @ expected @ Uadj

        self.assertAllClose(value, expected, rtol=1e-10, atol=1e-10)

    def test_vectorized_conjugate_and_add_conventions(self):
        """Test whether passing a vectorized (dim**2, k) operator to _conjugate_and_add
        with vectorized_operators = True is the same as passing a (k,dim,dim) array of
        operators."""
        vectorized_rhos = self.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        ).transpose()

        nonvectord_rhos = vectorized_rhos.reshape((2, 2, 3), order="F").transpose([2, 0, 1])
        rotating_frame = RotatingFrame(self.asarray([1j, 2j]))

        vectorized_result = rotating_frame._conjugate_and_add(
            0.1, vectorized_rhos, vectorized_operators=True
        )
        nonvectord_result = (
            rotating_frame._conjugate_and_add(0.1, nonvectord_rhos)
            .reshape(3, 4, order="F")
            .transpose()
        )
        self.assertAllClose(vectorized_result, nonvectord_result)

    def test_vectorized_frame_basis(self):
        """Test correct lazy evaluation of vectorized_frame_basis."""

        rng = np.random.default_rng(12983)
        F = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )
        F = F - F.conj().transpose()

        rotating_frame = RotatingFrame(F)

        op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        op1 = rotating_frame.operator_into_frame_basis(op)
        op2 = rotating_frame.vectorized_frame_basis_adjoint @ op.flatten(order="F")

        self.assertAllClose(op1, op2.reshape((6, 6), order="F"))

        op = rng.uniform(low=-10, high=10, size=(6, 6)) + 1j * rng.uniform(
            low=-10, high=10, size=(6, 6)
        )

        op1 = rotating_frame.operator_out_of_frame_basis(op)
        op2 = rotating_frame.vectorized_frame_basis @ op.flatten(order="F")

        self.assertAllClose(op1, op2.reshape((6, 6), order="F"))


@partial(test_array_backends, array_libraries=["numpy", "jax"])
class TestRotatingFrameTypeHandling:
    """Type handling testing with rotating frame functions"""

    def test_state_transformations_no_frame_csr_matrix_type(self):
        """Test frame transformations with no frame."""

        rotating_frame = RotatingFrame(None)

        t = 0.123
        y = csr_matrix([1.0, 1j])
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllCloseSparse(out, y)
        self.assertTrue(isinstance(out, csr_matrix))
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllCloseSparse(out, y)
        self.assertTrue(isinstance(out, csr_matrix))


    def test_state_transformations_no_frame_qobj_type(self):
        """Test frame transformations with no frame."""

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=Warning)
                import qutip
        except ImportError:
            return
        rotating_frame = RotatingFrame(None)

        t = 0.123
        y = qutip.Qobj([[1.0, 1]])
        out = rotating_frame.state_into_frame(t, y)
        self.assertTrue(isinstance(out, csr_matrix))
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertTrue(isinstance(out, csr_matrix))

        t = 100.12498
        y = csr_matrix(np.eye(2))
        out = rotating_frame.state_into_frame(t, y)
        self.assertTrue(isinstance(out, csr_matrix))
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertTrue(isinstance(out, csr_matrix))

    def test_state_transformations_no_frame_Operator_types(self):
        """Test frame transformations with no frame."""

        rotating_frame = RotatingFrame(None)

        t = 0.123
        y = Operator([1.0, 1j])
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())

        t = 100.12498
        y = self.asarray(Operator(np.eye(2)))
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())

    def test_state_transformations_no_frame_array_type(self):
        """Test frame transformations with no frame."""

        rotating_frame = RotatingFrame(None)

        t = 0.123
        y = np.asarray([1.0, 1j])
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())

        t = 100.12498
        y = self.asarray(np.eye(2))
        out = rotating_frame.state_into_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())
        out = rotating_frame.state_out_of_frame(t, y)
        self.assertAllClose(out, y)
        self.assertEqual(DYNAMICS_NUMPY_ALIAS.infer_libs(out)[0], self.array_library())


@partial(test_array_backends, array_libraries=["jax"])
class TestRotatingFrameJAXBCOO:
    """Test correct handling of JAX BCOO arrays in relevant functions."""

    def test_conjugate_and_add_BCOO(self):
        """Test _conjugate_and_add with operator being BCOO."""

        rotating_frame = RotatingFrame(self.asarray([1.0, -1.0]))

        t = 0.123
        op = self.asarray([[1.0, -1j], [0.0, 1.0]])
        op_to_add = self.asarray([[0.0, -0.11j], [0.0, 1.0]])
        out = rotating_frame._conjugate_and_add(t, BCOO.fromdense(op), BCOO.fromdense(op_to_add))
        self.assertTrue(type(out).__name__ == "BCOO")

        self.assertAllClose(
            BCOO.todense(out),
            rotating_frame._conjugate_and_add(t, op, op_to_add),
        )

    def test_operator_into_frame_basis(self):
        """Test operator_into_frame_basis with operator being BCOO, for
        frame specified as full matrix.
        """

        rotating_frame = RotatingFrame(self.asarray([[1.0, 0.0], [0.0, -1.0]]))
        op = self.asarray([[1.0, -1j], [0.0, 1.0]])
        output = rotating_frame.operator_into_frame_basis(BCOO.fromdense(op))
        expected = rotating_frame.operator_into_frame_basis(op)

        self.assertAllClose(output, expected)

    def test_operator_out_of_frame_basis(self):
        """Test operator_out_of_frame_basis with operator being BCOO, for
        frame specified as full matrix.
        """

        rotating_frame = RotatingFrame(self.asarray([[1.0, 0.0], [0.0, -1.0]]))

        op = self.asarray([[1.0, -1j], [0.0, 1.0]])
        output = rotating_frame.operator_out_of_frame_basis(BCOO.fromdense(op))
        expected = rotating_frame.operator_out_of_frame_basis(op)

        self.assertAllClose(output, expected)


class TestRotatingFrameNumpy(NumpyTestBase):
    """Tests for RotatingFrameNumpy."""

    def setUp(self):
        self.X = self.asarray(Operator.from_label("X"))
        self.Y = self.asarray(Operator.from_label("Y"))
        self.Z = self.asarray(Operator.from_label("Z"))

    def test_instantiation_errors(self):
        """Check different modes of error raising for frame setting."""

        with self.assertRaisesRegex(QiskitError, "Hermitian or anti-Hermitian"):
            RotatingFrame(self.asarray([1.0, 1j]))

        with self.assertRaisesRegex(QiskitError, "Hermitian or anti-Hermitian"):
            RotatingFrame(self.asarray([[1.0, 0.0], [0.0, 1j]]))

        with self.assertRaisesRegex(QiskitError, "Hermitian or anti-Hermitian"):
            RotatingFrame(self.Z + 1j * self.X)


class TestRotatingFrameJax(JAXTestBase):
    """Jax version of TestRotatingFrame tests.

    Note: This class has more tests due to inheritance.
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

    def setUp(self):
        self.X = self.asarray(Operator.from_label("X"))
        self.Y = self.asarray(Operator.from_label("Y"))
        self.Z = self.asarray(Operator.from_label("Z"))

    def test_instantiation_errors(self):
        """Check different modes of error raising for frame setting.
        Needs to be overwrititen for jax due to different behaviour.
        """

        rotating_frame = RotatingFrame(self.asarray([1.0, 1j]))
        self.assertTrue(jnp.isnan(rotating_frame.frame_diag[0]))

        rotating_frame = RotatingFrame(self.asarray([[1.0, 0.0], [0.0, 1j]]))
        self.assertTrue(jnp.isnan(rotating_frame.frame_diag[0]))

        rotating_frame = RotatingFrame(self.Z + 1j * self.X)
        self.assertTrue(jnp.isnan(rotating_frame.frame_diag[0]))

    def test_jitting(self):
        """Test jitting of state_into_frame and _conjugate_and_add."""

        rotating_frame = RotatingFrame(self.asarray([1.0, -1.0]))

        jit(rotating_frame.state_into_frame)(t=0.1, y=self.asarray([0.0, 1.0]))
        jit(rotating_frame._conjugate_and_add)(
            t=0.1, operator=self.asarray([[0.0, 1.0], [1.0, 0.0]])
        )

    def test_jit_and_grad(self):
        """Test jitting and gradding of state_into_frame and _conjugate_and_add."""

        rotating_frame = RotatingFrame(self.asarray([1.0, -1.0]))

        self.jit_grad(rotating_frame.state_into_frame)(0.1, self.asarray([0.0, 1.0]))
        self.jit_grad(rotating_frame._conjugate_and_add)(
            0.1, self.asarray([[0.0, 1.0], [1.0, 0.0]])
        )
