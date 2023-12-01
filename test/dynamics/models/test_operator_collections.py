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
# pylint: disable=invalid-name,no-member

"""Tests for operator_collections.py."""

from functools import partial

import numpy as np
import numpy.random as rand
from scipy.sparse import issparse
from scipy.sparse import csr_matrix

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit_dynamics.models.operator_collections import (
    OperatorCollection,
    ScipySparseOperatorCollection,
    LindbladCollection,
    ScipySparseLindbladCollection,
    VectorizedLindbladCollection,
    ScipySparseVectorizedLindbladCollection,
)
from qiskit_dynamics.array import Array
from qiskit_dynamics.type_utils import to_array
from ..common import test_array_backends, QiskitDynamicsTestCase


@partial(test_array_backends, array_libraries=["numpy", "jax", "jax_sparse"])
class TestOperatorCollection:
    """Test cases for OperatorCollection."""

    def setUp(self):
        """Build a simple OperatorCollection."""
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data

        self.test_operator_list = [self.X, self.Y, self.Z]
        self.simple_collection = OperatorCollection(
            operators=self.test_operator_list,
            static_operator=None,
            array_library=self.array_library(),
        )

    def test_empty_collection_error(self):
        """Verify that evaluating with no operators or static_operator raises an error."""

        collection = OperatorCollection(operators=None, static_operator=None)
        with self.assertRaisesRegex(QiskitError, "cannot be evaluated."):
            collection(None)

    def test_known_values_basic_functionality(self):
        """Test OperatorCollection evaluation against analytically known values."""
        rand.seed(34983)
        coeffs = rand.uniform(-1, 1, 3)

        res = self.simple_collection(coeffs)
        self.assertArrayType(res)
        self.assertAllClose(res, coeffs[0] * self.X + coeffs[1] * self.Y + coeffs[2] * self.Z)

        res = (
            OperatorCollection(
                operators=self.test_operator_list,
                static_operator=np.eye(2),
                array_library=self.array_library(),
            )
        )(coeffs)

        self.assertArrayType(res)
        self.assertAllClose(
            res, np.eye(2) + coeffs[0] * self.X + coeffs[1] * self.Y + coeffs[2] * self.Z
        )

    def test_basic_functionality_pseudorandom(self):
        """Test OperatorCollection evaluation using pseudorandom arrays."""
        rand.seed(342)
        vals = rand.uniform(-1, 1, 32) + 1j * rand.uniform(-1, 1, (10, 32))
        arr = rand.uniform(-1, 1, (32, 128, 128))
        collection = OperatorCollection(operators=arr, array_library=self.array_library())
        for i in range(10):
            res = collection(vals[i])
            self.assertArrayType(res)
            total = 0
            for j in range(32):
                total = total + vals[i, j] * arr[j]
            self.assertAllClose(res, total)

    def test_static_collection(self):
        """Test the case in which only a static operator is present."""
        collection = OperatorCollection(
            operators=None, static_operator=self.X, array_library=self.array_library()
        )
        self.assertAllClose(self.X, collection(None))

    def test_collection_with_no_explicit_array_library(self):
        """Test when array_library is not explicitly passed."""
        rand.seed(342)
        vals = rand.uniform(-1, 1, 32) + 1j * rand.uniform(-1, 1, (10, 32))
        arr = rand.uniform(-1, 1, (32, 128, 128))
        collection = OperatorCollection(operators=self.asarray(arr))
        for i in range(10):
            res = collection(vals[i])
            self.assertArrayType(res)
            total = 0
            for j in range(32):
                total = total + vals[i, j] * arr[j]
            self.assertAllClose(res, total)


@partial(test_array_backends, array_libraries=["jax", "jax_sparse"])
class TestOperatorCollectionJAXTransformations:
    """Test JAX transformations applied to OperatorCollection evaluation methods."""

    def setUp(self):
        """Build simple OperatorCollection instance."""
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data

        self.test_operator_list = [self.X, self.Y, self.Z]
        self.simple_collection = OperatorCollection(
            operators=self.test_operator_list,
            static_operator=None,
            array_library=self.array_library(),
        )

    def test_functions_gradable(self):
        """Tests that all class functions are gradable."""
        doc = OperatorCollection(
            operators=self.test_operator_list,
            static_operator=self.test_operator_list[0],
            array_library=self.array_library(),
        )
        rand.seed(5433)
        coeffs = rand.uniform(-1, 1, 3)
        self.jit_grad(doc.evaluate)(coeffs)
        self.jit_grad(doc.evaluate_rhs)(coeffs, self.X)


class TestScipySparseOperatorCollection(QiskitDynamicsTestCase):
    """Tests for ScipySparseOperatorCollection."""

    def test_empty_collection_error(self):
        """Verify that evaluating with no operators or static_operator raises an error."""

        collection = ScipySparseOperatorCollection(operators=None, static_operator=None)
        with self.assertRaisesRegex(QiskitError, "cannot be evaluated."):
            collection(None)

        with self.assertRaisesRegex(QiskitError, "cannot be evaluated."):
            collection(None, np.array([1.0, 0.0]))

    def test_evaluate_simple_case(self):
        """Simple test case."""

        collection = ScipySparseOperatorCollection(operators=[np.eye(2), [[0.0, 1.0], [1.0, 0.0]]])

        value = collection(np.array([1.0, 2.0]))
        self.assertTrue(issparse(value))
        self.assertAllCloseSparse(value, csr_matrix([[1.0, 2.0], [2.0, 1.0]]))

        # 2d case
        value = collection(np.array([1.0, 2.0]), np.ones((2, 2)))
        self.assertTrue(isinstance(value, (np.ndarray, Array)))
        self.assertAllClose(value, 3.0 * np.ones((2, 2)))

        # 1d case
        value = collection(np.array([1.0, 2.0]), np.array([1.0, 1.0]))
        self.assertTrue(isinstance(value, (np.ndarray, Array)))
        self.assertAllClose(value, np.array([3.0, 3.0]))

    def test_consistency_with_dense_pseudorandom(self):
        """Tests if SparseOperatorCollection agrees with
        the DenseOperatorCollection."""
        r = lambda *args: np.random.uniform(-1, 1, [*args]) + 1j * np.random.uniform(-1, 1, [*args])
        state = r(16)
        mat = r(4, 16, 16)
        sigVals = r(4)
        static_operator = r(16, 16)
        dense_collection = OperatorCollection(operators=mat, static_operator=static_operator)
        sparse_collection = ScipySparseOperatorCollection(
            operators=mat, static_operator=static_operator
        )
        dense_val = dense_collection(sigVals)
        sparse_val = sparse_collection(sigVals)
        self.assertAllClose(dense_val, sparse_val.toarray())
        sparse_val = sparse_collection(sigVals, state)
        self.assertAllClose(dense_val @ state, sparse_val)

    def test_constructor_takes_operators(self):
        """Checks that the SparseOperatorcollection constructor
        is able to convert Operator types to csr_matrix."""
        ham_ops = []
        ham_ops_alt = []
        r = lambda *args: np.random.uniform(-1, 1, [*args]) + 1j * np.random.uniform(-1, 1, [*args])

        for _ in range(4):
            op = r(3, 3)
            ham_ops.append(Operator(op))
            ham_ops_alt.append(Array(op))
        sigVals = r(4)
        static_operator_numpy_array = r(3, 3)
        sparse_collection_operator_list = ScipySparseOperatorCollection(
            operators=ham_ops, static_operator=Operator(static_operator_numpy_array)
        )
        sparse_collection_array_list = ScipySparseOperatorCollection(
            operators=ham_ops_alt, static_operator=to_array(static_operator_numpy_array)
        )
        sparse_collection_pure_array = ScipySparseOperatorCollection(
            operators=to_array(ham_ops), static_operator=to_array(static_operator_numpy_array)
        )
        a = sparse_collection_operator_list(sigVals)
        b = sparse_collection_array_list(sigVals)
        c = sparse_collection_pure_array(sigVals)
        self.assertAllClose(c.toarray(), a.toarray())
        self.assertAllClose(c.toarray(), b.toarray())

    def test_static_collection(self):
        """Test the case in which only a static operator is present."""
        X = csr_matrix([[0.0, 1.0], [1.0, 0.0]])
        collection = ScipySparseOperatorCollection(static_operator=X)
        self.assertAllCloseSparse(X, collection(None))
        self.assertAllClose(np.array([0.0, 1.0]), collection(None, np.array([1.0, 0.0])))


@partial(test_array_backends, array_libraries=["numpy", "jax", "scipy_sparse", "jax_sparse"])
class TestLindbladCollection:
    """Tests for LindbladCollection and ScipySparseLindbladCollection."""

    def setUp(self):
        """Build pseudo-random LindbladCollection instance."""
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data
        rand.seed(2134024)
        n = 16
        k = 8
        m = 4
        l = 2
        self.hamiltonian_operators = rand.uniform(-1, 1, (k, n, n)) + 1j * rand.uniform(
            -1, 1, (k, n, n)
        )
        self.dissipator_operators = rand.uniform(-1, 1, (m, n, n))
        self.static_hamiltonian = rand.uniform(-1, 1, (n, n)) + 1j * rand.uniform(-1, 1, (n, n))
        self.rho = rand.uniform(-1, 1, (n, n)) + 1j * rand.uniform(-1, 1, (n, n))
        self.multiple_rho = rand.uniform(-1, 1, (l, n, n)) + 1j * rand.uniform(-1, 1, (l, n, n))
        self.ham_sig_vals = rand.uniform(-1, 1, (k))
        self.dis_sig_vals = rand.uniform(-1, 1, (m))
        self.r = lambda *args: rand.uniform(-1, 1, args) + 1j * rand.uniform(-1, 1, args)

    def construct_collection(self, *args, **kwargs):
        """Construct collection to be tested by this class
        Used for inheritance.
        """
        if self.array_library() == "scipy_sparse":
            return ScipySparseLindbladCollection(*args, **kwargs)

        return LindbladCollection(*args, **kwargs, array_library=self.array_library())

    def test_empty_collection_error(self):
        """Test errors get raised for empty collection."""
        collection = self.construct_collection()
        with self.assertRaisesRegex(QiskitError, "cannot evaluate rhs"):
            collection(None, None, np.array([[1.0, 0.0], [0.0, 0.0]]))

        with self.assertRaisesRegex(QiskitError, "cannot evaluate Hamiltonian"):
            collection.evaluate_hamiltonian(None)

    def test_no_static_hamiltonian_no_dissipator(self):
        """Test evaluation with just hamiltonian operators."""

        ham_only_collection = self.construct_collection(
            hamiltonian_operators=self.hamiltonian_operators,
            static_hamiltonian=None,
            dissipator_operators=None,
        )
        hamiltonian = np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
        res = ham_only_collection(self.ham_sig_vals, None, self.rho)

        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

    def test_static_hamiltonian_no_dissipator(self):
        """Tests evaluation with a static_hamiltonian and no dissipator."""
        # Now, test adding a static_hamiltonian
        ham_static_hamiltonian_collection = self.construct_collection(
            hamiltonian_operators=self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=None,
        )
        hamiltonian = (
            np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
            + self.static_hamiltonian
        )
        res = ham_static_hamiltonian_collection(self.ham_sig_vals, None, self.rho)
        # In the case of no dissipator terms, expect the Von Neumann equation
        expected = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        self.assertAllClose(res, expected)

    def test_static_hamiltonian_dissipator(self):
        """Tests if providing both static_hamiltonian and dissipator is OK."""
        full_lindblad_collection = self.construct_collection(
            hamiltonian_operators=self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
        )
        res = full_lindblad_collection(self.ham_sig_vals, self.dis_sig_vals, self.rho)
        hamiltonian = (
            np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
            + self.static_hamiltonian
        )
        ham_terms = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        dis_anticommutator = (-1 / 2) * np.tensordot(
            self.dis_sig_vals,
            np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1]))
            @ self.dissipator_operators,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(self.rho) + self.rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            self.dis_sig_vals,
            self.dissipator_operators
            @ self.rho
            @ np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(ham_terms + dis_anticommutator + dis_extra, res)

    def test_full_collection(self):
        """Tests correct evaluation with all terms."""
        full_lindblad_collection = self.construct_collection(
            hamiltonian_operators=self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
        )
        res = full_lindblad_collection(self.ham_sig_vals, self.dis_sig_vals, self.rho)
        hamiltonian = (
            np.tensordot(self.ham_sig_vals, self.hamiltonian_operators, axes=1)
            + self.static_hamiltonian
        )
        ham_terms = -1j * (hamiltonian.dot(self.rho) - self.rho.dot(hamiltonian))
        dis_anticommutator = (-1 / 2) * np.tensordot(
            self.dis_sig_vals,
            np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1]))
            @ self.dissipator_operators,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(self.rho) + self.rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            self.dis_sig_vals,
            self.dissipator_operators
            @ self.rho
            @ np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(ham_terms + dis_anticommutator + dis_extra, res)

    def test_multiple_density_matrix_evaluation(self):
        """Test to ensure that passing multiple density matrices as a (k,n,n) Array functions."""

        # Now, test if vectorization works as intended
        full_lindblad_collection = self.construct_collection(
            hamiltonian_operators=self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
        )
        res = full_lindblad_collection(self.ham_sig_vals, self.dis_sig_vals, self.multiple_rho)
        for i, _ in enumerate(self.multiple_rho):
            self.assertAllClose(
                res[i],
                full_lindblad_collection(
                    self.ham_sig_vals, self.dis_sig_vals, self.multiple_rho[i]
                ),
            )

    def test_static_hamiltonian_only(self):
        """Test construction and evaluation with a static hamiltonian only."""

        collection = self.construct_collection(static_hamiltonian=self.X)

        self.assertAllClose(collection.evaluate_hamiltonian(None), self.X)
        rho = np.array([[1.0, 0.0], [0.0, 0.0]])
        expected = -1j * (self.X @ rho - rho @ self.X)
        self.assertAllClose(collection.evaluate_rhs(None, None, rho), expected)

    def test_dissipators_only(self):
        """Tests correct evaluation with just dissipators."""
        collection = self.construct_collection(
            hamiltonian_operators=None,
            static_hamiltonian=None,
            dissipator_operators=self.dissipator_operators,
        )
        res = collection(None, self.dis_sig_vals, self.rho)
        dis_anticommutator = (-1 / 2) * np.tensordot(
            self.dis_sig_vals,
            np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1]))
            @ self.dissipator_operators,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(self.rho) + self.rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            self.dis_sig_vals,
            self.dissipator_operators
            @ self.rho
            @ np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(dis_anticommutator + dis_extra, res)

    def test_static_dissipator_only(self):
        """Test correct evaluation with just static dissipators."""
        collection = self.construct_collection(static_dissipators=self.dissipator_operators)
        res = collection(None, None, self.rho)
        dis_anticommutator = (-1 / 2) * np.tensordot(
            np.ones_like(self.dis_sig_vals),
            np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1]))
            @ self.dissipator_operators,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(self.rho) + self.rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            np.ones_like(self.dis_sig_vals),
            self.dissipator_operators
            @ self.rho
            @ np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(dis_anticommutator + dis_extra, res)

    def test_both_dissipators(self):
        """Test correct evaluation with both kinds of dissipators."""

        sin_ops = np.sin(self.dissipator_operators)

        collection = self.construct_collection(
            static_dissipators=self.dissipator_operators, dissipator_operators=sin_ops
        )
        res = collection(None, self.dis_sig_vals, self.rho)
        dis_anticommutator = (-1 / 2) * np.tensordot(
            np.ones_like(self.dis_sig_vals),
            np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1]))
            @ self.dissipator_operators,
            axes=1,
        ) + (-1 / 2) * np.tensordot(
            self.dis_sig_vals,
            np.conjugate(np.transpose(sin_ops, [0, 2, 1])) @ sin_ops,
            axes=1,
        )
        dis_anticommutator = dis_anticommutator.dot(self.rho) + self.rho.dot(dis_anticommutator)
        dis_extra = np.tensordot(
            np.ones_like(self.dis_sig_vals),
            self.dissipator_operators
            @ self.rho
            @ np.conjugate(np.transpose(self.dissipator_operators, [0, 2, 1])),
            axes=1,
        ) + np.tensordot(
            self.dis_sig_vals,
            sin_ops @ self.rho @ np.conjugate(np.transpose(sin_ops, [0, 2, 1])),
            axes=1,
        )
        self.assertAllClose(dis_anticommutator + dis_extra, res)

    def test_operator_type_construction(self):
        """Tests if collection can take Operator specification of components."""
        ham_op_terms = []
        ham_ar_terms = []
        dis_op_terms = []
        dis_ar_terms = []
        # pylint: disable=unused-variable
        for i in range(4):
            H_i = self.r(3, 3)
            L_i = self.r(3, 3)
            ham_op_terms.append(Operator(H_i))
            ham_ar_terms.append(H_i)
            dis_op_terms.append(Operator(L_i))
            dis_ar_terms.append(L_i)
        H_d = self.r(3, 3)
        op_static_hamiltonian = Operator(H_d)
        ar_static_hamiltonian = H_d
        op_collection = self.construct_collection(
            hamiltonian_operators=ham_op_terms,
            static_hamiltonian=op_static_hamiltonian,
            dissipator_operators=dis_op_terms,
        )
        ar_collection = self.construct_collection(
            hamiltonian_operators=ham_ar_terms,
            static_hamiltonian=ar_static_hamiltonian,
            dissipator_operators=dis_ar_terms,
        )
        sigVals = self.r(4)
        rho = self.r(3, 3)
        many_rho = self.r(16, 3, 3)
        self.assertAllClose(
            op_collection(sigVals, sigVals, rho), ar_collection(sigVals, sigVals, rho)
        )
        self.assertAllClose(
            op_collection(sigVals, sigVals, many_rho), ar_collection(sigVals, sigVals, many_rho)
        )


@partial(test_array_backends, array_libraries=["jax", "jax_sparse"])
class TestLindbladCollectionJAXTransformations:
    """JAX transformation tests for LindbladCollection."""

    def setUp(self):
        """Build re-usable operators and pseudo-random LindbladCollection instance."""
        self.X = Operator.from_label("X").data
        self.Y = Operator.from_label("Y").data
        self.Z = Operator.from_label("Z").data
        rand.seed(2134024)
        n = 16
        k = 8
        m = 4
        l = 2
        self.hamiltonian_operators = rand.uniform(-1, 1, (k, n, n))
        self.dissipator_operators = rand.uniform(-1, 1, (m, n, n))
        self.static_hamiltonian = rand.uniform(-1, 1, (n, n))
        self.rho = rand.uniform(-1, 1, (n, n))
        self.multiple_rho = rand.uniform(-1, 1, (l, n, n))
        self.ham_sig_vals = rand.uniform(-1, 1, (k))
        self.dis_sig_vals = rand.uniform(-1, 1, (m))
        self.r = lambda *args: rand.uniform(-1, 1, args) + 1j * rand.uniform(-1, 1, args)

    def test_functions_gradable(self):
        """Tests if all class functions are gradable"""
        dlc = LindbladCollection(
            hamiltonian_operators=self.hamiltonian_operators,
            static_hamiltonian=self.static_hamiltonian,
            dissipator_operators=self.dissipator_operators,
            array_library=self.array_library(),
        )
        self.jit_grad(dlc.evaluate_rhs)(self.ham_sig_vals, self.dis_sig_vals, self.rho)
        self.jit_grad(dlc.evaluate_hamiltonian)(self.ham_sig_vals)


@partial(test_array_backends, array_libraries=["numpy", "jax", "jax_sparse", "scipy_sparse"])
class TestVectorizedLindbladCollection:
    """Tests for VectorizedLindbladCollection."""

    def setUp(self) -> None:
        """Build pseudo random operators."""
        rand.seed(123098341)
        n = 16
        k = 4
        m = 2
        r = lambda *args: rand.uniform(-1, 1, [*args]) + 1j * rand.uniform(-1, 1, [*args])

        self.r = r
        self.rand_ham = r(k, n, n)
        self.rand_dis = r(m, n, n)
        self.rand_dft = r(n, n)
        self.rand_static_dis = r(k, n, n)
        self.rho = r(n, n)
        self.t = r()
        self.rand_ham_coeffs = r(k)
        self.rand_dis_coeffs = r(m)

    def _build_vectorized_collection(self, *args, **kwargs):
        if self.array_library() == "scipy_sparse":
            return ScipySparseVectorizedLindbladCollection(*args, **kwargs)
        else:
            return VectorizedLindbladCollection(*args, **kwargs, array_library=self.array_library())

    def _build_non_vectorized_collection(self, *args, **kwargs):
        if self.array_library() == "scipy_sparse":
            return ScipySparseLindbladCollection(*args, **kwargs)
        else:
            return LindbladCollection(*args, **kwargs, array_library=self.array_library())

    def test_empty_collection_error(self):
        """Test errors get raised for empty collection."""
        collection = self._build_vectorized_collection()
        with self.assertRaisesRegex(QiskitError, "OperatorCollection with None"):
            collection(None, None, np.array([[1.0, 0.0], [0.0, 0.0]]))

        with self.assertRaisesRegex(QiskitError, "LindbladCollection with None"):
            collection.evaluate_hamiltonian(None)

    def test_consistency_all_terms(self):
        """Check consistency with non-vectorized class when hamiltonian,
        static_hamiltonian, and dissipator terms defined."""
        self._consistency_test(
            static_hamiltonian=self.rand_dft,
            hamiltonian_operators=self.rand_ham,
            static_dissipators=self.rand_static_dis,
            dissipator_operators=self.rand_dis,
        )

    def test_consistency_no_dissipators(self):
        """Check consistency with non-vectorized class when only hamiltonian and
        static_hamiltonian terms defined.
        """
        self._consistency_test(
            static_hamiltonian=self.rand_dft,
            hamiltonian_operators=self.rand_ham,
            static_dissipators=None,
            dissipator_operators=None,
        )

    def test_consistency_no_static_terms(self):
        """Check consistency with DenseLindbladCollection without static terms."""
        self._consistency_test(
            static_hamiltonian=None,
            hamiltonian_operators=self.rand_ham,
            static_dissipators=None,
            dissipator_operators=self.rand_dis,
        )

    def test_consistency_no_hamiltonian_operators(self):
        """Check consistency with non-vectorized class when hamiltonian,
        static_hamiltonian, static_dissipators, and dissipator terms defined."""
        self._consistency_test(
            static_hamiltonian=self.rand_dft,
            hamiltonian_operators=None,
            static_dissipators=self.rand_static_dis,
            dissipator_operators=self.rand_dis,
        )

    def test_consistency_only_dissipators(self):
        """Check consistency with non-vectorized class when no hamiltonian
        or static_hamiltonian defined."""
        self._consistency_test(
            static_hamiltonian=None,
            hamiltonian_operators=None,
            static_dissipators=self.rand_static_dis,
            dissipator_operators=self.rand_dis,
        )

    def test_consistency_only_static_hamiltonian(self):
        """Check consistency with non-vectorized class when only
        static_hamiltonian defined."""
        self._consistency_test(
            static_hamiltonian=self.rand_dft,
            hamiltonian_operators=None,
            static_dissipators=None,
            dissipator_operators=None,
        )

    def test_consistency_only_hamiltonian_operators(self):
        """Check consistency with non-vectorized class when only hamiltonian operators defined."""
        self._consistency_test(
            static_hamiltonian=None,
            hamiltonian_operators=self.rand_ham,
            static_dissipators=None,
            dissipator_operators=None,
        )

    def test_consistency_only_static_dissipators(self):
        """Check consistency with non-vectorized class when only hamiltonian operators defined."""
        self._consistency_test(
            static_hamiltonian=None,
            hamiltonian_operators=None,
            static_dissipators=self.rand_static_dis,
            dissipator_operators=None,
        )

    def test_consistency_only_static_terms(self):
        """Check consistency with non-vectorized class when only hamiltonian operators defined."""
        self._consistency_test(
            static_hamiltonian=self.rand_dft,
            hamiltonian_operators=None,
            static_dissipators=self.rand_static_dis,
            dissipator_operators=None,
        )

    def _consistency_test(
        self,
        static_hamiltonian=None,
        hamiltonian_operators=None,
        static_dissipators=None,
        dissipator_operators=None,
    ):
        """Consistency test template for non-vectorized class and vectorized class."""

        collection = self._build_non_vectorized_collection(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
        )
        vec_collection = self._build_vectorized_collection(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=hamiltonian_operators,
            static_dissipators=static_dissipators,
            dissipator_operators=dissipator_operators,
        )

        a = collection.evaluate_rhs(self.rand_ham_coeffs, self.rand_dis_coeffs, self.rho).flatten(
            order="F"
        )
        b = vec_collection.evaluate_rhs(
            self.rand_ham_coeffs, self.rand_dis_coeffs, self.rho.flatten(order="F")
        )
        self.assertAllClose(a, b)
