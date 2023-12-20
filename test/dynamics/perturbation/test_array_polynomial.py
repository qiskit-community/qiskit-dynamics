# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name,invalid-unary-operand-type

"""Tests for array_polynomial.py."""

from functools import partial

import numpy as np
from ddt import ddt, data, unpack

from multiset import Multiset

from qiskit import QiskitError

from qiskit_dynamics.perturbation import ArrayPolynomial

from ..common import QiskitDynamicsTestCase, test_array_backends

try:
    from jax import jit, grad
except ImportError:
    pass


@partial(test_array_backends, array_libraries=["numpy", "jax"])
@ddt
class TestArrayPolynomialAlgebra:
    """Test algebraic operations on ArrayPolynomials."""

    def test_addition_type_error(self):
        """Test addition type handling error."""
        ap = ArrayPolynomial(constant_term=np.eye(2))

        with self.assertRaisesRegex(QiskitError, "castable"):
            # pylint: disable=pointless-statement
            ap + {}

    def test_addition_shape_error(self):
        """Test shape broadcasting failure."""
        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(1, 4, 6) + 1j * np.random.rand(1, 4, 6),
            monomial_labels=[[0]],
            constant_term=np.random.rand(4, 6) + 1j * np.random.rand(4, 6),
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )

        with self.assertRaisesRegex(QiskitError, "broadcastable"):
            # pylint: disable=pointless-statement
            ap1 + ap2

    def test_negation(self):
        """Test negation of an array polynomial."""

        ap = ArrayPolynomial(
            array_coefficients=np.random.rand(1, 4, 6) + 1j * np.random.rand(1, 4, 6),
            monomial_labels=[[0]],
            constant_term=np.random.rand(4, 6) + 1j * np.random.rand(4, 6),
            array_library=self.array_library()
        )

        neg_ap = -ap

        self.assertAllClose(neg_ap.constant_term, -ap.constant_term)
        self.assertAllClose(neg_ap.array_coefficients, -ap.array_coefficients)
        self.assertTrue(neg_ap.monomial_labels == ap.monomial_labels)

    def test_addition_only_constant(self):
        """Addition with constant only ArrayPolynomials."""

        result = ArrayPolynomial(constant_term=1.0, array_library=self.array_library()) + ArrayPolynomial(constant_term=2.0, array_library=self.array_library())

        self.assertTrue(result.constant_term == 3.0)
        self.assertTrue(result.monomial_labels == [])
        self.assertTrue(result.array_coefficients is None)

    def test_addition_only_non_constant(self):
        """Addition with ArrayPolynomials with no constant part."""

        ap1 = ArrayPolynomial(monomial_labels=[[0]], array_coefficients=np.array([1.0]), array_library=self.array_library())
        ap2 = ArrayPolynomial(monomial_labels=[[1]], array_coefficients=np.array([2.0]), array_library=self.array_library())
        result = ap1 + ap2

        self.assertTrue(result.constant_term is None)
        self.assertTrue(result.monomial_labels == [Multiset(x) for x in [[0], [1]]])
        self.assertAllClose(result.array_coefficients, np.array([1.0, 2.0]))

    def test_addition_simple(self):
        """Test basic addition."""

        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        result = ap1 + ap2

        self.assertAllClose(
            result.array_coefficients, ap1.array_coefficients + ap2.array_coefficients
        )
        self.assertTrue(result.monomial_labels == ap1.monomial_labels)
        self.assertAllClose(result.constant_term, ap1.constant_term + ap2.constant_term)

    def test_addition_non_overlapping_labels(self):
        """Test non-overlapping labels."""
        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [3], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        result = ap1 + ap2

        expected_coefficients = np.array(
            [
                ap1.array_coefficients[0] + ap2.array_coefficients[0],
                ap1.array_coefficients[1],
                ap1.array_coefficients[2],
                ap2.array_coefficients[1],
                ap2.array_coefficients[2],
            ]
        )
        expected_monomial_labels = [Multiset(l) for l in [[0], [1], [2], [3], [2, 2]]]
        expected_constant_term = ap1.constant_term + ap2.constant_term

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)

    def test_add_scalar(self):
        """Test addition of scalar."""
        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )

        result = ap1 + 1.2
        self.assertAllClose(result.array_coefficients, ap1.array_coefficients)
        self.assertTrue(result.monomial_labels == ap1.monomial_labels)
        self.assertAllClose(result.constant_term, ap1.constant_term + 1.2)

    def test_add_array(self):
        """Test addition of an array."""

        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(1, 2, 2) + 1j * np.random.rand(1, 2, 2),
            monomial_labels=[[0]],
            constant_term=np.random.rand(2, 2) + 1j * np.random.rand(2, 2),
            array_library=self.array_library()
        )

        result = ap1 + np.eye(2)
        self.assertAllClose(result.array_coefficients, ap1.array_coefficients)
        self.assertTrue(result.monomial_labels == ap1.monomial_labels)
        self.assertAllClose(result.constant_term, ap1.constant_term + np.eye(2))

    def test_add_monomial_filter(self):
        """Test adding with a monomial filter."""

        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [3], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        result = ap1.add(ap2, monomial_filter=lambda x: len(x) <= 1)

        expected_coefficients = np.array(
            [
                ap1.array_coefficients[0] + ap2.array_coefficients[0],
                ap1.array_coefficients[1],
                ap1.array_coefficients[2],
                ap2.array_coefficients[1],
            ]
        )
        expected_monomial_labels = [Multiset(l) for l in [[0], [1], [2], [3]]]
        expected_constant_term = ap1.constant_term + ap2.constant_term

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)

    def test_add_monomial_filter_case2(self):
        """Test adding with a monomial filter case 2."""

        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [0, 0, 0], [0, 0, 0, 0]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [3], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )

        ms = Multiset({0: 3})
        monomial_filter = lambda x: len(x) <= 2 or x <= ms
        result = ap1.add(ap2, monomial_filter=monomial_filter)

        expected_coefficients = np.array(
            [
                ap1.array_coefficients[0] + ap2.array_coefficients[0],
                ap2.array_coefficients[1],
                ap2.array_coefficients[2],
                ap1.array_coefficients[1],
            ]
        )
        expected_monomial_labels = [Multiset(l) for l in [[0], [3], [2, 2], [0, 0, 0]]]
        expected_constant_term = ap1.constant_term + ap2.constant_term

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)

    def test_add_monomial_filter_case3(self):
        """Test adding with a monomial filter case 3."""

        ap1 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [0, 0, 0], [0, 0, 0, 0]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [3], [2, 2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )

        ms_list = [Multiset({0: 3})]
        monomial_filter = lambda x: x in ms_list
        result = ap1.add(ap2, monomial_filter=monomial_filter)

        expected_coefficients = np.array([ap1.array_coefficients[1]])
        expected_monomial_labels = [Multiset({0: 3})]

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertTrue(result.constant_term is None)

    @unpack
    @data((lambda A, B: A @ B,), (lambda A, B: A * B,))
    def test_distributive_binary_op(self, binary_op):
        """Test distributive binary operation of two ArrayPolynomials."""

        ap1 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(1, 2, 2),
            monomial_labels=[[0]],
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(1, 2, 2),
            monomial_labels=[[0]],
            array_library=self.array_library()
        )

        result = binary_op(ap1, ap2)
        expected_constant_term = binary_op(ap1.constant_term, ap2.constant_term)
        expected_monomial_labels = [Multiset({0: 1}), Multiset({0: 2})]
        expected_coefficients = np.array(
            [
                binary_op(ap1.constant_term, ap2.array_coefficients[0])
                + binary_op(ap1.array_coefficients[0], ap2.constant_term),
                binary_op(ap1.array_coefficients[0], ap2.array_coefficients[0]),
            ]
        )

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)

    @unpack
    @data((lambda A, B: A @ B,), (lambda A, B: A * B,))
    def test_distributive_binary_op_constant_only(self, binary_op):
        """Test binary_op of two ArrayPolynomials with only constant terms."""

        ap1 = ArrayPolynomial(constant_term=np.random.rand(2, 2), array_library=self.array_library())
        ap2 = ArrayPolynomial(constant_term=np.random.rand(2, 2), array_library=self.array_library())

        result = binary_op(ap1, ap2)
        self.assertAllClose(result.constant_term, binary_op(ap1.constant_term, ap2.constant_term))
        self.assertTrue(result.monomial_labels == [])
        self.assertTrue(result.array_coefficients is None)

    @unpack
    @data((lambda A, B: A @ B,), (lambda A, B: A * B,))
    def test_distributive_binary_op_case2(self, binary_op):
        """Test binary_op of two ArrayPolynomials."""

        ap1 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [1]],
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [0, 0]],
            array_library=self.array_library()
        )

        result = binary_op(ap1, ap2)
        expected_constant_term = binary_op(ap1.constant_term, ap2.constant_term)
        expected_monomial_labels = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 3}),
            Multiset({0: 2, 1: 1}),
        ]
        expected_coefficients = np.array(
            [
                binary_op(ap1.constant_term, ap2.array_coefficients[0])
                + binary_op(ap1.array_coefficients[0], ap2.constant_term),
                binary_op(ap1.array_coefficients[1], ap2.constant_term),
                binary_op(ap1.array_coefficients[0], ap2.array_coefficients[0])
                + binary_op(ap1.constant_term, ap2.array_coefficients[1]),
                binary_op(ap1.array_coefficients[1], ap2.array_coefficients[0]),
                binary_op(ap1.array_coefficients[0], ap2.array_coefficients[1]),
                binary_op(ap1.array_coefficients[1], ap2.array_coefficients[1]),
            ]
        )

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)

    @unpack
    @data((lambda A, B: A @ B,), (lambda A, B: A * B,))
    def test_distributive_binary_op_on_array(self, binary_op):
        """Test distributive binary op directly on an array."""
        ap = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [1]],
            array_library=self.array_library()
        )
        v = np.random.rand(2, 2)

        output = binary_op(ap, v)
        expected_constant_term = binary_op(ap.constant_term, v)
        expected_array_coefficients = binary_op(ap.array_coefficients, v)
        self.assertAllClose(output.constant_term, expected_constant_term)
        self.assertAllClose(output.array_coefficients, expected_array_coefficients)

        output = binary_op(v, ap)
        expected_constant_term = binary_op(v, ap.constant_term)
        expected_array_coefficients = binary_op(v, ap.array_coefficients)
        self.assertAllClose(output.constant_term, expected_constant_term)
        self.assertAllClose(output.array_coefficients, expected_array_coefficients)

    def test_mult_scalar(self):
        """Test multiplication with a scalar."""
        ap = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [1]],
            array_library=self.array_library()
        )

        c = 2.324
        output = c * ap
        expected_constant_term = c * ap.constant_term
        expected_array_coefficients = c * ap.array_coefficients
        self.assertAllClose(output.constant_term, expected_constant_term)
        self.assertAllClose(output.array_coefficients, expected_array_coefficients)

        output = ap * c
        self.assertAllClose(output.constant_term, expected_constant_term)
        self.assertAllClose(output.array_coefficients, expected_array_coefficients)

    def test_matmul_array(self):
        """Test matmul of an ArrayPolynomial with an array."""
        ap = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [1]],
            array_library=self.array_library()
        )
        v = np.random.rand(2)

        output = ap @ v
        expected_constant_term = ap.constant_term @ v
        expected_array_coefficients = ap.array_coefficients @ v
        self.assertAllClose(output.constant_term, expected_constant_term)
        self.assertAllClose(output.array_coefficients, expected_array_coefficients)

        v = np.random.rand(1, 2)

        output = v @ ap
        expected_constant_term = v @ ap.constant_term
        expected_array_coefficients = v @ ap.array_coefficients
        self.assertAllClose(output.constant_term, expected_constant_term)
        self.assertAllClose(output.array_coefficients, expected_array_coefficients)

    @unpack
    @data((lambda A, B: A @ B, "matmul"), (lambda A, B: A * B, "mul"))
    def test_distributive_binary_op_monomial_filter(self, binary_op, method_name):
        """Test distributive binary op with a monomial filter."""

        ap1 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [1]],
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [0, 0]],
            array_library=self.array_library()
        )

        # keep only terms with degree <= 2
        result = getattr(ap1, method_name)(ap2, monomial_filter=lambda x: len(x) <= 2)
        expected_constant_term = binary_op(ap1.constant_term, ap2.constant_term)
        expected_monomial_labels = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
        ]
        expected_coefficients = np.array(
            [
                binary_op(ap1.constant_term, ap2.array_coefficients[0])
                + binary_op(ap1.array_coefficients[0], ap2.constant_term),
                binary_op(ap1.array_coefficients[1], ap2.constant_term),
                binary_op(ap1.array_coefficients[0], ap2.array_coefficients[0])
                + binary_op(ap1.constant_term, ap2.array_coefficients[1]),
                binary_op(ap1.array_coefficients[1], ap2.array_coefficients[0]),
            ]
        )

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)

    @unpack
    @data((lambda A, B: A @ B, "matmul"), (lambda A, B: A * B, "mul"))
    def test_distributive_binary_op_monomial_filter_case2(self, binary_op, method_name):
        """Test distributive binary op with a monomial filter case 2."""

        ap1 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [1]],
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [0, 0]],
            array_library=self.array_library()
        )

        # keep if degree <= 2 or if it is a submultiset of Multiset({0: 3})
        ms = Multiset({0: 3})
        monomial_filter = lambda x: len(x) <= 2 or x <= ms
        result = getattr(ap1, method_name)(ap2, monomial_filter=monomial_filter)
        expected_constant_term = binary_op(ap1.constant_term, ap2.constant_term)
        expected_monomial_labels = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 3}),
        ]
        expected_coefficients = np.array(
            [
                binary_op(ap1.constant_term, ap2.array_coefficients[0])
                + binary_op(ap1.array_coefficients[0], ap2.constant_term),
                binary_op(ap1.array_coefficients[1], ap2.constant_term),
                binary_op(ap1.array_coefficients[0], ap2.array_coefficients[0])
                + binary_op(ap1.constant_term, ap2.array_coefficients[1]),
                binary_op(ap1.array_coefficients[1], ap2.array_coefficients[0]),
                binary_op(ap1.array_coefficients[0], ap2.array_coefficients[1]),
            ]
        )

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertAllClose(result.constant_term, expected_constant_term)

    @unpack
    @data((lambda A, B: A @ B, "matmul"), (lambda A, B: A * B, "mul"))
    def test_distributive_binary_op_monomial_filter_case3(self, binary_op, method_name):
        """Test distributive binary op with a monomial filter case 3."""

        ap1 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [1]],
            array_library=self.array_library()
        )
        ap2 = ArrayPolynomial(
            constant_term=np.random.rand(2, 2),
            array_coefficients=np.random.rand(2, 2, 2),
            monomial_labels=[[0], [0, 0]],
            array_library=self.array_library()
        )

        # keep only a specific set of terms
        ms_list = [Multiset({0: 2, 1: 1})]
        monomial_filter = lambda x: x in ms_list
        result = getattr(ap1, method_name)(ap2, monomial_filter=monomial_filter)
        expected_monomial_labels = [Multiset({0: 2, 1: 1})]
        expected_coefficients = np.array(
            [binary_op(ap1.array_coefficients[1], ap2.array_coefficients[1])]
        )

        self.assertAllClose(result.array_coefficients, expected_coefficients)
        self.assertTrue(result.monomial_labels == expected_monomial_labels)
        self.assertTrue(result.constant_term is None)

@partial(test_array_backends, array_libraries=["jax"])
@ddt
class TestArrayPolynomialAlgebraJAXTransformations:
    """Test JAX transformations through array polynomial algebraic operations."""

    @unpack
    @data((lambda A, B: A + B,), (lambda A, B: A @ B,), (lambda A, B: A * B,))
    def test_jit_grad_alg(self, op):
        """Test that construction and algebraic operations can be differentiated through."""

        # some random matrices
        rand1 = np.random.rand(2, 2)
        rand2 = np.random.rand(2, 2, 2)
        rand3 = np.random.rand(2, 2)
        rand4 = np.random.rand(2, 2, 2)

        def func(a, b, c, d, e):
            ap1 = ArrayPolynomial(
                constant_term=a * rand1,
                array_coefficients=b * rand2,
                monomial_labels=[[0], [1]],
            )
            ap2 = ArrayPolynomial(
                constant_term=c * rand3,
                array_coefficients=d * rand4,
                monomial_labels=[[0], [0, 0]],
            )

            ap3 = op(ap1, ap2)
            return ap3(e).real.sum()

        self.jit_grad(func)(1.0, 2.0, 3.0, 4.0, np.array([5.0, 6.0]))


@partial(test_array_backends, array_libraries=["numpy", "jax"])
class TestArrayPolynomial:
    """Test the ArrayPolynomial class."""

    def setUp(self):
        """Set up typical polynomials including edge cases."""

        self.constant_0d = ArrayPolynomial(constant_term=3.0, array_library=self.array_library())
        self.constant_22d = ArrayPolynomial(constant_term=np.eye(2), array_library=self.array_library())
        self.non_constant_0d = ArrayPolynomial(
            array_coefficients=np.array([1.0, 2.0, 3.0]), monomial_labels=[[0], [1], [2]], array_library=self.array_library()
        )
        self.non_constant_32d = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 3, 2),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]]),
            array_library=self.array_library()
        )
        self.non_constant_complex = ArrayPolynomial(
            array_coefficients=np.random.rand(3, 4, 5) + 1j * np.random.rand(3, 4, 5),
            monomial_labels=[[0], [1], [2]],
            constant_term=np.random.rand(4, 5) + 1j * np.random.rand(4, 5),
            array_library=self.array_library()
        )

    def test_validation_error_no_ops(self):
        """Test validation error when no information specified."""
        with self.assertRaisesRegex(QiskitError, "At least one"):
            ArrayPolynomial()

    def test_validation_non_negative_ints(self):
        """Test validation error if monomial contains something other than non-negative ints."""
        with self.assertRaisesRegex(QiskitError, "non-negative integers"):
            ArrayPolynomial(
                array_coefficients=np.array([0.0, 1.0, 2.0]), monomial_labels=["a", "b", "c"]
            )

    def test_trace_validation(self):
        """Test attempting to trace an AP with ndim < 2 raises an error."""
        with self.assertRaisesRegex(QiskitError, "at least 2."):
            self.non_constant_0d.trace()

    def test_only_constant_term(self):
        """Test construction and evaluation with only a constant term."""
        self.assertAllClose(self.constant_0d(), 3.0)

    def test_shape(self):
        """Test shape property."""
        self.assertTrue(self.constant_22d.shape == (2, 2))
        self.assertTrue(self.non_constant_0d.shape == tuple())
        self.assertTrue(self.non_constant_32d.shape == (3, 2))

    def test_len(self):
        """Test len."""
        self.assertTrue(len(self.constant_0d) == 1)
        self.assertTrue(len(self.non_constant_0d) == 3)
        self.assertTrue(len(self.non_constant_complex) == 4)

    def test_ndim(self):
        """Test ndim."""
        self.assertTrue(self.constant_22d.ndim == 2)
        self.assertTrue(self.non_constant_0d.ndim == 0)
        self.assertTrue(self.non_constant_32d.ndim == 2)

    def test_transpose(self):
        """Test transpose."""
        trans = self.constant_0d.transpose()
        self.assertAllClose(trans.constant_term, 3.0)
        self.assertTrue(trans.array_coefficients is None)
        self.assertTrue(trans.monomial_labels == self.constant_0d.monomial_labels)

        trans = self.non_constant_32d.transpose()
        self.assertAllClose(trans.constant_term, self.non_constant_32d.constant_term.transpose())
        self.assertAllClose(
            trans.array_coefficients, self.non_constant_32d.array_coefficients.transpose((0, 2, 1))
        )
        self.assertTrue(trans.monomial_labels == self.non_constant_32d.monomial_labels)

        trans = self.non_constant_32d.transpose((1, 0))
        self.assertAllClose(trans.constant_term, self.non_constant_32d.constant_term.transpose())
        self.assertAllClose(
            trans.array_coefficients, self.non_constant_32d.array_coefficients.transpose((0, 2, 1))
        )
        self.assertTrue(trans.monomial_labels == self.non_constant_32d.monomial_labels)

    def test_conj(self):
        """Test conj."""
        conj = self.constant_0d.conj()
        self.assertAllClose(conj.constant_term, 3.0)
        self.assertTrue(conj.array_coefficients is None)
        self.assertTrue(conj.monomial_labels == self.constant_0d.monomial_labels)

        conj = self.non_constant_complex.conj()
        self.assertAllClose(conj.constant_term, self.non_constant_complex.constant_term.conj())
        self.assertAllClose(
            conj.array_coefficients, self.non_constant_complex.array_coefficients.conj()
        )
        self.assertTrue(conj.monomial_labels == self.non_constant_complex.monomial_labels)

    def test_trace(self):
        """Test trace."""
        poly_trace = self.non_constant_32d.trace()

        self.assertAllClose(poly_trace.constant_term, self.non_constant_32d.constant_term.trace())
        self.assertAllClose(
            poly_trace.array_coefficients,
            self.non_constant_32d.array_coefficients.trace(axis1=1, axis2=2),
        )
        self.assertTrue(poly_trace.monomial_labels == self.non_constant_32d.monomial_labels)

    def test_sum(self):
        """Test sum."""
        poly_sum = self.non_constant_32d.sum()

        self.assertAllClose(poly_sum.constant_term, self.non_constant_32d.constant_term.sum())
        self.assertAllClose(
            poly_sum.array_coefficients,
            self.non_constant_32d.array_coefficients.sum(axis=(1, 2)),
        )
        self.assertTrue(poly_sum.monomial_labels == self.non_constant_32d.monomial_labels)

    def test_sum_case2(self):
        """Test sum case 2."""
        poly_sum = self.non_constant_32d.sum(axis=0)

        self.assertAllClose(poly_sum.constant_term, self.non_constant_32d.constant_term.sum(axis=0))
        self.assertAllClose(
            poly_sum.array_coefficients,
            self.non_constant_32d.array_coefficients.sum(axis=(1,)),
        )
        self.assertTrue(poly_sum.monomial_labels == self.non_constant_32d.monomial_labels)

    def test_sum_0d(self):
        """Test sum for a 0d case."""
        poly_sum = self.non_constant_0d.sum()

        self.assertAllClose(
            poly_sum.array_coefficients,
            self.non_constant_0d.array_coefficients,
        )
        self.assertTrue(poly_sum.monomial_labels == self.non_constant_0d.monomial_labels)

    def test_real(self):
        """Test taking the real part."""

        ap = ArrayPolynomial(
            constant_term=np.random.rand(2, 2) + 1j * np.random.rand(2, 2),
            array_coefficients=np.random.rand(3, 2, 2) + 1j * np.random.rand(3, 2, 2),
            monomial_labels=[[0], [1], [2]],
            array_library=self.array_library()
        )

        ap_real = ap.real
        self.assertAllClose(ap_real.constant_term, ap.constant_term.real)
        self.assertAllClose(ap_real.array_coefficients, ap.array_coefficients.real)
        self.assertTrue(ap_real.monomial_labels == ap.monomial_labels)

    def test_real_scalar(self):
        """Test taking the real part of a scalar array polynomial."""

        ap = ArrayPolynomial(
            constant_term=np.random.rand() + 1j * np.random.rand(),
            array_coefficients=np.random.rand(3) + 1j * np.random.rand(3),
            monomial_labels=[[0], [1], [2]],
            array_library=self.array_library()
        )

        ap_real = ap.real
        self.assertAllClose(ap_real.constant_term, ap.constant_term.real)
        self.assertAllClose(ap_real.array_coefficients, ap.array_coefficients.real)
        self.assertTrue(ap_real.monomial_labels == ap.monomial_labels)

    def test_call_simple_case(self):
        """Typical expected usage case."""

        rng = np.random.default_rng(18471)
        coeffs = rng.uniform(low=-1, high=1, size=(5, 10, 10))
        monomial_labels = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
        ]

        ap = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=monomial_labels, array_library=self.array_library())

        c = np.array([3.0, 4.0])
        output = ap(c)
        expected = (
            c[0] * coeffs[0]
            + c[1] * coeffs[1]
            + c[0] * c[0] * coeffs[2]
            + c[0] * c[1] * coeffs[3]
            + c[1] * c[1] * coeffs[4]
        )
        self.assertAllClose(expected, output)

        c = np.array([3.2123, 4.1])
        output = ap(c)
        expected = (
            c[0] * coeffs[0]
            + c[1] * coeffs[1]
            + c[0] * c[0] * coeffs[2]
            + c[0] * c[1] * coeffs[3]
            + c[1] * c[1] * coeffs[4]
        )
        self.assertAllClose(expected, output)

    def test_compute_monomials_simple_case(self):
        """Simple test case for compute_monomials."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
            Multiset({0: 3}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(938122)
        c = rng.uniform(size=(2,))

        ap = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list, array_library=self.array_library())

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [c[0], c[1], c[0] * c[0], c[0] * c[1], c[1] * c[1], c[0] * c[0] * c[0]]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_skipped_variable(self):
        """Test compute monomials case with skipped variable."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 2: 1}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(22321)
        c = rng.uniform(size=(3,))

        ap = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list, array_library=self.array_library())

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[2],
                c[0] * c[0],
                c[0] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_medium_case(self):
        """Test compute_monomials medium complexity test case."""
        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 1: 1}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({2: 3}),
            Multiset({0: 3, 1: 1}),
            Multiset({2: 4}),
        ]

        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3,))

        ap = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list, array_library=self.array_library())

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[1],
                c[2],
                c[0] * c[0],
                c[0] * c[1],
                c[0] * c[2],
                c[1] * c[1],
                c[1] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[1],
                c[0] * c[1] * c[2],
                c[2] * c[2] * c[2],
                c[0] * c[0] * c[0] * c[1],
                c[2] * c[2] * c[2] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_vectorized(self):
        """Test vectorized evaluation."""
        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({2: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({0: 1, 2: 1}),
            Multiset({1: 2}),
            Multiset({1: 1, 2: 1}),
            Multiset({2: 2}),
            Multiset({0: 3}),
            Multiset({0: 2, 1: 1}),
            Multiset({0: 1, 1: 1, 2: 1}),
            Multiset({2: 3}),
            Multiset({0: 3, 1: 1}),
            Multiset({2: 4}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)

        rng = np.random.default_rng(23421)
        c = rng.uniform(size=(3, 20))

        ap = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list, array_library=self.array_library())

        output_monomials = ap.compute_monomials(c)
        expected_monomials = np.array(
            [
                c[0],
                c[1],
                c[2],
                c[0] * c[0],
                c[0] * c[1],
                c[0] * c[2],
                c[1] * c[1],
                c[1] * c[2],
                c[2] * c[2],
                c[0] * c[0] * c[0],
                c[0] * c[0] * c[1],
                c[0] * c[1] * c[2],
                c[2] * c[2] * c[2],
                c[0] * c[0] * c[0] * c[1],
                c[2] * c[2] * c[2] * c[2],
            ]
        )
        self.assertAllClose(output_monomials, expected_monomials)

    def test_compute_monomials_only_first_order_terms(self):
        """Test a case with only first order terms."""

        multiset_list = [Multiset({0: 1}), Multiset({1: 1})]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        ap = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list, array_library=self.array_library())

        c = np.array([3.0, 2.0])
        self.assertAllClose(ap.compute_monomials(c), c)

    def test_compute_monomials_incomplete_list(self):
        """Test case where the multiset_list is unordered and incomplete."""

        multiset_list = [Multiset({2: 2}), Multiset({0: 1}), Multiset({1: 1, 2: 1})]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        ap = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list, array_library=self.array_library())

        c = np.array([3.0, 2.0, 4.0])
        self.assertAllClose(ap.compute_monomials(c), np.array([16.0, 3.0, 8.0]))

    def test_get_item(self):
        """Test fancy indexing."""

        ap = self.non_constant_complex
        ap_indexed = ap[0:2, 0:2]

        self.assertAllClose(ap_indexed.constant_term, ap.constant_term[0:2, 0:2])
        self.assertAllClose(ap_indexed.array_coefficients, ap.array_coefficients[:, 0:2, 0:2])
        self.assertTrue(ap_indexed.monomial_labels, ap.monomial_labels)


@partial(test_array_backends, array_libraries=["jax"])
class TestArrayPolynomialJAXTransformations:
    """Test JAX transformations with ArrayPolynomial evaluation."""

    def test_jit_compute_monomials(self):
        """Test jitting works."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
            Multiset({0: 3}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list)

        monomial_function_jit = jit(mp.compute_monomials)

        rng = np.random.default_rng(4122)
        c = rng.uniform(size=(2,))

        self.assertAllClose(mp.compute_monomials(c), monomial_function_jit(c))

    def test_compute_monomials_grad(self):
        """Test grad works."""

        multiset_list = [
            Multiset({0: 1}),
            Multiset({1: 1}),
            Multiset({0: 2}),
            Multiset({0: 1, 1: 1}),
            Multiset({1: 2}),
        ]
        # coeffs don't matter in this case
        coeffs = np.zeros((len(multiset_list), 2, 2), dtype=complex)
        mp = ArrayPolynomial(array_coefficients=coeffs, monomial_labels=multiset_list)

        monomial_function_jit_grad = jit(grad(lambda c: mp.compute_monomials(c).sum()))

        c = np.array([2.0, 3.0])
        expected = np.array([1.0 + 0.0 + 4.0 + 3.0 + 0.0, 0.0 + 1.0 + 0.0 + 2.0 + 6.0])

        self.assertAllClose(expected, monomial_function_jit_grad(c))