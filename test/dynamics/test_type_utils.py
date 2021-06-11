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

"""Tests for type_utils.py."""

import numpy as np
from qiskit_dynamics.dispatch import Array

from qiskit_dynamics.type_utils import convert_state, type_spec_from_instance, StateTypeConverter

from .common import QiskitDynamicsTestCase, TestJaxBase


class TestTypeUtils(QiskitDynamicsTestCase):
    """type_utils.py tests."""

    def test_convert_state_order_C(self):
        """Test convert_state with order parameter 'C'."""

        type_spec = {"type": "array", "shape": (4,)}
        y = Array([[1, 2], [3, 4]])
        expected = Array([1, 2, 3, 4])

        self.assertAllClose(convert_state(y, type_spec, order="C"), expected)

        type_spec = {"type": "array"}
        y = [[1, 2], [3, 4]]
        expected = Array([[1, 2], [3, 4]])

        self.assertAllClose(convert_state(y, type_spec, order="C"), expected)

    def test_convert_state_order_F(self):
        """Test convert_state with order 'F'.

        Note the order='F' is the default so no need to specify
        """

        type_spec = {"type": "array", "shape": (4,)}
        y = Array([[1, 2], [3, 4]])
        expected = Array([1, 3, 2, 4])

        self.assertAllClose(convert_state(y, type_spec), expected)

        type_spec = {"type": "array"}
        y = [[1, 2], [3, 4]]
        expected = Array([[1, 2], [3, 4]])

        self.assertAllClose(convert_state(y, type_spec), expected)

    def test_type_spec_from_instance(self):
        """Test type_spec_from_instance"""

        y = Array([1, 2, 3, 4])
        type_spec = type_spec_from_instance(y)

        self.assertEqual(type_spec, {"type": "array", "shape": (4,)})

        y = Array([[1, 2], [3, 4], [5, 6]])
        type_spec = type_spec_from_instance(y)

        self.assertEqual(type_spec, {"type": "array", "shape": (3, 2)})

    def test_converter_inner_outer_order_C(self):
        """Test standard constructor of StateTypeConverter along with
        basic state conversion functions with array order 'C'.
        """

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec, order="C")

        y_in = Array([1, 2, 3, 4])
        y_out = Array([[1, 2], [3, 4]])

        convert_out = converter.inner_to_outer(y_in)
        convert_in = converter.outer_to_inner(y_out)

        self.assertAllClose(convert_out, y_out)
        self.assertAllClose(convert_in, y_in)

    def test_converter_inner_outer_order_F(self):
        """Test standard constructor of StateTypeConverter along with
        basic state conversion functions with array order 'C'.
        """

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec, order="F")

        y_in = Array([1, 3, 2, 4])
        y_out = Array([[1, 2], [3, 4]])

        convert_out = converter.inner_to_outer(y_in)
        convert_in = converter.outer_to_inner(y_out)

        self.assertAllClose(convert_out, y_out)
        self.assertAllClose(convert_in, y_in)

    def test_from_instances(self):
        """Test from_instances constructor"""

        inner_y = Array([1, 2, 3, 4])
        outer_y = Array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_instances(inner_y, outer_y)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (4,)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (2, 2)})

        converter = StateTypeConverter.from_instances(inner_y)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (4,)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (4,)})

    def test_from_outer_instance_inner_type_spec(self):
        """Test from_outer_instance_inner_type_spec constructor"""

        # test case for inner type spec with 1d array
        inner_type_spec = {"type": "array", "ndim": 1}
        outer_y = Array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (4,)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (2, 2)})

        # inner type spec is a generic array
        inner_type_spec = {"type": "array"}
        outer_y = Array([[1, 2], [3, 4]])

        converter = StateTypeConverter.from_outer_instance_inner_type_spec(outer_y, inner_type_spec)

        self.assertEqual(converter.inner_type_spec, {"type": "array", "shape": (2, 2)})
        self.assertEqual(converter.outer_type_spec, {"type": "array", "shape": (2, 2)})

    def test_transform_rhs_funcs_order_C(self):
        """Test rhs function conversion for order C array reshaping."""

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec, order="C")

        X = Array([[0.0, 1.0], [1.0, 0.0]])

        # do matrix multiplication (a truly '2d' operation)
        def rhs(t, y):
            return t * (y @ y)

        # pylint: disable=unused-argument
        def generator(t):
            return X

        new_rhs = converter.rhs_outer_to_inner(rhs)

        test_t = np.pi
        y_2d = Array([[1, 2], [3, 4]])
        y_1d = y_2d.flatten()

        expected_output = rhs(test_t, y_2d).flatten()
        output = new_rhs(test_t, y_1d)

        self.assertAllClose(output, expected_output)

        new_generator = converter.generator_outer_to_inner(generator)

        # verify generator vectorization
        expected_output = np.kron(X, Array(np.eye(2)))
        output = new_generator(test_t)

        self.assertAllClose(output, expected_output)

    def test_transform_rhs_funcs_order_F(self):
        """Test rhs function conversion"""

        inner_spec = {"type": "array", "shape": (4,)}
        outer_spec = {"type": "array", "shape": (2, 2)}
        converter = StateTypeConverter(inner_spec, outer_spec)

        X = Array([[0.0, 1.0], [1.0, 0.0]])

        # do matrix multiplication (a truly '2d' operation)
        def rhs(t, y):
            return t * (y @ y)

        # pylint: disable=unused-argument
        def generator(t):
            return X

        new_rhs = converter.rhs_outer_to_inner(rhs)

        test_t = np.pi
        y_2d = Array([[1, 2], [3, 4]])
        y_1d = y_2d.flatten(order="F")

        expected_output = rhs(test_t, y_2d).flatten(order="F")
        output = new_rhs(test_t, y_1d)

        self.assertAllClose(output, expected_output)

        new_generator = converter.generator_outer_to_inner(generator)

        # verify generator vectorization
        expected_output = np.kron(Array(np.eye(2)), X)
        output = new_generator(test_t)

        self.assertAllClose(output, expected_output)


class TestTypeUtilsJax(TestTypeUtils, TestJaxBase):
    """Jax version of TestTypeUtils tests.

    Note: This class has no body but contains tests due to inheritance.
    """
