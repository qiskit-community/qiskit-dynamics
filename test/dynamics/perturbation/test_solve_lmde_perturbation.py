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
# pylint: disable=unused-argument, invalid-name

"""Tests for solve_lmde_perturbation and related functions."""

import numpy as np

from qiskit import QiskitError

from qiskit_dynamics.array import Array
from qiskit_dynamics.perturbation.solve_lmde_perturbation import solve_lmde_perturbation

from ..common import QiskitDynamicsTestCase, TestJaxBase

try:
    from jax import jit, grad
except ImportError:
    pass


class Testsolve_lmde_perturbation_errors(QiskitDynamicsTestCase):
    """Test cases for argument errors."""

    def test_invalid_method(self):
        """Test error when invalid method is specified."""

        with self.assertRaisesRegex(QiskitError, "not supported"):
            # give a valid order argument
            solve_lmde_perturbation(
                perturbations=[], t_span=[], expansion_method="whoops", expansion_order=1
            )

    def test_no_terms_specified(self):
        """Test error when neither expansion_order or expansion_labels are specified."""

        with self.assertRaisesRegex(QiskitError, "At least one"):
            solve_lmde_perturbation(perturbations=[], t_span=[], expansion_method="dyson")

    def test_non_square_y0(self):
        """Test error when y0 is non-square."""

        with self.assertRaisesRegex(QiskitError, "square"):
            solve_lmde_perturbation(
                perturbations=[],
                t_span=[],
                expansion_method="dyson",
                expansion_order=1,
                y0=np.array([1.0, 0.0]),
            )


class Testsolve_lmde_perturbation(QiskitDynamicsTestCase):
    """Test cases for perturbation theory computation."""

    def setUp(self):
        self.integration_method = "DOP853"

    def test_dyson_analytic_case1(self):
        """Analytic test of computing dyson terms.

        Note: The expected values were computed using a symbolic computation package.
        """

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(t):
            return Array([[0, t], [t**2, 0]], dtype=complex).data

        def A1(t):
            return Array([[t, 0], [0, t**2]], dtype=complex).data

        T = np.pi * 1.2341

        results = solve_lmde_perturbation(
            perturbations=[A0, A1],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="dyson",
            expansion_labels=[[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        T2 = T**2
        T3 = T * T2
        T4 = T * T3
        T5 = T * T4
        T6 = T * T5
        T7 = T * T6
        T8 = T * T7
        T9 = T * T8
        T10 = T * T9
        T11 = T * T10

        expected_D0 = np.array([[0, T2 / 2], [T3 / 3, 0]], dtype=complex)
        expected_D1 = np.array([[T2 / 2, 0], [0, T3 / 3]], dtype=complex)
        expected_D00 = np.array([[T5 / 15, 0], [0, T5 / 10]], dtype=complex)
        expected_D01 = np.array([[0, T5 / 15], [T5 / 10, 0]], dtype=complex)
        expected_D10 = np.array([[0, T4 / 8], [T6 / 18, 0]], dtype=complex)
        expected_D11 = np.array([[T4 / 8, 0], [0, T6 / 18]], dtype=complex)
        expected_D001 = np.array([[T7 / 70, 0], [0, T8 / 120]], dtype=complex)
        expected_D010 = np.array([[T8 / 144, 0], [0, T7 / 56]], dtype=complex)
        expected_D100 = np.array([[T7 / 105, 0], [0, T8 / 80]], dtype=complex)
        expected_D0001 = np.array([[0, T10 / 1200], [T10 / 700, 0]], dtype=complex)
        expected_D0011 = np.array([[T9 / 504, 0], [0, T11 / 1584]], dtype=complex)

        self.assertAllClose(expected_D0, results.perturbation_results[[0]][-1])
        self.assertAllClose(expected_D1, results.perturbation_results[[1]][-1])
        self.assertAllClose(expected_D00, results.perturbation_results[[0, 0]][-1])
        self.assertAllClose(expected_D01, results.perturbation_results[[0, 1]][-1])
        self.assertAllClose(expected_D10, results.perturbation_results[[1, 0]][-1])
        self.assertAllClose(expected_D11, results.perturbation_results[[1, 1]][-1])
        self.assertAllClose(expected_D001, results.perturbation_results[[0, 0, 1]][-1])
        self.assertAllClose(expected_D010, results.perturbation_results[[0, 1, 0]][-1])
        self.assertAllClose(expected_D100, results.perturbation_results[[1, 0, 0]][-1])
        self.assertAllClose(expected_D0001, results.perturbation_results[[0, 0, 0, 1]][-1])
        self.assertAllClose(expected_D0011, results.perturbation_results[[0, 0, 1, 1]][-1])

    def test_dyson_analytic_case1_reduced(self):
        """Analytic test of computing dyson terms, with reduced perturbation terms requested
        (integration test verifying correct construction/solving of a reduced system).

        Note: The expected values were computed using a symbolic computation package.
        """

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(t):
            return Array([[0, t], [t**2, 0]], dtype=complex).data

        def A1(t):
            return Array([[t, 0], [0, t**2]], dtype=complex).data

        T = np.pi * 1.2341

        results = solve_lmde_perturbation(
            perturbations=[A0, A1],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="dyson",
            expansion_labels=[[0, 0, 0, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        T2 = T**2
        T3 = T * T2
        T4 = T * T3
        T5 = T * T4
        T6 = T * T5
        T7 = T * T6
        T8 = T * T7
        T9 = T * T8
        T10 = T * T9

        expected_D1 = np.array([[T2 / 2, 0], [0, T3 / 3]], dtype=complex)
        expected_D01 = np.array([[0, T5 / 15], [T5 / 10, 0]], dtype=complex)
        expected_D001 = np.array([[T7 / 70, 0], [0, T8 / 120]], dtype=complex)
        expected_D0001 = np.array([[0, T10 / 1200], [T10 / 700, 0]], dtype=complex)

        self.assertAllClose(expected_D1, results.perturbation_results[[1]][-1])
        self.assertAllClose(expected_D01, results.perturbation_results[[0, 1]][-1])
        self.assertAllClose(expected_D001, results.perturbation_results[[0, 0, 1]][-1])
        self.assertAllClose(expected_D0001, results.perturbation_results[[0, 0, 0, 1]][-1])

    def test_dyson_semi_analytic_case1(self):
        """Semi-analytic test case 1 for computing dyson terms.

        Note: The expected values were computed using a symbolic computation package,
        though the formulas were too complicated, so they were explicitly evaluated
        (hence the complicated expected values).
        """

        v = 5.0

        def generator(t):
            return -1j * 2 * np.pi * v * Array([[1, 0], [0, -1]], dtype=complex).data / 2

        def A0(t):
            t = Array(t)
            return np.cos(2 * np.pi * v * t) * Array([[0, 1], [1, 0]], dtype=complex).data

        def A1(t):
            t = Array(t)
            return np.sin(2 * np.pi * 1.1 * v * t) * Array([[0, -1j], [1j, 0]], dtype=complex).data

        T = 1.1 / v

        results = solve_lmde_perturbation(
            perturbations=[A0, A1],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="dyson",
            expansion_labels=[
                [0],
                [1],
                [0, 0],
                [1, 0],
                [0, 1],
                [1, 1],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
            ],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        expected_D0 = np.array(
            [
                [0, 0.117568267286407 + 1j * 0.00549866804688611],
                [0.117568267286407 - 1j * 0.00549866804688611, 0],
            ],
            dtype=complex,
        )
        expected_D1 = np.array(
            [
                [0, 0.0944025824472078 - 1j * 0.0468927034685131],
                [0.0944025824472078 + 1j * 0.0468927034685131, 0],
            ],
            dtype=complex,
        )
        expected_D00 = np.array(
            [
                [0.00692626641150889 - 1j * 0.000210272344384338, 0],
                [0, 0.00692626641150889 + 1j * 0.000210272344384338],
            ],
            dtype=complex,
        )
        expected_D01 = np.array(
            [
                [0.00619100126844247 + 1j * 0.001512690629806, 0],
                [0, 0.00619100126844247 - 1j * 0.001512690629806],
            ],
            dtype=complex,
        )
        expected_D10 = np.array(
            [
                [0.00464989936704347 - 1j * 0.00451949172900835, 0],
                [0, 0.00464989936704347 + 1j * 0.00451949172900835],
            ],
            dtype=complex,
        )
        expected_D11 = np.array(
            [
                [0.00555538660564389 - 1j * 0.00040501161659033, 0],
                [0, 0.00555538660564389 + 1j * 0.00040501161659033],
            ],
            dtype=complex,
        )
        expected_D001 = np.array(
            [
                [0, 0.000253187261645855 - 1j * 0.000009578048853540],
                [0.000253187261645855 + 1j * 0.000009578048853540, 0],
            ],
            dtype=complex,
        )
        expected_D010 = np.array(
            [
                [0, 0.000237394765591716 + 1j * 0.000144038559268169],
                [0.000237394765591716 - 1j * 0.000144038559268169, 0],
            ],
            dtype=complex,
        )
        expected_D100 = np.array(
            [
                [0, 0.000180721860772827 - 1j * 0.000170716998071582],
                [0.000180721860772827 + 1j * 0.000170716998071582, 0],
            ],
            dtype=complex,
        )

        self.assertAllClose(
            expected_D0, results.perturbation_results[[0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D1, results.perturbation_results[[1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D00, results.perturbation_results[[0, 0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D01, results.perturbation_results[[0, 1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D10, results.perturbation_results[[1, 0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D11, results.perturbation_results[[1, 1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D001, results.perturbation_results[[0, 0, 1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D010, results.perturbation_results[[0, 1, 0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D100, results.perturbation_results[[1, 0, 0]][-1], rtol=1e-10, atol=1e-10
        )

    def test_symmetric_dyson_analytic_case1(self):
        """Analytic test of computing symmetric dyson terms.

        Notes:
            - The expected values were computed using a symbolic computation package.
            - Expected results for Magnus expansion were computed using explicit integral
              formulas, which is a totally different method from how solve_lmde_perturb
              computes them.
        """

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(t):
            return Array([[0, t], [t**2, 0]], dtype=complex).data

        def A1(t):
            return Array([[t, 0], [0, t**2]], dtype=complex).data

        T = np.pi * 1.2341

        results = solve_lmde_perturbation(
            perturbations=[A0, A1],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="symmetric_dyson",
            expansion_order=2,
            expansion_labels=[[0, 0, 1], [0, 0, 0, 1], [0, 0, 1, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        T2 = T**2
        T3 = T * T2
        T4 = T * T3
        T5 = T * T4
        T6 = T * T5
        T7 = T * T6
        T8 = T * T7
        T9 = T * T8
        T10 = T * T9
        T11 = T * T10

        expected_D0 = np.array([[0, T2 / 2], [T3 / 3, 0]], dtype=complex)
        expected_D1 = np.array([[T2 / 2, 0], [0, T3 / 3]], dtype=complex)
        expected_D00 = np.array([[T5 / 15, 0], [0, T5 / 10]], dtype=complex)
        expected_D01 = np.array([[0, T5 / 15], [T5 / 10, 0]], dtype=complex) + np.array(
            [[0, T4 / 8], [T6 / 18, 0]], dtype=complex
        )
        expected_D11 = np.array([[T4 / 8, 0], [0, T6 / 18]], dtype=complex)
        expected_D001 = (
            np.array([[T7 / 70, 0], [0, T8 / 120]], dtype=complex)
            + np.array([[T8 / 144, 0], [0, T7 / 56]], dtype=complex)
            + np.array([[T7 / 105, 0], [0, T8 / 80]], dtype=complex)
        )
        expected_D0001 = np.array(
            [[0, (T10 / 480) + (T9 / 280)], [(T11 / 720) + (T10 / 420), 0]], dtype=complex
        )
        expected_D0011 = np.array(
            [
                [(T11 / 1782) + 7 * (T10 / 3600) + (T9 / 216), 0],
                [0, (T11 / 396) + 23 * (T10 / 8400) + (T9 / 432)],
            ],
            dtype=complex,
        )

        self.assertAllClose(expected_D0, results.perturbation_results[[0]][-1])
        self.assertAllClose(expected_D1, results.perturbation_results[[1]][-1])
        self.assertAllClose(expected_D00, results.perturbation_results[[0, 0]][-1])
        self.assertAllClose(expected_D01, results.perturbation_results[[0, 1]][-1])
        self.assertAllClose(expected_D11, results.perturbation_results[[1, 1]][-1])
        self.assertAllClose(expected_D001, results.perturbation_results[[0, 0, 1]][-1])
        self.assertAllClose(expected_D0001, results.perturbation_results[[0, 0, 0, 1]][-1])
        self.assertAllClose(expected_D0011, results.perturbation_results[[0, 0, 1, 1]][-1])

    def test_symmetric_magnus_analytic_case1(self):
        """Analytic test of computing symmetric magnus terms.

        Notes:
            - The expected values were computed using a symbolic computation package.
            - Expected results for Magnus expansion were computed using explicit integral
              formulas, which is a totally different method from how solve_lmde_perturb
              computes them.
        """

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(t):
            return Array([[0, t], [t**2, 0]], dtype=complex).data

        def A1(t):
            return Array([[t, 0], [0, t**2]], dtype=complex).data

        T = np.pi * 1.2341

        results = solve_lmde_perturbation(
            perturbations=[A0, A1],
            t_span=[0, T],
            generator=generator,
            expansion_method="symmetric_magnus",
            expansion_labels=[
                [0],
                [1],
                [0, 0],
                [0, 1],
                [1, 1],
                [0, 0, 1],
                [0, 0, 0, 1],
                [0, 0, 1, 1],
            ],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        T2 = T**2
        T3 = T * T2
        T4 = T * T3
        T5 = T * T4
        T6 = T * T5
        T7 = T * T6
        T8 = T * T7
        T9 = T * T8
        T10 = T * T9
        T11 = T * T10

        expected_M0 = np.array([[0, T2 / 2], [T3 / 3, 0]], dtype=complex)
        expected_M1 = np.array([[T2 / 2, 0], [0, T3 / 3]], dtype=complex)
        expected_M00 = 0.5 * (
            np.array([[T5 / 15, 0], [0, T5 / 10]], dtype=complex)
            - np.array([[T5 / 10, 0], [0, T5 / 15]], dtype=complex)
        )
        expected_M01 = 0.5 * (
            np.array([[0, T5 / 15], [T5 / 10, 0]], dtype=complex)
            + np.array([[0, T4 / 8], [T6 / 18, 0]], dtype=complex)
            - np.array([[0, T4 / 8], [T6 / 18, 0]], dtype=complex)
            - np.array([[0, T5 / 10], [T5 / 15, 0]], dtype=complex)
        )
        expected_M11 = np.zeros(2, dtype=complex)
        expected_M001 = (1.0 / 6) * (
            np.array([[T8 / 360, 0], [0, -T8 / 360]], dtype=complex)
            + np.array([[T7 / 840, 0], [0, -T7 / 840]], dtype=complex)
        )
        expected_M0001 = (1.0 / 12) * np.array([[0, T10 / 504], [-T10 / 504, 0]], dtype=complex)
        expected_M0011 = (1.0 / 12) * np.array(
            [
                [(T11 / 3960) - (T10 / 1008) + (T9 / 1260), 0],
                [0, -(T11 / 3960) + (T10 / 1008) - (T9 / 1260)],
            ],
            dtype=complex,
        )

        self.assertAllClose(expected_M0, results.perturbation_results[[0]][-1])
        self.assertAllClose(expected_M1, results.perturbation_results[[1]][-1])
        self.assertAllClose(expected_M00, results.perturbation_results[[0, 0]][-1])
        self.assertAllClose(expected_M01, results.perturbation_results[[0, 1]][-1])
        self.assertAllClose(expected_M11, results.perturbation_results[[1, 1]][-1])
        self.assertAllClose(expected_M001, results.perturbation_results[[0, 0, 1]][-1])
        self.assertAllClose(expected_M0001, results.perturbation_results[[0, 0, 0, 1]][-1])
        self.assertAllClose(expected_M0011, results.perturbation_results[[0, 0, 1, 1]][-1])

    def test_symmetric_dyson_semi_analytic_case1(self):
        """Semi-analytic test case 1 for computing symmetric dyson terms.

        Note: The expected values were computed using a symbolic computation package,
        though the formulas were too complicated, so they were explicitly evaluated
        (hence the complicated expected values).
        """

        v = 5.0

        def generator(t):
            t = Array(t)
            return -1j * 2 * np.pi * v * Array([[1, 0], [0, -1]], dtype=complex) / 2

        def A0(t):
            t = Array(t)
            return np.cos(2 * np.pi * v * t) * Array([[0, 1], [1, 0]], dtype=complex)

        def A1(t):
            t = Array(t)
            return np.sin(2 * np.pi * 1.1 * v * t) * Array([[0, -1j], [1j, 0]], dtype=complex)

        T = 1.1 / v

        results = solve_lmde_perturbation(
            perturbations=[A0, A1],
            t_span=[0, T],
            generator=generator,
            expansion_method="symmetric_dyson",
            expansion_order=3,
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        expected_D0 = np.array(
            [
                [0, 0.117568267286407 + 1j * 0.00549866804688611],
                [0.117568267286407 - 1j * 0.00549866804688611, 0],
            ],
            dtype=complex,
        )
        expected_D1 = np.array(
            [
                [0, 0.0944025824472078 - 1j * 0.0468927034685131],
                [0.0944025824472078 + 1j * 0.0468927034685131, 0],
            ],
            dtype=complex,
        )
        expected_D00 = np.array(
            [
                [0.00692626641150889 - 1j * 0.000210272344384338, 0],
                [0, 0.00692626641150889 + 1j * 0.000210272344384338],
            ],
            dtype=complex,
        )
        expected_D01 = np.array(
            [
                [0.00619100126844247 + 1j * 0.001512690629806, 0],
                [0, 0.00619100126844247 - 1j * 0.001512690629806],
            ],
            dtype=complex,
        ) + np.array(
            [
                [0.00464989936704347 - 1j * 0.00451949172900835, 0],
                [0, 0.00464989936704347 + 1j * 0.00451949172900835],
            ],
            dtype=complex,
        )
        expected_D11 = np.array(
            [
                [0.00555538660564389 - 1j * 0.00040501161659033, 0],
                [0, 0.00555538660564389 + 1j * 0.00040501161659033],
            ],
            dtype=complex,
        )
        expected_D001 = (
            np.array(
                [
                    [0, 0.000253187261645855 - 1j * 0.000009578048853540],
                    [0.000253187261645855 + 1j * 0.000009578048853540, 0],
                ],
                dtype=complex,
            )
            + np.array(
                [
                    [0, 0.000237394765591716 + 1j * 0.000144038559268169],
                    [0.000237394765591716 - 1j * 0.000144038559268169, 0],
                ],
                dtype=complex,
            )
            + np.array(
                [
                    [0, 0.000180721860772827 - 1j * 0.000170716998071582],
                    [0.000180721860772827 + 1j * 0.000170716998071582, 0],
                ],
                dtype=complex,
            )
        )

        self.assertAllClose(
            expected_D0, results.perturbation_results[[0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D1, results.perturbation_results[[1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D00, results.perturbation_results[[0, 0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D01, results.perturbation_results[[1, 0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D11, results.perturbation_results[[1, 1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_D001, results.perturbation_results[[0, 0, 1]][-1], rtol=1e-10, atol=1e-10
        )

    def test_symmetric_magnus_semi_analytic_case1(self):
        """Semi-analytic test case 1 for computing symmetric magnus terms.

        Note: The expected values were computed using a symbolic computation package,
        though the formulas were too complicated, so they were explicitly evaluated
        (hence the complicated expected values).
        """

        v = 5.0

        def generator(t):
            t = Array(t)
            return -1j * 2 * np.pi * v * Array([[1, 0], [0, -1]], dtype=complex) / 2

        def A0(t):
            t = Array(t)
            return np.cos(2 * np.pi * v * t) * Array([[0, 1], [1, 0]], dtype=complex)

        def A1(t):
            t = Array(t)
            return np.sin(2 * np.pi * 1.1 * v * t) * Array([[0, -1j], [1j, 0]], dtype=complex)

        T = 1.1 / v

        results = solve_lmde_perturbation(
            perturbations=[A0, A1],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="symmetric_magnus",
            expansion_order=2,
            expansion_labels=[[0, 0, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        expected_M0 = np.array(
            [
                [0, 0.117568267286407 + 1j * 0.00549866804688611],
                [0.117568267286407 - 1j * 0.00549866804688611, 0],
            ],
            dtype=complex,
        )
        expected_M1 = np.array(
            [
                [0, 0.0944025824472078 - 1j * 0.0468927034685131],
                [0.0944025824472078 + 1j * 0.0468927034685131, 0],
            ],
            dtype=complex,
        )
        expected_M00 = 0.5 * np.array(
            [[-1j * 0.000420544688768675, 0], [0, 1j * 0.000420544688768675]], dtype=complex
        )
        expected_M01 = 0.5 * np.array(
            [[-1j * 0.00601360219840467, 0], [0, 1j * 0.00601360219840467]], dtype=complex
        )
        expected_M11 = 0.5 * np.array(
            [[-1j * 0.000810023233180661, 0], [0, 1j * 0.000810023233180661]], dtype=complex
        )

        self.assertAllClose(
            expected_M0, results.perturbation_results[[0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_M1, results.perturbation_results[[1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_M00, results.perturbation_results[[0, 0]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_M01, results.perturbation_results[[0, 1]][-1], rtol=1e-10, atol=1e-10
        )
        self.assertAllClose(
            expected_M11, results.perturbation_results[[1, 1]][-1], rtol=1e-10, atol=1e-10
        )

    def test_symmetric_dyson_power_series_case1(self):
        """Test consistency of computing power series decompositions across different methods."""

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(t):
            return Array([[0, t], [t**2, 0]], dtype=complex).data

        def A1(t):
            return Array([[t, 0], [0, t**2]], dtype=complex).data

        def A00(t):
            return Array([[1.0, 2.0 * 1j], [3.0 * (t**2), 4.0 * t]], dtype=complex).data

        def A01(t):
            return Array(
                [[4.0, 2.0 * 1j * t], [1j + 3.0 * (t**2), 4.0 * (t**3)]], dtype=complex
            ).data

        def A11(t):
            return Array(
                [[4j + (t + t**2), 2.0 * 1j * t], [1.0 + 3j * (t**2), 1.0 * (t**3)]],
                dtype=complex,
            ).data

        T = np.pi * 1.2341 / 3

        results_dyson = solve_lmde_perturbation(
            perturbations=[A0, A1, A00, A01, A11],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="dyson",
            expansion_labels=[[0], [1], [2], [3], [4], [0, 0], [0, 1], [1, 0], [1, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        results_sym_dyson = solve_lmde_perturbation(
            perturbations=[A0, A1, A00, A01, A11],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="symmetric_dyson",
            expansion_labels=[[0], [1], [2], [3], [4], [0, 0], [0, 1], [1, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        results_sym_dyson_ps = solve_lmde_perturbation(
            perturbations=[A0, A1, A00, A01, A11],
            t_span=[0, T],
            perturbation_labels=[[0], [1], [0, 0], [0, 1], [1, 1]],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="symmetric_dyson",
            expansion_labels=[[0], [1], [0, 0], [0, 1], [1, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        # 1st order consistency
        self.assertAllClose(
            results_dyson.perturbation_results[[0]][-1],
            results_sym_dyson.perturbation_results[[0]][-1],
        )
        self.assertAllClose(
            results_dyson.perturbation_results[[0]][-1],
            results_sym_dyson_ps.perturbation_results[[0]][-1],
        )
        self.assertAllClose(
            results_dyson.perturbation_results[[1]][-1],
            results_sym_dyson.perturbation_results[[1]][-1],
        )
        self.assertAllClose(
            results_dyson.perturbation_results[[1]][-1],
            results_sym_dyson_ps.perturbation_results[[1]][-1],
        )

        # 2nd order consistency
        dyson00 = (
            results_dyson.perturbation_results[[0, 0]][-1]
            + results_dyson.perturbation_results[[2]][-1]
        )
        sym_dyson00 = (
            results_sym_dyson.perturbation_results[[0, 0]][-1]
            + results_sym_dyson.perturbation_results[[2]][-1]
        )
        self.assertAllClose(dyson00, results_sym_dyson_ps.perturbation_results[[0, 0]][-1])
        self.assertAllClose(sym_dyson00, results_sym_dyson_ps.perturbation_results[[0, 0]][-1])

        dyson01 = (
            results_dyson.perturbation_results[[0, 1]][-1]
            + results_dyson.perturbation_results[[1, 0]][-1]
            + results_dyson.perturbation_results[[3]][-1]
        )
        sym_dyson01 = (
            results_sym_dyson.perturbation_results[[0, 1]][-1]
            + results_sym_dyson.perturbation_results[[3]][-1]
        )
        self.assertAllClose(dyson01, results_sym_dyson_ps.perturbation_results[[0, 1]][-1])
        self.assertAllClose(sym_dyson01, results_sym_dyson_ps.perturbation_results[[0, 1]][-1])

        dyson11 = (
            results_dyson.perturbation_results[[1, 1]][-1]
            + results_dyson.perturbation_results[[4]][-1]
        )
        sym_dyson11 = (
            results_sym_dyson.perturbation_results[[1, 1]][-1]
            + results_sym_dyson.perturbation_results[[4]][-1]
        )
        self.assertAllClose(dyson11, results_sym_dyson_ps.perturbation_results[[1, 1]][-1])
        self.assertAllClose(sym_dyson11, results_sym_dyson_ps.perturbation_results[[1, 1]][-1])

    def test_symmetric_dyson_power_series_case2(self):
        """Test consistency of computing power series decompositions across different methods."""

        rng = np.random.default_rng(938122)
        d = 10

        def generator(t):
            return Array(np.eye(d), dtype=complex).data

        A0_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A0_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A0(t):
            return A0_0 + t * A0_1

        A1_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A1_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A1(t):
            return 1j * A1_0 + (t**2) * A1_1

        A2_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A2_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A2(t):
            return 1j * t * A2_0 + (t**3) * A2_1

        A00_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A00_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A00(t):
            return A00_0 * (t**2) + A00_1 * (t**3) * 1j

        A01_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A01_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A01(t):
            return A01_0 * t + A01_1 * (t**4) * 1j

        T = np.pi * 1.2341 / 3

        results_sym_dyson = solve_lmde_perturbation(
            perturbations=[A0, A1, A2, A00, A01],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(d, dtype=complex),
            expansion_method="symmetric_dyson",
            expansion_labels=[[0, 0, 1, 2], [1, 2, 3], [0, 2, 4]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        results_sym_dyson_ps = solve_lmde_perturbation(
            perturbations=[A0, A1, A2, A00, A01],
            t_span=[0, T],
            perturbation_labels=[[0], [1], [2], [0, 0], [0, 1]],
            generator=generator,
            y0=np.eye(d, dtype=complex),
            expansion_method="symmetric_dyson",
            expansion_labels=[[0, 0, 1, 2]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        sym_dyson = (
            results_sym_dyson.perturbation_results[[0, 0, 1, 2]][-1]
            + results_sym_dyson.perturbation_results[[1, 2, 3]][-1]
            + results_sym_dyson.perturbation_results[[0, 2, 4]][-1]
        )
        self.assertAllClose(sym_dyson, results_sym_dyson_ps.perturbation_results[[0, 0, 1, 2]][-1])

    def test_symmetric_magnus_power_series_case1(self):
        """Test consistency of computing power series decompositions across different methods."""

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(t):
            return Array([[0, t], [t**2, 0]], dtype=complex).data

        def A1(t):
            return Array([[t, 0], [0, t**2]], dtype=complex).data

        def A00(t):
            return Array([[1.0, 2.0 * 1j], [3.0 * (t**2), 4.0 * t]], dtype=complex).data

        def A01(t):
            return Array(
                [[4.0, 2.0 * 1j * t], [1j + 3.0 * (t**2), 4.0 * (t**3)]], dtype=complex
            ).data

        def A11(t):
            return Array(
                [[4j + (t + t**2), 2.0 * 1j * t], [1.0 + 3j * (t**2), 1.0 * (t**3)]],
                dtype=complex,
            ).data

        T = np.pi * 1.2341 / 3

        results_sym_magnus = solve_lmde_perturbation(
            perturbations=[A0, A1, A00, A01, A11],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="symmetric_magnus",
            expansion_labels=[[0], [1], [2], [3], [4], [0, 0], [0, 1], [1, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        results_sym_magnus_ps = solve_lmde_perturbation(
            perturbations=[A0, A1, A00, A01, A11],
            t_span=[0, T],
            perturbation_labels=[[0], [1], [0, 0], [0, 1], [1, 1]],
            generator=generator,
            y0=np.eye(2, dtype=complex),
            expansion_method="symmetric_magnus",
            expansion_labels=[[0], [1], [0, 0], [0, 1], [1, 1]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        # 1st order consistency
        self.assertAllClose(
            results_sym_magnus.perturbation_results[[0]][-1],
            results_sym_magnus_ps.perturbation_results[[0]][-1],
        )
        self.assertAllClose(
            results_sym_magnus.perturbation_results[[1]][-1],
            results_sym_magnus_ps.perturbation_results[[1]][-1],
        )

        # 2nd order consistency
        sym_magnus00 = (
            results_sym_magnus.perturbation_results[[0, 0]][-1]
            + results_sym_magnus.perturbation_results[[2]][-1]
        )
        self.assertAllClose(sym_magnus00, results_sym_magnus_ps.perturbation_results[[0, 0]][-1])

        sym_magnus01 = (
            results_sym_magnus.perturbation_results[[0, 1]][-1]
            + results_sym_magnus.perturbation_results[[3]][-1]
        )
        self.assertAllClose(sym_magnus01, results_sym_magnus_ps.perturbation_results[[0, 1]][-1])

        sym_magnus11 = (
            results_sym_magnus.perturbation_results[[1, 1]][-1]
            + results_sym_magnus.perturbation_results[[4]][-1]
        )
        self.assertAllClose(sym_magnus11, results_sym_magnus_ps.perturbation_results[[1, 1]][-1])

    def test_symmetric_magnus_power_series_case2(self):
        """Test consistency of computing power series decompositions across different methods."""

        rng = np.random.default_rng(938122)
        d = 10

        def generator(t):
            return Array(np.eye(d), dtype=complex).data

        A0_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A0_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A0(t):
            return A0_0 + t * A0_1

        A1_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A1_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A1(t):
            return 1j * A1_0 + (t**2) * A1_1

        A2_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A2_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A2(t):
            return 1j * t * A2_0 + (t**3) * A2_1

        A00_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A00_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A00(t):
            return A00_0 * (t**2) + A00_1 * (t**3) * 1j

        A01_0 = Array(rng.uniform(size=(d, d)), dtype=complex).data
        A01_1 = Array(rng.uniform(size=(d, d)), dtype=complex).data

        def A01(t):
            return A01_0 * t + A01_1 * (t**4) * 1j

        T = np.pi * 1.2341 / 3

        results_sym_magnus = solve_lmde_perturbation(
            perturbations=[A0, A1, A2, A00, A01],
            t_span=[0, T],
            generator=generator,
            y0=np.eye(d, dtype=complex),
            expansion_method="symmetric_magnus",
            expansion_labels=[[0, 0, 1, 2], [1, 2, 3], [0, 2, 4]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        results_sym_magnus_ps = solve_lmde_perturbation(
            perturbations=[A0, A1, A2, A00, A01],
            t_span=[0, T],
            perturbation_labels=[[0], [1], [2], [0, 0], [0, 1]],
            generator=generator,
            y0=np.eye(d, dtype=complex),
            expansion_method="symmetric_magnus",
            expansion_labels=[[0, 0, 1, 2]],
            integration_method=self.integration_method,
            atol=1e-13,
            rtol=1e-13,
        )

        sym_magnus = (
            results_sym_magnus.perturbation_results[[0, 0, 1, 2]][-1]
            + results_sym_magnus.perturbation_results[[1, 2, 3]][-1]
            + results_sym_magnus.perturbation_results[[0, 2, 4]][-1]
        )
        self.assertAllClose(
            sym_magnus, results_sym_magnus_ps.perturbation_results[[0, 0, 1, 2]][-1]
        )


class Testsolve_lmde_perturbationJAX(Testsolve_lmde_perturbation, TestJaxBase):
    """Test cases for jax perturbation theory computation."""

    def setUp(self):
        self.integration_method = "jax_odeint"

    def test_jit_grad_dyson(self):
        """Test that we can jit and grad a dyson computation."""

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(a, t):
            return (a**2) * Array([[0, t], [t**2, 0]], dtype=complex).data

        T = np.pi * 1.2341

        def func(a, t):
            results = solve_lmde_perturbation(
                perturbations=[lambda t: A0(a, t)],
                t_span=[0, T],
                generator=generator,
                y0=np.eye(2, dtype=complex),
                expansion_method="dyson",
                expansion_labels=[[0]],
                integration_method=self.integration_method,
                atol=1e-13,
                rtol=1e-13,
            )
            return results.perturbation_results[[0]][-1]

        jit_func = jit(func)

        T2 = T**2
        T3 = T * T2

        expected_D0 = np.array([[0, T2 / 2], [T3 / 3, 0]], dtype=complex)
        output = jit_func(1.0, T)

        self.assertAllClose(expected_D0, output)

        # test grad
        grad_func = jit(grad(lambda a: func(a, T)[0, 1].real))
        output = grad_func(1.0)
        self.assertAllClose(T2, output)

    def test_jit_grad_symmetric(self):
        """Test that we can jit and grad a symmetric Dyson/Magnus computation."""

        def generator(t):
            return Array([[1, 0], [0, 1]], dtype=complex).data

        def A0(a, t):
            return (a**2) * Array([[0, t], [t**2, 0]], dtype=complex).data

        def A1(t):
            return Array([[t, 0], [0, t**2]], dtype=complex).data

        T = np.pi * 1.2341

        def func(a, t):
            results = solve_lmde_perturbation(
                perturbations=[lambda t: A0(a, t), A1],
                t_span=[0, T],
                generator=generator,
                y0=np.eye(2, dtype=complex),
                expansion_method="symmetric_magnus",
                expansion_labels=[[0, 1]],
                integration_method=self.integration_method,
                atol=1e-13,
                rtol=1e-13,
            )
            return results.perturbation_results[[0, 1]][-1]

        jit_func = jit(func)

        T2 = T**2
        T3 = T * T2
        T4 = T * T3
        T5 = T * T4
        T6 = T * T5

        expected_M01 = 0.5 * (
            np.array([[0, T5 / 15], [T5 / 10, 0]], dtype=complex)
            + np.array([[0, T4 / 8], [T6 / 18, 0]], dtype=complex)
            - np.array([[0, T4 / 8], [T6 / 18, 0]], dtype=complex)
            - np.array([[0, T5 / 10], [T5 / 15, 0]], dtype=complex)
        )
        output = jit_func(1.0, T)

        self.assertAllClose(expected_M01, output)

        # test grad
        grad_func = jit(grad(lambda a: func(a, T)[0, 1].real))
        output = grad_func(1.0)
        self.assertAllClose(2 * expected_M01[0, 1].real, output)
