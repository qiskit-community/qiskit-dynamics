# -*- coding: utf-8 -*-

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
# pylint: disable=invalid-name

"""
Model class for perturbative expansion of solution of an LMDE.
"""

from typing import Optional, Tuple, Callable, List, Union

import numpy as np
from numpy.polynomial.chebyshev import chebpts1, chebvander, chebval

from multiset import Multiset

from qiskit import QiskitError
from qiskit.quantum_info import Operator

from qiskit_dynamics import Signal, RotatingFrame
from qiskit_dynamics.perturbation import solve_lmde_perturbation, ArrayPolynomial
from qiskit_dynamics.array import Array
from qiskit_dynamics.type_utils import to_array

try:
    import jax.numpy as jnp
    from jax.lax import scan
except ImportError:
    pass


class ExpansionModel:
    """Represents a perturbative expansion of an LMDE."""

    def __init__(
        self,
        operators: List[Operator],
        rotating_frame: Union[Array, Operator, RotatingFrame, None],
        dt: float,
        carrier_freqs: Array,
        chebyshev_orders: List[int],
        expansion_method: Optional[str] = "dyson",
        expansion_order: Optional[int] = None,
        expansion_labels: Optional[List[Multiset]] = None,
        integration_method: Optional[str] = None,
        include_imag: Optional[List[bool]] = None,
        **kwargs,
    ):
        r"""Initialize.

        Args:
            operators: List of constant operators specifying the operators with signal coefficients.
            rotating_frame: Rotating frame to setup the solver in.
                            Must be Hermitian or anti-Hermitian.
            dt: Fixed step size to compile to.
            carrier_freqs: Carrier frequencies of the signals in the generator decomposition.
            chebyshev_orders: Approximation degrees for each signal over the interval [0, dt].
            expansion_method: Either 'dyson' or 'magnus'.
            expansion_order: Order of perturbation terms to compute up to. Specifying this
                                argument results in computation of all terms up to the given order.
                                Can be used in conjunction with ``expansion_terms``.
            expansion_labels: Specific perturbation terms to compute. If both ``expansion_order``
                                and ``expansion_terms`` are specified, then all terms up to
                                ``expansion_order`` are computed, along with the additional terms
                                specified in ``expansion_terms``. Labels are specified either as
                                ``Multiset`` or as valid arguments to the ``Multiset`` constructor.
                                This function further requires that ``Multiset``\s consist only of
                                non-negative integers.
            integration_method: ODE solver method to use when computing perturbation terms.
            include_imag: List of bools determining whether to keep imaginary components in
                          the signal approximation. Defaults to True for all signals.
            kwargs: Additional arguments to pass to the solver when computing perturbation terms.

        Raises:
            QiskitError: if invalid expansion_method passed.
        """
        self._expansion_method = expansion_method

        if expansion_method not in ["dyson", "magnus"]:
            raise QiskitError(
                "_PerturbativeSolver only accepts expansion_method 'dyson' or 'magnus'."
            )

        if include_imag is None:
            include_imag = [True] * len(carrier_freqs)

        # construct signal approximation function
        def collective_dct(signal_list, t0, n_steps):
            return _signal_list_envelope_DCT(
                signal_list,
                reference_freqs=carrier_freqs,
                degrees=chebyshev_orders,
                t0=t0,
                dt=dt,
                n_intervals=n_steps,
                include_imag=include_imag,
            )

        self._approximate_signals = collective_dct

        # setup rotating frame and single time-step frame change operator
        self._rotating_frame = RotatingFrame(rotating_frame)
        operators = to_array(operators)
        self._operators = operators
        self._Udt = Array(
            self.rotating_frame.state_out_of_frame(dt, np.eye(operators[0].shape[0], dtype=complex))
        ).data

        perturbations = None
        # set jax-logic dependent components
        if Array.default_backend() == "jax":
            # compute perturbative terms
            perturbations = _construct_cheb_perturbations_jax(
                operators, chebyshev_orders, carrier_freqs, dt, self.rotating_frame, include_imag
            )
            integration_method = integration_method or "jax_odeint"
        else:
            perturbations = _construct_cheb_perturbations(
                operators, chebyshev_orders, carrier_freqs, dt, self.rotating_frame, include_imag
            )
            integration_method = integration_method or "DOP853"

        self._dt = dt

        # compute perturbative terms
        results = solve_lmde_perturbation(
            perturbations=perturbations,
            t_span=[0, dt],
            expansion_method=expansion_method,
            expansion_order=expansion_order,
            expansion_labels=expansion_labels,
            integration_method=integration_method,
            **kwargs,
        )

        if self.expansion_method == "dyson":
            # if Dyson multiply by single step frame transformation
            results.perturbation_data.data = Array(self.Udt) @ results.perturbation_data.data
            new_coeffs = results.perturbation_data.data[:, -1]
            self._expansion_polynomial = ArrayPolynomial(
                array_coefficients=new_coeffs,
                monomial_labels=results.perturbation_data.labels,
                constant_term=self.Udt,
            )
        elif self.expansion_method == "magnus":
            self._expansion_polynomial = ArrayPolynomial(
                array_coefficients=results.perturbation_data.data[:, -1],
                monomial_labels=results.perturbation_data.labels,
            )

    @property
    def expansion_method(self):
        """Perturbation method used in solver."""
        return self._expansion_method

    @property
    def dt(self):
        """Step size of solver."""
        return self._dt

    @property
    def Udt(self):
        """Single step frame transformation."""
        return self._Udt

    @property
    def operators(self):
        """Original operators in the generator."""
        return self._operators

    @property
    def rotating_frame(self):
        """Rotating frame."""
        return self._rotating_frame

    @property
    def expansion_polynomial(self) -> ArrayPolynomial:
        """ArrayPolynomial object for evaluating the perturbation series."""
        return self._expansion_polynomial

    def approximate_signals(self, signals: List[Signal], t0: float, n_steps: int) -> np.ndarray:
        """Approximate a list of signals over a series of intervals according to the Chebyshev
        time-step and Chebyshev approximation structure set at instantiation.

        Args:
            signals: List of Signals to approximate.
            t0: Start time.
            n_steps: Number of time steps to approximate over.

        Returns:
            np.ndarray: The Chebyshev coefficients of each signal over each time-step.
            The first dimension indexes the signals, and the second the time-step.
        """
        return self._approximate_signals(signals, t0, n_steps)

    def evaluate(self, coeffs: Array) -> Array:
        """Evaluate the expansion polynomial for a given list of coefficients."""
        return self.expansion_polynomial(coeffs)


def _construct_cheb_perturbations(
    operators: np.ndarray,
    chebyshev_orders: List[int],
    carrier_freqs: np.ndarray,
    dt: float,
    rotating_frame: RotatingFrame,
    include_imag: Optional[bool] = None,
) -> List[Callable]:
    r"""Helper function for constructing perturbation terms in the expansions used by
    _PerturbativeSolver.

    Constructs a list of operator-valued functions of the form:

    .. math::
        \cos(2 \pi \nu_j t)T_m(t) G_j\textnormal{, and } \sin(-2 \pi \nu_j t)T_m(t) G_j,

    where :math:`\nu_j` and :math:`G_j` are carrier frequency and operator pairs specified in
    ``operators`` and ``carrier_freqs``, and :math:`m \in [0, \dots, r]` where :math:`r`
    is the corresponding integer in ``chebyshev_orders``. The output list is ordered according
    to the lexicographic ordering of the tuples :math:`(j, m)`, with all cosine terms given
    before sine terms.

    Args:
        operators: List of operators.
        chebyshev_orders: List of chebyshev orders for each operator.
        carrier_freqs: List of frequencies for each operator.
        dt: Interval over which the chebyshev polynomials are defined.
        rotating_frame: Frame the operators are in.
        include_imag: List of bools determining whether to keep imaginary components in
                      the signal approximation. Defaults to True for all signals.
                      Corresponds to whether or not to include the sine terms in this function.

    Returns:
        List of operator-valued functions as described above.
    """

    if include_imag is None:
        include_imag = [True] * len(operators)

    # define functions for constructing perturbations
    def cheb_func(t, deg):
        return _evaluate_cheb_series(t, [0] * deg + [1], domain=[0, dt])

    def get_cheb_func_cos_op(deg, freq, op):
        rad_freq = 2 * np.pi * freq

        def cheb_func_op(t):
            op_in_frame = rotating_frame.operator_into_frame(t, op)
            return cheb_func(t, deg) * np.cos(rad_freq * t) * op_in_frame

        return cheb_func_op

    def get_cheb_func_sin_op(deg, freq, op):
        rad_freq = 2 * np.pi * freq

        def cheb_func_op(t):
            op_in_frame = rotating_frame.operator_into_frame(t, op)
            return cheb_func(t, deg) * np.sin(-rad_freq * t) * op_in_frame

        return cheb_func_op

    # iterate through and construct perturbations list
    perturbations = []
    for deg, op, freq, inc_imag in zip(chebyshev_orders, operators, carrier_freqs, include_imag):
        # construct cosine terms
        for k in range(deg + 1):
            # construct cheb function of appropriate degree
            perturbations.append(get_cheb_func_cos_op(k, freq, op))

        if inc_imag:
            # construct sine terms
            for k in range(deg + 1):
                # construct cheb function of appropriate degree
                perturbations.append(get_cheb_func_sin_op(k, freq, op))

    return perturbations


def _construct_cheb_perturbations_jax(
    operators, chebyshev_orders, carrier_freqs, dt, rotating_frame, include_imag=None
):
    """JAX version of _construct_cheb_perturbations."""

    if include_imag is None:
        include_imag = [True] * len(operators)

    # define functions for constructing perturbations list
    def get_cheb_func(deg):
        c = jnp.array([0] * deg + [1], dtype=float)

        def cheb_func(t):
            return _evaluate_cheb_series_jax(t, c, domain=[0, dt])

        return cheb_func

    def get_cheb_func_cos_op(deg, freq, op):
        rad_freq = 2 * np.pi * freq
        cheb_func = get_cheb_func(deg)

        def cheb_func_op(t):
            op_in_frame = rotating_frame.operator_into_frame(t, op).data
            return cheb_func(t) * jnp.cos(rad_freq * t) * op_in_frame

        return cheb_func_op

    def get_cheb_func_sin_op(deg, freq, op):
        rad_freq = 2 * np.pi * freq
        cheb_func = get_cheb_func(deg)

        def cheb_func_op(t):
            op_in_frame = rotating_frame.operator_into_frame(t, op).data
            return cheb_func(t) * jnp.sin(-rad_freq * t) * op_in_frame

        return cheb_func_op

    # iterate through and construct perturbations list
    perturbations = []
    for deg, op, freq, inc_imag in zip(chebyshev_orders, operators, carrier_freqs, include_imag):
        # construct cosine terms
        for k in range(deg + 1):
            # construct cheb function of appropriate degree
            perturbations.append(get_cheb_func_cos_op(k, freq, op))

        if inc_imag:
            # construct sine terms
            for k in range(deg + 1):
                # construct cheb function of appropriate degree
                perturbations.append(get_cheb_func_sin_op(k, freq, op))

    return perturbations


def _evaluate_cheb_series(
    x: Union[float, np.ndarray], c: np.ndarray, domain: Optional[List] = None
) -> Union[float, np.ndarray]:
    """Evaluate a Chebyshev series on a given domain.

    This calls ``numpy.polynomial.chebyshev.chebval`` but on a stretched domain.

    Args:
        x: Array of x values to evaluate the Chebyshev series on.
        c: Array of Chebyshev coefficients.
        domain: Domain over which the the Chebyshev series is defined. Defaults to [-1, 1].

    Returns:
        array: Chebyshev series evaluated on x.
    """
    domain = domain or [-1, 1]
    x = (2 * x - domain[1] - domain[0]) / (domain[1] - domain[0])
    return chebval(x, c)


def _evaluate_cheb_series_jax(
    x: Union[float, np.ndarray], c: np.ndarray, domain: Optional[List] = None
) -> Union[float, np.ndarray]:
    """Evaluate Chebyshev series on a on a given domain using JAX looping logic.
    This follows the same algorithm as ``numpy.polynomial.chebyshev.chebval``
    but uses JAX looping logic.

    Args:
        x: Array of x values to evaluate the Chebyshev series on.
        c: Array of Chebyshev coefficients.
        domain: Domain over which the the Chebyshev series is defined. Defaults to [-1, 1].

    Returns:
        array: Chebyshev series evaluated on x.
    """
    domain = domain or [-1, 1]

    x = (2 * x - domain[1] - domain[0]) / (domain[1] - domain[0])

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x

        def scan_fun(carry, c_val):
            c0, c1 = carry
            tmp = c0
            c0 = c_val - c1
            c1 = tmp + c1 * x2
            return (c0, c1), None

        init_c1 = c[-1] * jnp.ones_like(x)
        init_c2 = c[-2] * jnp.ones_like(x)
        c0, c1 = scan(scan_fun, init=(init_c2, init_c1), xs=jnp.flip(c)[2:])[0]

    return c0 + c1 * x


def _signal_list_envelope_DCT(
    signal_list: List[Signal],
    reference_freqs: Array,
    degrees: List[int],
    t0: float,
    dt: float,
    n_intervals: int,
    include_imag: Optional[List] = None,
) -> Array:
    """Compute envelope DCT on a list of signals, and compile results into a single list,
    separating real and imaginary.

    Args:
        signal_list: List of signals whose envelopes the DCT will be performed on.
        reference_freqs: Reference frequencies for defining the envelopes.
        degrees: List of Chebyshev degrees for doing the approximation (one for each signal).
        t0: Start time.
        dt: Time step size.
        n_intervals: Number of Intervals.
        include_imag: Whether or not to include the imaginary components of the envelopes. Defaults
                      to ``True`` for all.

    Returns:
        Discrete Chebyshev transform of the envelopes with reference frequencies.
    """

    if include_imag is None:
        include_imag = [True] * len(signal_list)

    envelope_DCT = lambda sig, freq, degree: _signal_envelope_DCT(
        sig, freq, degree, t0, dt, n_intervals
    )

    # initialize coefficient array with first signal
    coeffs = Array(envelope_DCT(signal_list[0], reference_freqs[0], degrees[0]))
    if include_imag[0]:
        coeffs = np.append(coeffs.real, coeffs.imag, axis=0)
    else:
        coeffs = coeffs.real

    for sig, freq, deg, inc_imag in zip(
        signal_list[1:], reference_freqs[1:], degrees[1:], include_imag[1:]
    ):
        new_coeffs = Array(envelope_DCT(sig, freq, deg))

        coeffs = np.append(coeffs, new_coeffs.real, axis=0)
        if inc_imag:
            coeffs = np.append(coeffs, new_coeffs.imag, axis=0)

    return coeffs


def _signal_envelope_DCT(
    signal: Signal, reference_freq: float, degree: int, t0: float, dt: float, n_intervals: int
) -> Array:
    """Perform multi-interval DCT on the envelope of a Signal relative to a reference frequency.
    I.e. This is equivalent to shifting the frequency of the signal to the reference frequency,
    and performing the multi-interval DCT on the resultant envelope.

    Args:
        signal: Signal to approximate.
        reference_freq: Reference frequency to shift the signal frequency to.
        degree: Degree of Chebyshev approximation.
        t0: Start time.
        dt: Length of intervals.
        n_intervals: Number of intervals to perform DCT over.

    Returns:
        Array: Multi-interval DCT stored as a 2d array.
    """

    t_vals = t0 + np.arange(n_intervals) * dt
    phase_arg = -1j * 2 * np.pi * Array(reference_freq)
    final_phase_shift = np.exp(-phase_arg * t_vals)

    def shifted_env(t):
        return signal.complex_value(t) * np.exp(phase_arg * t)

    return _multi_interval_DCT(shifted_env, degree, t0, dt, n_intervals) * np.expand_dims(
        final_phase_shift, axis=0
    )


def _multi_interval_DCT(f: Callable, degree: int, t0: float, dt: float, n_intervals: int) -> Array:
    """Evaluate the multi-interval DCT of a function f over contiguous intervals of size dt
    starting at t0.

    Note: This function assumes that ``f`` is vectorized, and the returned array is indexed as:
        - First axis is coefficient.
        - Second axis is interval.

    Args:
        f: Vectorized function to approximate.
        degree: Degree of Chebyshev approximation.
        t0: Start time.
        dt: Length of intervals.
        n_intervals: Number of intervals to perform DCT over.

    Returns:
        Array: Multi-interval DCT stored as a 2d array.
    """

    dct_mat, xcheb = _construct_DCT(degree, domain=[0, dt])

    # compute all times at which the function needs to be evaluated
    interval_start_times = t0 + np.arange(n_intervals) * dt

    # time values: columns correspond to interval, rows are the shifted chebyshev values
    x_vals = np.add.outer(xcheb, interval_start_times)

    return dct_mat @ f(x_vals)


def _construct_DCT(degree: int, domain: Optional[List] = None) -> Tuple:
    """Construct the matrix and evaluation points for performing the Discrete Chebyshev
    Transform (DCT) over an interval specified by ``domain``. This utilizes code from
    :mod:`numpy.polynomial.chebyshev.chebinterpolate`, but modified to allow for interval
    specification, and to return the constructed matrices and evaluation points for
    performing the DCT.

    I.e. For outputs ``(M, x)``, the DCT of a function ``f`` over the specified domain can be
    evaluated as ``M @ f(x)``.

    Args:
        degree: Degree of Chebyshev approximation.
        domain: Interval of approximation. Defaults to [-1, 1].

    Returns:
        Tuple: Pair of arrays for performing the DCT.
    """
    domain = domain or [-1, 1]
    order = degree + 1

    xcheb = chebpts1(order)
    xcheb_shifted = 0.5 * ((domain[1] - domain[0]) * xcheb + (domain[1] + domain[0]))

    dct_mat = chebvander(xcheb, degree).T
    dct_mat[0] /= order
    dct_mat[1:] /= 0.5 * order

    return dct_mat, xcheb_shifted
