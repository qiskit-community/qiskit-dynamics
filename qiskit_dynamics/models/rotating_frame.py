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

"""
Module for rotating frame handling classes.
"""

from typing import Union, List, Optional
import numpy as np
from scipy.sparse import issparse

from qiskit import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit_dynamics.dispatch import Array
from qiskit_dynamics.type_utils import to_array


class RotatingFrame:
    """
    A 'rotating frame' is given by an anti-Hermitian matrix :math:`F`, specified
    either directly, or in terms of a Hermitian matrix :math:`H` with
    :math:`F = -iH`. Frames have relevance within the context of linear
    matrix differential equations (LMDEs), which are of the form:

    .. math::

        \dot{y}(t) = G(t)y(t).

    For the above DE, 'entering the rotating frame' specified by :math:`F`
    corresponds to a change of state variable
    :math:`y(t) \mapsto z(t) = e^{-tF}y(t)`.
    Using the definition, we may write down a differential equation for
    :math:`z(t)`:

    .. math::

        \dot{z}(t) &= -F z(t) + e^{-tF}G(t)y(t) \\
                   &= (e^{-tF}G(t)e^{tF} - F)z(t)

    In some cases it is computationally easier to solve for :math:`z(t)`
    than it is to solve for :math:`y(t)`.

    While entering a frame is mathematically well-defined for arbitrary
    matrices :math:`F`, this interface assumes that :math:`F` is
    anti-Hermitian, ensuring beneficial properties:

        - :math:`F` is unitarily diagonalizable.
        - :math:`e^{\pm tF}` is easily inverted by taking the adjoint.
        - The frame transformation is norm preserving.

    That :math:`F` is diagonalizable is especially important, as :math:`e^{tF}`
    will need repeated evaluation for different :math:`t` (e.g. at every RHS
    sample point when solving a DE), so it is useful to work in a basis in
    which :math:`F` is diagonal to minimize the cost of this.

    Given an anti-Hermitian matrix :math:`F`, this class offers functions
    for:

        - Bringing a "state" into/out of the frame:
          :math:`t, y \mapsto e^{\mp tF}y`
        - Bringing an "operator" into/out of the frame:
          :math:`t, A \mapsto e^{\mp tF}Ae^{\pm tF}`
        - Bringing a generator for a BMDE into/out of the frame:
          :math:`t, G \mapsto e^{\mp tF}Ge^{\pm tF} - F`

    It also contains functions for bringing states/operators into/out of
    the basis in which :math:`F` is diagonalized, which we refer to as the
    "frame basis". All previously mentioned functions also include optional
    arguments specifying whether the input/output are meant to be in the
    frame basis. This is to facilitate use in solvers in which working
    completely in the frame basis is beneficial to minimize costs associated
    with evaluation of :math:`e^{tF}`.
    """

    def __init__(
        self,
        frame_operator: Union[Array, Operator],
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ):
        """Initialize with a frame operator.

        Args:
            frame_operator: The frame operator, must be either
                            Hermitian or anti-Hermitian.
            atol: Absolute tolerance when verifying that the frame_operator is
                  Hermitian or anti-Hermitian.
            rtol: Relative tolerance when verifying that the frame_operator is
                  Hermitian or anti-Hermitian.
        """
        if isinstance(frame_operator, RotatingFrame):
            frame_operator = frame_operator.frame_operator

        self._frame_operator = frame_operator
        frame_operator = to_array(frame_operator)

        if frame_operator is None:
            self._dim = None
            self._frame_diag = None
            self._frame_basis = None
            self._frame_basis_adjoint = None
        # if frame_operator is a 1d array, assume already diagonalized
        elif frame_operator.ndim == 1:

            # verify Hermitian or anti-Hermitian
            # if Hermitian convert to anti-Hermitian
            frame_operator = _is_herm_or_anti_herm(frame_operator, atol=atol, rtol=rtol)

            self._frame_diag = Array(frame_operator)
            self._frame_basis = None
            self._frame_basis_adjoint = None
            self._dim = len(self._frame_diag)
        # if not, diagonalize it
        else:

            # verify Hermitian or anti-Hermitian
            # if Hermitian convert to anti-Hermitian
            frame_operator = _is_herm_or_anti_herm(frame_operator, atol=atol, rtol=rtol)

            # diagonalize with eigh, utilizing assumption of anti-hermiticity
            frame_diag, frame_basis = np.linalg.eigh(1j * frame_operator)

            self._frame_diag = Array(-1j * frame_diag)
            self._frame_basis = Array(frame_basis)
            self._frame_basis_adjoint = frame_basis.conj().transpose()
            self._dim = len(self._frame_diag)

        # lazily evaluate change-of-basis matrices for vectorized operators.

    @property
    def dim(self) -> int:
        """The dimension of the frame."""
        return self._dim

    @property
    def frame_operator(self) -> Array:
        """The original frame operator."""
        return self._frame_operator

    @property
    def frame_diag(self) -> Array:
        """Diagonal of the frame operator."""
        return self._frame_diag

    @property
    def frame_basis(self) -> Array:
        """Array containing diagonalizing unitary."""
        return self._frame_basis

    @property
    def frame_basis_adjoint(self) -> Array:
        """Adjoint of the diagonalizing unitary."""
        return self._frame_basis_adjoint

    def state_into_frame_basis(self, y: Array) -> Array:
        r"""Take a state into the frame basis, i.e. return
        ``self.frame_basis_adjoint @ y``.

        Args:
            y: The state.
        Returns:
            Array: The state in the frame basis.
        """
        if self.frame_basis_adjoint is None:
            return to_array(y)

        return self.frame_basis_adjoint @ y

    def state_out_of_frame_basis(self, y: Array) -> Array:
        r"""Take a state out of the frame basis, i.e.
        ``return self.frame_basis @ y``.

        Args:
            y: The state.
        Returns:
            Array: The state in the frame basis.
        """
        if self.frame_basis is None:
            return to_array(y)

        return self.frame_basis @ y

    def operator_into_frame_basis(self, op: Union[Operator, List[Operator], Array]) -> Array:
        r"""Take an operator into the frame basis, i.e. return
        ``self.frame_basis_adjoint @ A @ self.frame_basis``

        Args:
            op: The operator or array of operators.
        Returns:
            Array: The operator in the frame basis.
        """
        op = to_array(op)
        if self.frame_basis is None:
            return op
        # parentheses are necessary for sparse op evaluation
        return self.frame_basis_adjoint @ (op @ self.frame_basis)

    def operator_out_of_frame_basis(self, op: Union[Operator, Array]) -> Array:
        r"""Take an operator out of the frame basis, i.e. return
        ``self.frame_basis @ to_array(op) @ self.frame_basis_adjoint``.

        Args:
            op: The operator or array of operators.
        Returns:
            Array: The operator in the frame basis.
        """
        op = to_array(op)
        if self.frame_basis is None:
            return op
        # parentheses are necessary for sparse op evaluation
        return self.frame_basis @ (op @ self.frame_basis_adjoint)

    def state_into_frame(
        self,
        t: float,
        y: Array,
        y_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ):
        """Take a state into the rotating frame, i.e. return exp(-tF) @ y.

        Args:
            t: Time.
            y: State (array of appropriate size).
            y_in_frame_basis: Whether or not the array y is already in
                              the basis in which the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result
                                   in the frame basis.

        Returns:
            Array: State in frame.
        """
        if self._frame_operator is None:
            return to_array(y)

        out = y

        # if not in frame basis convert it
        if not y_in_frame_basis:
            out = self.state_into_frame_basis(out)

        # go into the frame using fast diagonal matrix multiplication
        out = (np.exp(-t * self.frame_diag) * out.transpose()).transpose()  # = e^{tF}out

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.state_out_of_frame_basis(out)

        return out

    def state_out_of_frame(
        self,
        t: float,
        y: Array,
        y_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> Array:
        r"""Take a state out of the rotating frame, i.e. ``return exp(tF) @ y``.

        Calls ``self.state_into_frame`` with time reversed.

        Args:
            t: Time.
            y: State (array of appropriate size).
            y_in_frame_basis: Whether or not the array y is already in
                              the basis in which the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result
                                   in the frame basis.

        Returns:
            Array: State out of frame.
        """
        return self.state_into_frame(-t, y, y_in_frame_basis, return_in_frame_basis)

    def _conjugate_and_add(
        self,
        t: float,
        operator: Array,
        op_to_add_in_fb: Optional[Array] = None,
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ):
        r"""General helper function for computing :math:`\exp(-tF)G\exp(tF) + B`.

        Note: B is added in the frame basis before any potential final change
        out of the frame basis.

        Args: 
            t: Time.
            operator: The operator G.
            op_to_add_in_fb: The additional operator B.
            operator_in_fame_basis: Whether ``operator`` is already in the basis
                in which the frame operator is diagonal. 
            vectorized_operators: Whether ``operator`` is passed as a vectorized,
                ``(dim^2,)`` Array, rather than a ``(dim,dim)`` Array.

        """
        if vectorized_operators:
            # If passing vectorized operator, undo vectorization temporarily
            if self._frame_operator is None:
                if op_to_add_in_fb is None:
                    return to_array(operator)
                else:
                    return to_array(operator + op_to_add_in_fb)
            operator = operator.reshape((self.dim, self.dim) + operator.shape[1:], order="F")

        if self._frame_operator is None:
            if op_to_add_in_fb is None:
                return to_array(operator)
            else:
                return to_array(operator + op_to_add_in_fb)

        out = to_array(operator)

        # if not in frame basis convert it
        if not operator_in_frame_basis:
            out = self.operator_into_frame_basis(out)

        # get frame transformation matrix in diagonal basis
        # assumption that F is anti-Hermitian implies conjugation of
        # diagonal gives inversion
        exp_freq = np.exp(t * self.frame_diag)
        frame_mat = exp_freq.conj().reshape(self.dim, 1) * exp_freq
        if issparse(out):
            out = out.multiply(frame_mat)
        else:
            out = frame_mat * out

        if op_to_add_in_fb is not None:
            out = out + op_to_add_in_fb

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.operator_out_of_frame_basis(out)

        if vectorized_operators:
            # If a vectorized output is required, reshape correctly
            out = out.reshape((self.dim * self.dim,) + out.shape[2:], order="F")

        return out

    def operator_into_frame(
        self,
        t: float,
        operator: Union[Operator, Array],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ) -> Array:
        r"""Bring an operator into the frame, i.e. return
        ``exp(-tF) @ operator @ exp(tF)``.

        Default implementation is to use ``self._conjugate_and_add``.

        Args:
            t: Time.
            operator: Array of appropriate size.
            operator_in_frame_basis: Whether or not the operator is already in
                              the basis in which the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result
                                   in the frame basis.
        Returns:
            Array: operator in frame.
        """
        return self._conjugate_and_add(
            t,
            operator,
            operator_in_frame_basis=operator_in_frame_basis,
            return_in_frame_basis=return_in_frame_basis,
            vectorized_operators=vectorized_operators,
        )

    def operator_out_of_frame(
        self,
        t: float,
        operator: Union[Operator, Array],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ):
        r"""Bring an operator into the rotating frame, i.e. return
        ``exp(tF) @ operator @ exp(-tF)``.

        Default implmentation is to use `self.operator_into_frame`.

        Args:
            t: Time.
            operator: Array of appropriate size.
            operator_in_frame_basis: Whether or not the operator is already in
                              the basis in which the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result
                                   in the frame basis.

        Returns:
            Array: operator out of frame.
        """
        return self.operator_into_frame(
            -t,
            operator,
            operator_in_frame_basis=operator_in_frame_basis,
            return_in_frame_basis=return_in_frame_basis,
            vectorized_operators=vectorized_operators,
        )

    def generator_into_frame(
        self,
        t: float,
        operator: Union[Operator, Array],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ):
        r"""Take an generator into the rotating frame, i.e. return
        ``exp(-tF) @ operator @ exp(tF) - F``.

        Default implementation is to use `self._conjugate_and_add`.

        Args:
            t: Time.
            operator: Generator (array of appropriate size).
            operator_in_frame_basis: Whether or not the generator is already in
                              the basis in which the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result
                                   in the frame basis.

        Returns:
            Array: Generator in frame.
        """
        if self.frame_operator is None:
            return to_array(operator)
        else:
            # conjugate and subtract the frame diagonal
            return self._conjugate_and_add(
                t,
                operator,
                op_to_add_in_fb=-np.diag(self.frame_diag),
                operator_in_frame_basis=operator_in_frame_basis,
                return_in_frame_basis=return_in_frame_basis,
                vectorized_operators=vectorized_operators,
            )

    def generator_out_of_frame(
        self,
        t: float,
        operator: Union[Operator, Array],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> Array:
        r"""Take an operator out of the frame, i.e. return
        ``exp(tF) @ operator @ exp(-tF) + F``.

        Default implementation is to use `self._conjugate_and_add`.

        Args:
            t: Time
            operator: Generator (array of appropriate size).
            operator_in_frame_basis: Whether or not the operator is already in
                              the basis in which the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result
                                   in the frame basis.

        Returns:
            Array: Generator out of frame.
        """
        if self.frame_operator is None:
            return to_array(operator)
        else:
            # conjugate and add the frame diagonal
            return self._conjugate_and_add(
                -t,
                operator,
                op_to_add_in_fb=Array(np.diag(self.frame_diag)),
                operator_in_frame_basis=operator_in_frame_basis,
                return_in_frame_basis=return_in_frame_basis,
            )

    def vectorized_operator_into_frame(
        self,
        time: float,
        op: Array,
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> Array:
        r"""Given a linear map involving left- and right-multiplication vectorized in 
        the column stacking convention which includes the frame shift -F, computes 
        the vectorized operator :math:`B^T\otimes A` in the rotating frame by computing
        .. math::
            (e^{-tF}Be^{tF})^T\otimes(e^{-tF}Ae^{tF}),

        using elementwise multiplication as a faster alternative to actual conjugation.
        This is done by computing :math:`A_{vec}\to \Delta\otimes\bar{\Delta}\circ A_{vec}`,
        where :math:`\Delta_{ij}=\exp((-d_i+d_j)t)` is the frame difference matrix.

        If necessary, also brings the operator into the basis in which the frame operator
        is diagonal through conjugation by :math:`\bar{C}\otimes C`, where ``C = self.frame_basis``. 
        This is much slower than alternatives like pre- and post-rotations, so it should be
        avoided unless putting a vectorized operator into the rotating frame is absolutely
        necessary. 

        Args:
            time: The time t.
            op: The (dim^2,dim^2) Array.
            operator_in_frame_basis: Whether the operator is in the frame basis.
            return_in_frame_basis: Whether the operator should be returned in the
                frame basis.
        Returns:
            op in the frame.
        """
        if self.frame_diag is not None:
            # Put the vectorized operator into the frame basis
            if not operator_in_frame_basis and self.frame_basis is not None:
                if self._cached_c_bar_otimes_c is None:
                    self._cached_c_bar_otimes_c = np.kron(self.frame_basis.conj(), self.frame_basis)
                    self._cached_c_trans_otimes_c_dagg = np.kron(
                        self.frame_basis.T, self.frame_basis_adjoint
                    )
                op = self._cached_c_trans_otimes_c_dagg @ op @ self._cached_c_bar_otimes_c

            expvals = np.exp(self.frame_diag * time)  # = e^{td_i} = e^{it*Im(d_i)}
            # = kron(e^{-it*Im(d_i)},e^{it*Im(d_i)}), but ~3x faster
            temp_outer = (expvals.conj().reshape(self.dim, 1) * expvals).flatten()
            delta_bar_otimes_delta = np.outer(
                temp_outer.conj(), temp_outer
            )  # = kron(delta.conj(),delta) but >3x faster
            op = delta_bar_otimes_delta * op  # hadamard product

            if not return_in_frame_basis and self.frame_basis is not None:
                if self._cached_c_bar_otimes_c is None:
                    self._cached_c_bar_otimes_c = np.kron(self.frame_basis.conj(), self.frame_basis)
                    self._cached_c_trans_otimes_c_dagg = np.kron(
                        self.frame_basis.T, self.frame_basis_adjoint
                    )
                op = self._cached_c_bar_otimes_c @ op @ self._cached_c_trans_otimes_c_dagg

        return op


def _is_herm_or_anti_herm(mat: Array, atol: Optional[float] = 1e-10, rtol: Optional[float] = 1e-10):
    r"""Given `mat`, the logic of this function is:
        - if `mat` is hermitian, return `-1j * mat`
        - if `mat` is anti-hermitian, return `mat`
        - otherwise:
            - if `mat.backend == 'jax'` return `jnp.inf * mat`
            - otherwise raise an error

    The main purpose of this function is to hide the pecularities of the
    implementing the above logic in a compileable way in `jax`.

    Args:
        mat: array to check
        atol: absolute tolerance
        rtol: relative tolerance

    Returns:
        Array: anti-hermitian version of `mat` if applicable

    Raises:
        ImportError: if backend is jax and jax is not installed.
        QiskitError: if `mat` is not Hermitian or anti-Hermitian
    """
    mat = to_array(mat)
    mat = Array(mat, dtype=complex)

    if mat.backend == "jax":

        from jax.lax import cond
        import jax.numpy as jnp

        mat = mat.data

        if mat.ndim == 1:
            # this function checks if pure imaginary. If yes it returns the
            # array, otherwise it multiplies it by jnp.nan to raise an error
            # Note: pathways in conditionals in jax cannot raise Exceptions
            def anti_herm_conditional(b):
                aherm_pred = jnp.allclose(b, -b.conj(), atol=atol, rtol=rtol)
                return cond(aherm_pred, lambda A: A, lambda A: jnp.nan * A, b)

            # Check if it is purely real, if not apply anti_herm_conditional
            herm_pred = jnp.allclose(mat, mat.conj(), atol=atol, rtol=rtol)
            return Array(cond(herm_pred, lambda A: -1j * A, anti_herm_conditional, mat))
        else:
            # this function checks if anti-hermitian, if yes returns the array,
            # otherwise it multiplies it by jnp.nan
            def anti_herm_conditional(b):
                aherm_pred = jnp.allclose(b, -b.conj().transpose(), atol=atol, rtol=rtol)
                return cond(aherm_pred, lambda A: A, lambda A: jnp.nan * A, b)

            # the following lines check if a is hermitian, otherwise it feeds
            # it into the anti_herm_conditional
            herm_pred = jnp.allclose(mat, mat.conj().transpose(), atol=atol, rtol=rtol)
            return Array(cond(herm_pred, lambda A: -1j * A, anti_herm_conditional, mat))

    else:
        if mat.ndim == 1:
            if np.allclose(mat, mat.conj(), atol=atol, rtol=rtol):
                return -1j * mat
            elif np.allclose(mat, -mat.conj(), atol=atol, rtol=rtol):
                return mat
        else:
            if is_hermitian_matrix(mat, rtol=rtol, atol=atol):
                return -1j * mat
            elif is_hermitian_matrix(1j * mat, rtol=rtol, atol=atol):
                return mat

        # raise error if execution has made it this far
        raise QiskitError(
            """frame_operator must be either a Hermitian or
                           anti-Hermitian matrix."""
        )


# Vestigial docstring from operators_into_frame_basis_w_cutoff
# r"""Transform operators into the frame basis, and return two lists of
# operators: one with the 'frequency cutoff' and one with 'conjugate
# frequency cutoff' (explained below). This serves as a helper function
# for evaluating a time-dependent operator :math:`A(t)` specified as a
# linear combination of terms with carrier frequencies, in the frame
# :math:`F` with a cutoff frequency (in the frame basis).

# In particular, this function assumes the operator :math:`A(t)` is
# specified as:

# .. math::
#     A(t) = \sum_j Re[f_j(t) e^{i 2 \pi \nu_j t}] A_j

# For some functions :math:`f_j`, carrier frequencies :math:`nu_j`,
# and operators :math:`A_j`.

# Assume we are already in a basis in which :math:`F` is diagonal, and
# let :math:`D=F`. As described elsewhere in the docstrings for this
# class, evaluating :math:`A(t)` in this frame at a time :math:`t`
# means computing :math:`\exp(-t D)A(t)\exp(tD)`. The benefit of working
# in the basis in which the frame is diagonal is that this computation
# simplifies to:

# .. math::
#     [\exp( (-d_j + d_k) t)] \odot A(t),

# where above :math:`[\exp( (-d_j + d_k) t)]` denotes the matrix whose
# :math:`(j,k)` entry is :math:`\exp( (-d_j + d_k) t)`, and :math:`\odot`
# denotes entrywise multiplication.

# Evaluating the above with 'frequency cutoffs' requires expanding
# :math:`A(t)` into its linear combination. A single term in the sum
# (dropping the summation subscript) is:

# .. math::
#     Re[f(t) e^{i 2 \pi \nu t}] [\exp( (-d_j + d_k) t)] \odot A.

# Next, we expand this further using

# .. math::
#     Re[f(t) e^{i 2 \pi \nu t}] =
#     \frac{1}{2}(f(t) e^{i 2 \pi \nu t} +
#     \overline{f(t)} e^{-i 2 \pi \nu t})

# to get:

# .. math::
#     \frac{1}{2}f(t) e^{i 2 \pi \nu t} [\exp( (-d_j + d_k) t)] \odot A +
#     \frac{1}{2}\overline{f(t)} e^{-i 2 \pi \nu t}
#     [\exp( (-d_j + d_k) t)] \odot A

# Examining the first term in the sum, the 'frequency' associated with
# matrix element :math:`(j,k)` is
# :math:`\nu + \frac{Im[-d_j + d_k]}{2 \pi}`, and similarly for the
# second term: :math:`-\nu + \frac{Im[-d_j + d_k]}{2 \pi}`.

# Evaluating the above expression with a 'frequency cutoff' :math:`\nu_*`
# means computing it, but setting all matrix elements in either term
# with a frequency above :math:`\nu_*` to zero. This can be achieved
# by defining two matrices :math:`A^\pm` to be equal to :math:`A`,
# except the :math:`(j,k)` is set to zero if
# :math:`\pm\nu + \frac{Im[-d_j + d_k]}{2 \pi} \geq \nu_*`.

# Thus, the above expression is evaluated with frequency cutoff via

# .. math::
#     \frac{1}{2}f(t) e^{i 2 \pi \nu t} [\exp( (-d_j + d_k) t)] \odot A^+
#     + \frac{1}{2}\overline{f(t)} e^{-i 2 \pi \nu t}
#     [\exp( (-d_j + d_k) t)] \odot A^-

# Relative to the initial list of operators :math:`A_j`, this function
# returns two lists of matrices as a 3d array: :math:`A_j^+` and
# :math:`A_j^-`, corresponding to :math:`A_j` with frequency cutoffs and
# 'conjugate' frequency cutoffs, in the basis in which the frame has
# been diagonalized.

# To use the output of this function to evalute the original operator
# :math:`A(t)` in the rotating frame, compute the linear combination

# .. math::
#     \frac{1}{2} \sum_j f_j(t) e^{i 2 \pi \nu t} A_j^+
#     + \overline{f(t)} e^{-i 2 \pi \nu t} A_j^-

# then use `self.operator_into_frame` or `self.generator_into_frame`
# the frame transformation as required, using `operator_in_frame=True`.

# Args:
#     operators: list of operators
#     cutoff_freq: cutoff frequency
#     carrier_freqs: list of carrier frequencies

# Returns:
#     Tuple[Array, Array]: The operators with frequency cutoff
#     and conjugate frequency cutoff.
# """
