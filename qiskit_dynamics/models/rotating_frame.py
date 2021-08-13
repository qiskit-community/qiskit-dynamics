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

from abc import ABC, abstractmethod
from typing import Union, List, Optional, Tuple
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
            frame_operator: the frame operator, must be either
                            Hermitian or anti-Hermitian.
            atol: absolute tolerance when verifying that the frame_operator is
                  Hermitian or anti-Hermitian.
            rtol: relative tolerance when verifying that the frame_operator is
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

        # these properties are memory-expensive to store. Will fill out with values only if necessary.
        self._cached_c_bar_otimes_c = None
        self._cached_c_trans_otimes_c_dagg = None

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
            y: the state
        Returns:
            Array: the state in the frame basis
        """
        if self.frame_basis_adjoint is None:
            return to_array(y)

        return self.frame_basis_adjoint @ y

    def state_out_of_frame_basis(self, y: Array) -> Array:
        r"""Take a state out of the frame basis, i.e.
        ``return self.frame_basis @ y``.

        Args:
            y: the state
        Returns:
            Array: the state in the frame basis
        """
        if self.frame_basis is None:
            return to_array(y)

        return self.frame_basis @ y

    def operator_into_frame_basis(self, op: Union[Operator, List[Operator], Array]) -> Array:
        r"""Take an operator into the frame basis, i.e. return
        ``self.frame_basis_adjoint @ A @ self.frame_basis``

        Args:
            op: the operator or array of operators.
        Returns:
            Array: the operator in the frame basis
        """
        op = to_array(op)
        if self.frame_basis is None:
            return op
        # parentheses are necessary for sparse op as of writing this comment
        return self.frame_basis_adjoint @ (op @ self.frame_basis)

    def operator_out_of_frame_basis(self, op: Union[Operator, Array]) -> Array:
        r"""Take an operator out of the frame basis, i.e. return
        ``self.frame_basis @ to_array(op) @ self.frame_basis_adjoint``.

        Args:
            op: the operator or array of operators.
        Returns:
            Array: the operator in the frame basis
        """
        op = to_array(op)
        if self.frame_basis is None:
            return op
        # parentheses are necessary for sparse op as of writing this comment
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
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis

        Returns:
            Array: state in frame
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
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis

        Returns:
            Array: state out of frame
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
        r"""Concrete implementation of general helper function for computing
            exp(-tF)Gexp(tF) + B

        Note: B is added in the frame basis before any potential final change
        out of the frame basis.

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
        frame_mat = exp_freq.conj().reshape(self.dim,1)*exp_freq
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
        ``exp(-tF) @ operator @ exp(tF)``

        Default implementation is to use ``self._conjugate_and_add``.

        Args:
            t: time
            operator: array of appropriate size
            operator_in_frame_basis: whether or not the operator is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        Returns:
            Array: operator in frame
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
            t: time
            operator: array of appropriate size
            operator_in_frame_basis: whether or not the operator is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis

        Returns:
            Array: operator out of frame
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
            t: time
            operator: generator (array of appropriate size)
            operator_in_frame_basis: whether or not the generator is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis

        Returns:
            Array: generator in frame
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
            t: time
            operator: generator (array of appropriate size)
            operator_in_frame_basis: whether or not the operator is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis

        Returns:
            Array: generator out of frame
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

    def bring_vectorized_operator_into_frame(
        self,
        time: float,
        op: Array,
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> Array:
        r"""Sometimes-necessary function that will bring a vectorized operator
        (as an (dim^2,dim^2) Array) into the rotating frame. Much slower than
        operators_into_frame_basis, as it is faster to conjugate the vectorized
        state/density matrix as a (dim^2,) Array than to conjugate a (dim^2,dim^2)
        Array, but necessary in the case that the user requests the (dim^2,dim^2)
        generator matrix, which will then need to be put into the frame,
        independently of the state of the system.

        Args:
            time: the time t
            op: The (dim^2,dim^2) Array
            operator_in_frame_basis: whether the operator is in the frame basis
            return_in_frame_basis: whether the operator should be returned in the
                frame basis
        Returns:
            op in the frame.

        Formalism: Let the state of a system be described by a matrix \rho. Let
        \dot{\rho} = \sum_j A_j \rho B_j for some (dim,dim) matrices A_j,B_j. Note
        that the Lindbladian takes this form. If we vectorize this equation, we find
        that vec{\dot{\rho}} = \dot{\vec{\rho}} = [\sum_j (B_j)^T\otimes A_j]vec{\rho}.
        Now, when we go to the rotating frame, where \rho\to\zeta = e^{-tF}\rho e^{tF},
        we find that \dot{\zeta} = e^{-tF}(-F\rho + \dot{\rho} + \rho F)e^{tF} =
        e^{-tF}(\sum_j A_j\rho B_j - F\rho I + I\rho F)e^{tF}. Note that the frame terms
        -F\rho I + I\rho F may be incorporated as part of our sum \sum_j A_j\rho B_j,
        which we do in software by subtracting F from the drift terms. As a
        result, we will be treating op in this function as if it already incoroporates these
        \pm F terms. This means that we may sum by rolling the frame terms into it. This then
        yields \dot{\zeta} = \sum_j (e^{-tF}A_j e^{tF})\zeta (e^{-tF}B_j e^{tF}), or
        vec\dot{\zeta}=[\sum_j(e^{-tF}B_je^{tF})^T\otimes (e^{-tF}A_je^{tF})]vec{\zeta}.
        Because we work in the basis in which F is diagonal, we may write this sum as
        \sum_j (e^{tF}(B_j)^Te^{-tF})\otimes (e^{-tF}A_j e^{tF}). Define
        \Delta:= (\Delta)_{ij} = e^{(-d_i + d_j)t} where d_i is the i^{th} eigenvalue
        of F. We know from linear algebra that because F is diagonal, we may write
        e^{-tF}A_j e^{tF} = \Delta\circ A_j, where \circ is the Hadamard product.
        Then, note that because F is antihermitian, the d_i are purely imaginary.
        Hence, e^{tF}(B_j)^Te^{-tF} = \bar{\Delta} \circ (B_j)^T. Then, we have
            (e^{tF}(B_j)^Te^{-tF})\otimes (e^{-tF}A_je^{tF})
            = (\bar{\Delta}\circ (B_j)^T)\otimes (Delta\circ A_j)
            = (\bar{\Delta)\otimes \Delta)\circ ((B_j)^T\otimes A_j).
        This yields our full generator in the frame:
            G_{vec} = (\bar{\Delta}\otimes \Delta)\circ \sum_j (B_j)^T\otimes A_j

        All that is left is to consider the possibility of basis transformations. Let
        C = self.frame_basis, so that putting an operator X into the frame basis
        corresponds with X -> C^\dagger XC. Now, we consider the equation of motion when
        our matrices A_j,B_j are not in the frame basis. In this case, we write
        \dot{\rho} = \sum_j C^\dagger A_jC \rho C^\dagger B_j C so we have
        \dot{vec(\rho)} = (\sum_j (C^T(B_j)^T\bar{C})\otimes (C^\dagger A_jC))\vec{\rho}
        so we must write our generator as the following
            G_{vec} -> (C^T\otimes C^\dagger)G_{vec}(\bar{C}\otimes C)
        though I should note that in most cases, the generator should be in the frame basis
        from the start. If instead we wish to represent this generator in the lab basis,
        we would write this as \dot{\zeta} = \sum_j (CA_jC^\dagger)\zeta(CB_jC^{\dagger}) or
        \dot{vec{\zeta}}=\sum_j(\bar{C}(B_j)^TC^T)\otimes (CA_jC^\dagger) where the A_j,B_j
        are the matrices after the conjugation. This new generator may be expanded as
            G_{vec} -> (\bar{C}\otimes C)G_{vec}(C^T\otimes C^\dagger).
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
            temp_outer = (expvals.conj().reshape(self.dim,1)* expvals).flatten()  
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
