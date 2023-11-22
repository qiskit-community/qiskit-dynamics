# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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
from qiskit_dynamics.arraylias import ArrayLike
from qiskit_dynamics.arraylias import DYNAMICS_NUMPY_ALIAS
from qiskit_dynamics.arraylias import DYNAMICS_NUMPY as unp

try:
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    jsparse_matmul = jsparse.sparsify(jnp.matmul)
except ImportError:
    pass


class RotatingFrame:
    r"""Class for representing a rotation frame transformation.

    This class provides functionality for transforming various objects into or out-of a rotating
    frame specified by an anti-Hermitian operator :math:`F = -iH`. For example:

        * Bringing a "state" into/out of the frame: :math:`t, y \mapsto e^{\mp tF}y`
        * Bringing an "operator" into/out of the frame: :math:`t, A \mapsto e^{\mp tF}Ae^{\pm tF}`
        * Bringing a generator for a LMDE into/out of the frame:
          :math:`t, G \mapsto e^{\mp tF}Ge^{\pm tF} - F`

    This class also contains functions for bringing states/operators into/out of the basis in which
    :math:`F` is diagonalized, which we refer to as the "frame basis". All previously mentioned
    functions also include optional arguments specifying whether the input/output are meant to be in
    the frame basis.

    .. note::
        :class:`~qiskit_dynamics.models.RotatingFrame` can be instantiated
        with a 1d array, which is understood to correspond to the diagonal entries
        of a diagonal :math:`H` or :math:`F = -i H`.

    """

    def __init__(
        self,
        frame_operator: Union[ArrayLike, Operator, None],
        atol: float = 1e-10,
        rtol: float = 1e-10,
    ):
        """Initialize with a frame operator.

        Args:
            frame_operator: The frame operator, which must be either Hermitian or anti-Hermitian.
            atol: Absolute tolerance when verifying that the ``frame_operator`` is Hermitian or
                anti-Hermitian.
            rtol: Relative tolerance when verifying that the ``frame_operator`` is Hermitian or
                anti-Hermitian.
        """
        if isinstance(frame_operator, RotatingFrame):
            frame_operator = frame_operator.frame_operator

        self._frame_operator = frame_operator
        if frame_operator is not None:
            frame_operator = unp.asarray(frame_operator)

        if frame_operator is None:
            self._dim = None
            self._frame_diag = None
            self._frame_basis = None
            self._frame_basis_adjoint = None
        # if frame_operator is a 1d array, assume already diagonalized
        elif frame_operator.ndim == 1:
            # verify Hermitian or anti-Hermitian
            # if Hermitian convert to anti-Hermitian
            frame_operator = _enforce_anti_herm(frame_operator, atol=atol, rtol=rtol)

            self._frame_basis = None
            self._frame_basis_adjoint = None
            self._dim = len(self._frame_diag)
        # if not, diagonalize it
        else:
            # verify Hermitian or anti-Hermitian
            # if Hermitian convert to anti-Hermitian
            frame_operator = _enforce_anti_herm(frame_operator, atol=atol, rtol=rtol)

            # diagonalize with eigh, utilizing assumption of anti-hermiticity
            frame_diag, frame_basis = unp.linalg.eigh(1j * frame_operator)

            self._frame_diag = -1j * frame_diag
            self._frame_basis = frame_basis
            self._frame_basis_adjoint = frame_basis.conj().transpose()
            self._dim = len(self._frame_diag)

        # lazily evaluate change-of-basis matrices for vectorized operators.
        self._vectorized_frame_basis = None
        self._vectorized_frame_basis_adjoint = None

    @property
    def dim(self) -> int:
        """The dimension of the frame."""
        return self._dim

    @property
    def frame_operator(self) -> ArrayLike:
        """The original frame operator."""
        return self._frame_operator

    @property
    def frame_diag(self) -> ArrayLike:
        """The diagonal of the frame operator."""
        return self._frame_diag

    @property
    def frame_basis(self) -> ArrayLike:
        """The array containing diagonalizing unitary."""
        return self._frame_basis

    @property
    def frame_basis_adjoint(self) -> ArrayLike:
        """The adjoint of the diagonalizing unitary."""
        return self._frame_basis_adjoint

    def state_into_frame_basis(self, y: ArrayLike) -> ArrayLike:
        r"""Take a state into the frame basis, i.e. return ``self.frame_basis_adjoint @ y``.

        Args:
            y: The state.

        Returns:
            ArrayLike: The state in the frame basis.
        """
        y = unp.asarray(y)
        if self.frame_basis_adjoint is None:
            return y

        return self.frame_basis_adjoint @ y

    def state_out_of_frame_basis(self, y: ArrayLike) -> ArrayLike:
        r"""Take a state out of the frame basis, i.e. ``return self.frame_basis @ y``.

        Args:
            y: The state.

        Returns:
            Array: The state in the frame basis.
        """
        y = unp.asarray(y)
        if self.frame_basis is None:
            return y

        return self.frame_basis @ y

    def operator_into_frame_basis(
        self,
        op: Union[Operator, List[Operator], ArrayLike, None],
        convert_type: bool = True,
    ) -> ArrayLike:
        r"""Take an operator into the frame basis, i.e. return
        ``self.frame_basis_adjoint @ A @ self.frame_basis``

        Args:
            op: The operator or array of operators.
            convert_type: Whether or not to initially convert ``op`` into an expected type. Should
                          only be set to ``False`` in situations in which it is gauranteed that
                          ``op`` is a handled input type.

        Returns:
            Array: The operator in the frame basis.
        """
        if convert_type:
            op = unp.asarray(op)

        if self.frame_basis is None or op is None:
            return op

        if isinstance(op, list):
            return [self.operator_into_frame_basis(x, convert_type=False) for x in op]

        return unp.rmatmul(unp.matmul(op, self.frame_basis), self.frame_basis_adjoint)

    def operator_out_of_frame_basis(
        self,
        op: Union[Operator, List[Operator], ArrayLike, None],
        convert_type: bool = True,
    ) -> ArrayLike:
        r"""Take an operator out of the frame basis, i.e. return
        ``self.frame_basis @ to_array(op) @ self.frame_basis_adjoint``.

        Args:
            op: The operator or array of operators.
            convert_type: Whether or not to initially convert ``op`` into an expected type. Should
                          only be set to ``False`` in situations in which it is gauranteed that
                          that ``op`` is a handled input type.

        Returns:
            ArrayLike: The operator in the frame basis.
        """
        if convert_type:
            op = unp.asarray(op)

        if self.frame_basis is None or op is None:
            return op

        if isinstance(op, list):
            return [self.operator_out_of_frame_basis(x, convert_type=False) for x in op]

        return unp.rmatmul(unp.matmul(op, self.frame_basis_adjoint), self.frame_basis)

    def state_into_frame(
        self,
        t: float,
        y: ArrayLike,
        y_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> ArrayLike:
        """Take a state into the rotating frame, i.e. return ``exp(-tF) @ y``.

        Args:
            t: The time.
            y: The state.
            y_in_frame_basis: Whether or not the array y is already in the basis in which the frame
                is diagonal.
            return_in_frame_basis: Whether or not to return the result in the frame basis.

        Returns:
            ArrayLike: The state in the rotating frame.
        """
        y = unp.asarray(y)
        if self._frame_operator is None:
            return y

        out = y

        # if not in frame basis convert it
        if not y_in_frame_basis:
            out = self.state_into_frame_basis(out)

        # go into the frame using fast diagonal matrix multiplication
        out = (unp.exp(self.frame_diag * (-t)) * out.transpose()).transpose()  # = e^{tF}out

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.state_out_of_frame_basis(out)

        return out

    def state_out_of_frame(
        self,
        t: float,
        y: ArrayLike,
        y_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> ArrayLike:
        r"""Take a state out of the rotating frame, i.e. return ``exp(tF) @ y``.

        Calls ``self.state_into_frame`` with time reversed.

        Args:
            t: The time.
            y: The state..
            y_in_frame_basis: Whether or not the array y is already in the basis in which the frame
                is diagonal.
            return_in_frame_basis: Whether or not to return the result in the frame basis.

        Returns:
            ArrayLike: The state out of the rotating frame.
        """
        return self.state_into_frame(-t, y, y_in_frame_basis, return_in_frame_basis)

    def _conjugate_and_add(
        self,
        t: float,
        operator: ArrayLike,
        op_to_add_in_fb: Optional[ArrayLike] = None,
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ) -> ArrayLike:
        r"""General helper function for computing :math:`\exp(-tF)G\exp(tF) + B`.

        Note: :math:`B` is added in the frame basis before any potential final change
        out of the frame basis.

        Note that there are two conventions for passing multiple operators at the same time. For
        evaluation with ``vectorized_operators=False``, these operators should be passed as ``(k,
        dim, dim)`` array objects, with the :math:`i^{th}` operator being stored as the ``[i,:,:]``
        entry. For ``vectorized_operators = True``, these (vectorized) operators should be passed as
        a ``(dim**2, k)`` array, with the :math:`i^{th}` vectorized operator stored as the ``[:,i]``
        entry.

        Args:
            t: The time.
            operator: The operator :math:`G`.
            op_to_add_in_fb: The additional operator :math:`B`.
            operator_in_fame_basis: Whether ``operator`` is already in the basis in which the frame
                operator is diagonal.
            vectorized_operators: Whether ``operator`` is passed as a vectorized, ``(dim**2,)``
                array, rather than a ``(dim,dim)`` array.

        Returns:
            Array of the newly conjugated operator.
        """
        operator = unp.asarray(operator)
        if op_to_add_in_fb is not None:
            op_to_add_in_fb = unp.asarray(op_to_add_in_fb)
        if vectorized_operators:
            # If passing vectorized operator, undo vectorization temporarily
            if self._frame_operator is None:
                if op_to_add_in_fb is None:
                    return operator
                else:
                    return operator + op_to_add_in_fb
            if len(operator.shape) == 2:
                operator = operator.T
            operator = operator.reshape(operator.shape[:-1] + (self.dim, self.dim), order="F")

        if self._frame_operator is None:
            if op_to_add_in_fb is None:
                return operator
            else:
                op_to_add_in_fb = DYNAMICS_NUMPY_ALIAS(like=operator).asarray(op_to_add_in_fb)

                return operator + op_to_add_in_fb

        out = operator

        # if not in frame basis convert it
        if not operator_in_frame_basis:
            out = self.operator_into_frame_basis(out)

        # get frame transformation matrix in diagonal basis
        # assumption that F is anti-Hermitian implies conjugation of
        # diagonal gives inversion
        exp_freq = unp.exp(self.frame_diag * t)
        frame_mat = exp_freq.conj().reshape(self.dim, 1) * exp_freq

        out = unp.multiply(out, frame_mat)

        if op_to_add_in_fb is not None:
            op_to_add_in_fb = DYNAMICS_NUMPY_ALIAS(like=out).asarray(op_to_add_in_fb)

            out = out + op_to_add_in_fb

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.operator_out_of_frame_basis(out)

        if vectorized_operators:
            # If a vectorized output is required, reshape correctly
            out = out.reshape(out.shape[:-2] + (self.dim**2,), order="F")
            if len(out.shape) == 2:
                out = out.T

        return out

    def operator_into_frame(
        self,
        t: float,
        operator: Union[Operator, ArrayLike],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ) -> ArrayLike:
        r"""Bring an operator into the frame, i.e. return
        ``exp(-tF) @ operator @ exp(tF)``.

        The default implementation is to use ``self._conjugate_and_add``.

        Args:
            t: The time.
            operator: An array of appropriate size.
            operator_in_frame_basis: Whether or not the operator is already in the basis in which
                the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result in the frame basis.
            vectorized_operators: Whether ``operator`` is passed as a vectorized,
                ``(dim**2,)`` array, rather than a ``(dim,dim)`` array.

        Returns:
            ArrayLike: The operator in the rotating frame.
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
        operator: Union[Operator, ArrayLike],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ) -> ArrayLike:
        r"""Bring an operator into the rotating frame, i.e. return
        ``exp(tF) @ operator @ exp(-tF)``.

        The default implmentation is to use `self.operator_into_frame`.

        Args:
            t: Time.
            operator: An array of appropriate size.
            operator_in_frame_basis: Whether or not the operator is already in the basis in which
                the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result in the frame basis.
            vectorized_operators: Whether ``operator`` is passed as a vectorized, ``(dim**2,)``
                array, rather than a ``(dim,dim)`` array.

        Returns:
            ArrayLike: The operator out of the rotating frame.
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
        operator: Union[Operator, ArrayLike],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
        vectorized_operators: Optional[bool] = False,
    ) -> ArrayLike:
        r"""Take an generator into the rotating frame, i.e. return
        ``exp(-tF) @ operator @ exp(tF) - F``.

        The default implementation is to use :meth:`_conjugate_and_add`.

        Args:
            t: Time.
            operator: Generator (array of appropriate size).
            operator_in_frame_basis: Whether or not the generator is already in the basis in which
                the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result in the frame basis.
            vectorized_operators: Whether ``operator`` is passed as a vectorized, ``(dim**2,)``
                array, rather than a ``(dim,dim)`` array.

        Returns:
            ArrayLike: The generator in the rotating frame.
        """
        if self.frame_operator is None:
            return unp.asarray(operator)
        else:
            # conjugate and subtract the frame diagonal
            return self._conjugate_and_add(
                t,
                operator,
                op_to_add_in_fb=-unp.diag(self.frame_diag),
                operator_in_frame_basis=operator_in_frame_basis,
                return_in_frame_basis=return_in_frame_basis,
                vectorized_operators=vectorized_operators,
            )

    def generator_out_of_frame(
        self,
        t: float,
        operator: Union[Operator, ArrayLike],
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> ArrayLike:
        r"""Take an operator out of the frame using the generator transformaton rule, i.e. return
        ``exp(tF) @ operator @ exp(-tF) + F``.

        The default implementation is to use ``self._conjugate_and_add``.

        Args:
            t: The time
            operator: Generator (array of appropriate size).
            operator_in_frame_basis: Whether or not the operator is already in the basis in which
                the frame is diagonal.
            return_in_frame_basis: Whether or not to return the result in the frame basis.

        Returns:
            ArrayLike: The generator out of the rotating frame.
        """
        if self.frame_operator is None:
            return unp.asarray(operator)
        else:
            # conjugate and add the frame diagonal
            return self._conjugate_and_add(
                -t,
                operator,
                op_to_add_in_fb=unp.diag(self.frame_diag),
                operator_in_frame_basis=operator_in_frame_basis,
                return_in_frame_basis=return_in_frame_basis,
            )

    @property
    def vectorized_frame_basis(self):
        """Lazily evaluated operator for mapping vectorized operators into the frame basis."""

        if self.frame_basis is None:
            return None

        if self._vectorized_frame_basis is None:
            self._vectorized_frame_basis = unp.kron(self.frame_basis.conj(), self.frame_basis)
            self._vectorized_frame_basis_adjoint = self._vectorized_frame_basis.conj().transpose()

        return self._vectorized_frame_basis

    @property
    def vectorized_frame_basis_adjoint(self):
        """Lazily evaluated operator for mapping vectorized operators out of the frame basis."""

        if self.frame_basis is None:
            return None

        if self._vectorized_frame_basis_adjoint is None:
            # trigger lazy evaluation of vectorized_frame_basis
            # pylint: disable=pointless-statement
            self.vectorized_frame_basis

        return self._vectorized_frame_basis_adjoint

    def vectorized_map_into_frame(
        self,
        time: float,
        op: ArrayLike,
        operator_in_frame_basis: Optional[bool] = False,
        return_in_frame_basis: Optional[bool] = False,
    ) -> ArrayLike:
        r"""Given an operator ``op`` of dimension ``dim**2`` assumed to represent vectorized linear
        map in column stacking convention, returns:

        .. math::
            ((e^{tF})^T \otimes e^{-tF}) \times op \times ((e^{-tF})^T \otimes e^{tF}).

        Utilizes element-wise multiplication :math:`op \to \Delta\otimes\bar{\Delta} \odot op`,
        where :math:`\Delta_{ij}=\exp((-d_i+d_j)t)` is the frame difference matrix, as well as
        caches array :math:`\bar{C}\otimes C`, where ``C = self.frame_basis`` for future use.

        Args:
            time: The time :math:`t`.
            op: The ``(dim**2,dim**2)`` array.
            operator_in_frame_basis: Whether the operator is in the frame basis.
            return_in_frame_basis: Whether the operator should be returned in the frame basis.

        Returns:
            ArrayLike: ``op`` in the frame.
        """
        if self.frame_diag is not None:
            # Put the vectorized operator into the frame basis
            if not operator_in_frame_basis and self.frame_basis is not None:
                op = self.vectorized_frame_basis_adjoint @ (op @ self.vectorized_frame_basis)

            expvals = np.exp(self.frame_diag * time)  # = e^{td_i} = e^{it*Im(d_i)}
            # = kron(e^{-it*Im(d_i)},e^{it*Im(d_i)}), but ~3x faster
            temp_outer = (expvals.conj().reshape(self.dim, 1) * expvals).flatten()
            delta_bar_otimes_delta = np.outer(
                temp_outer.conj(), temp_outer
            )  # = kron(delta.conj(),delta) but >3x faster
            if issparse(op):
                op = op.multiply(delta_bar_otimes_delta)
            else:
                op = delta_bar_otimes_delta * op  # hadamard product

            if not return_in_frame_basis and self.frame_basis is not None:
                op = self.vectorized_frame_basis @ (op @ self.vectorized_frame_basis_adjoint)

        return op


def _enforce_anti_herm(
    mat: ArrayLike, atol: Optional[float] = 1e-10, rtol: Optional[float] = 1e-10
) -> ArrayLike:
    r"""Given ``mat``, the logic of this function is:
        - if ``mat`` is hermitian, return ``-1j * mat``
        - if ``mat`` is anti-hermitian, return ``mat``
        - otherwise:
            - if ``mat.backend == 'jax'`` return ``jnp.inf * mat``
            - otherwise raise an error

    The main purpose of this function is to hide the pecularities of implementing the above logic in
    a compileable way in ``jax``.

    Args:
        mat: The array to check.
        atol: The absolute tolerance.
        rtol: The relative tolerance.

    Returns:
        ArrayLike: Anti-hermitian version of ``mat``, if applicable.

    Raises:
        ImportError: If the backend is jax and jax is not installed.
        QiskitError: If ``mat`` is not Hermitian or anti-Hermitian.
    """
    mat = unp.asarray(mat)

    try:
        from jax.lax import cond, convert_element_type

        if isinstance(unp.asarray(mat), jnp.ndarray):
            if mat.ndim == 1:
                # this function checks if pure imaginary. If yes it returns the
                # array, otherwise it multiplies it by jnp.nan to raise an error
                # Note: pathways in conditionals in jax cannot raise Exceptions
                def anti_herm_conditional(b):
                    aherm_pred = jnp.allclose(b, -b.conj(), atol=atol, rtol=rtol)
                    return convert_element_type(
                        cond(aherm_pred, lambda A: A, lambda A: jnp.nan * A, b), jnp.complex128
                    )

                # Check if it is purely real, if not apply anti_herm_conditional
                herm_pred = jnp.allclose(mat, mat.conj(), atol=atol, rtol=rtol)
                return unp.asarray(cond(herm_pred, lambda A: -1j * A, anti_herm_conditional, mat))
            else:
                # this function checks if anti-hermitian, if yes returns the array,
                # otherwise it multiplies it by jnp.nan
                def anti_herm_conditional(b):
                    aherm_pred = jnp.allclose(b, -b.conj().transpose(), atol=atol, rtol=rtol)
                    return convert_element_type(
                        cond(aherm_pred, lambda A: A, lambda A: jnp.nan * A, b), jnp.complex128
                    )

                # the following lines check if a is hermitian, otherwise it feeds
                # it into the anti_herm_conditional
                herm_pred = jnp.allclose(mat, mat.conj().transpose(), atol=atol, rtol=rtol)
                return unp.asarray(cond(herm_pred, lambda A: -1j * A, anti_herm_conditional, mat))
    except (ImportError, NameError):
        pass

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
    raise QiskitError("""frame_operator must be either a Hermitian or anti-Hermitian matrix.""")
