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

from abc import ABC, abstractmethod
from typing import Callable, Union, List, Optional, Tuple
import numpy as np

from qiskit.quantum_info.operators import Operator

class BaseFrame(ABC):
    """Abstract interface for functionality for entering a constant rotating
    frame specified by an anti-Hermitian matrix F.

    Frames have relevance within the context of differential equations of the
    form :math:`\dot{y}(t) = G(t)y(t)`. "Entering a frame" given by :math:`F`
    corresponds to a change of solution variable :math:`z(t) = e^{-tF}y(t)`.
    Using the definition, we may write down a differential equation for
    :math:`z(t)`:

    .. math:
        \dot{z}(t) = -F z(t) + e^{-tF}G(t)y(t) = (e^{-tF}G(t)e^{tF} - F)z(t)

    In some cases it is computationally easier to solve for :math:`z(t)`
    than it is to solve for :math:`y(t)`.

    While entering a frame is mathematically well-defined for arbitrary
    matrices :math:`F`, we assume in this class that :math:`F` is
    anti-Hermitian, ensuring beneficial properties:
        - :math:`F` is unitarily diagonalizable.
        - :math:`e^{\pm tF}` is easily inverted by taking the adjoint.
        - The frame transformation is norm preserving.
    That :math:`F` is diagonalizable is especially important, as :math:`e^{tF}`
    will need repeated evaluation for different :math:`t` (e.g. at every RHS
    sample point when solving a DE), so it is useful to work in a basis in which
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

    Finally, this class offers support for evaluating linear combinations of
    operators with coefficients with carrier frequencies, along with frequency
    cutoffs for implementing the Rotating Wave Approximation (RWA). Frame
    information and carrier frequency information are intrinsically tied
    together in this context.

    Note: all abstract doc strings are written in a numpy style
    """

    @property
    @abstractmethod
    def frame_operator(self) -> Union[Operator, np.array]:
        """The original frame operator."""

    @property
    @abstractmethod
    def frame_diag(self) -> np.array:
        """Diagonal of the frame operator as a 1d array."""

    @property
    @abstractmethod
    def frame_basis(self) -> np.array:
        """Array containing the unitary :math:`U` that diagonalizes the
        frame operator, i.e. :math:`U` such that :math:`F = U D U^\dagger`.
        """

    @property
    @abstractmethod
    def frame_basis_adjoint(self) -> np.array:
        """Adjoint of self.frame_basis."""

    @abstractmethod
    def state_into_frame_basis(self, y: np.array) -> np.array:
        """Take a state into the frame basis, i.e. return
        self.frame_basis_adjoint @ y.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """

    @abstractmethod
    def state_out_of_frame_basis(self, y: np.array) -> np.array:
        """Take a state out of the frame basis, i.e.
        return self.frame_basis @ y.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """

    @abstractmethod
    def operator_into_frame_basis(self,
                                  op: Union[Operator, np.array]) -> np.array:
        """Take an operator into the frame basis, i.e. return
        self.frame_basis_adjoint @ A @ self.frame_basis

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """

    @abstractmethod
    def operator_out_of_frame_basis(self,
                                    op: Union[Operator, np.array]) -> np.array:
        """Take an operator out of the frame basis, i.e. return
        self.frame_basis @ to_array(op) @ self.frame_basis_adjoint

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """

    @abstractmethod
    def operators_into_frame_basis(self,
                                   operators: Union[List[Operator], np.array]) -> np.array:
        """Given a list of operators, apply self.operator_into_frame_basis to
        all and return as a 3d array.

        Args:
            operators: list of operators
        """

    @abstractmethod
    def operators_out_of_frame_basis(self,
                                     operators: Union[List[Operator], np.array]) -> np.array:
        """Given a list of operators, apply self.operator_out_of_frame_basis to
        all and return as a 3d array.

        Args:
            operators: list of operators
        """

    @abstractmethod
    def state_into_frame(self,
                         t: float,
                         y: np.array,
                         y_in_frame_basis: Optional[bool] = False,
                         return_in_frame_basis: Optional[bool] = False):
        """Take a state into the frame, i.e. return exp(-tF) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """

    def state_out_of_frame(self,
                           t: float,
                           y: np.array,
                           y_in_frame_basis: Optional[bool] = False,
                           return_in_frame_basis: Optional[bool] = False):
        """Take a state out of the frame, i.e. return exp(tF) @ y.

        Default implementation is to call self.state_into_frame.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        return self.state_into_frame(-t, y,
                                     y_in_frame_basis,
                                     return_in_frame_basis)

    @abstractmethod
    def _conjugate_and_add(self,
                           t: float,
                           operator: np.array,
                           op_to_add_in_fb: Optional[np.array] = None,
                           operator_in_frame_basis: Optional[bool] = False,
                           return_in_frame_basis: Optional[bool] = False):
        """Generalized helper function for taking operators and generators
        into/out of the frame.

        Given operator G, and op_to_add_in_fb B, returns exp(-tF)Gexp(tF) + B,
        where B is assumed to be specified in the frame basis.

        Args:
            t: time.
            operator: The operator G above.
            op_to_add_in_fb: The operator B above.
            operator_in_frame_basis: Whether G is specified in the frame basis.
            return_in_frame_basis: Whether the returned result should be in the
                                   frame basis.
        """

    def operator_into_frame(self,
                            t: float,
                            operator: Union[Operator, np.array],
                            operator_in_frame_basis: Optional[bool] = False,
                            return_in_frame_basis: Optional[bool] = False):
        """Bring an operator into the frame, i.e. return
        exp(-tF) @ operator @ exp(tF)

        Default implmentation is to use self._conjugate_and_add

        Args:
            t: time
            operator: array of appropriate size
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        return self._conjugate_and_add(t,
                                       operator,
                                       operator_in_frame_basis=operator_in_frame_basis,
                                       return_in_frame_basis=return_in_frame_basis)

    def operator_out_of_frame(self,
                              t: float,
                              operator: Union[Operator, np.array],
                              operator_in_frame_basis: Optional[bool] = False,
                              return_in_frame_basis: Optional[bool] = False):
        """Bring an operator into the frame, i.e. return
        exp(tF) @ operator @ exp(-tF)

        Default implmentation is to use self.operator_into_frame

        Args:
            t: time
            operator: array of appropriate size
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        return self.operator_into_frame(-t,
                                        operator,
                                        operator_in_frame_basis=operator_in_frame_basis,
                                        return_in_frame_basis=return_in_frame_basis)


    def generator_into_frame(self,
                             t: float,
                             operator: Union[Operator, np.array],
                             operator_in_frame_basis: Optional[bool] = False,
                             return_in_frame_basis: Optional[bool] = False):
        """Take an generator into the frame, i.e. return
        exp(-tF) @ operator @ exp(tF) - F.

        Default implementation is to use self._conjugate_and_add

        Args:
            t: time
            operator: generator (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        if self.frame_operator is None:
            return to_array(operator)
        else:
            # conjugate and subtract the frame diagonal
            return self._conjugate_and_add(t,
                                           operator,
                                           op_to_add_in_fb=-np.diag(self.frame_diag),
                                           operator_in_frame_basis=operator_in_frame_basis,
                                           return_in_frame_basis=return_in_frame_basis)

    def generator_out_of_frame(self,
                               t: float,
                               operator: Union[Operator, np.array],
                               operator_in_frame_basis: Optional[bool] = False,
                               return_in_frame_basis: Optional[bool] = False):
        """Take an operator out of the frame, i.e. return
        exp(tF) @ operator @ exp(-tF) + F.

        Default implementation is to use self._conjugate_and_add

        Args:
            t: time
            operator: generator (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        if self.frame_operator is None:
            return to_array(operator)
        else:
            # conjugate and add the frame diagonal
            return self._conjugate_and_add(-t,
                                           operator,
                                           op_to_add_in_fb=np.diag(self.frame_diag),
                                           operator_in_frame_basis=operator_in_frame_basis,
                                           return_in_frame_basis=return_in_frame_basis)

    @abstractmethod
    def operators_into_frame_basis_with_cutoff(self,
                                               operators: Union[np.array, List[Operator]],
                                               cutoff_freq: Optional[float] = None,
                                               carrier_freqs: Optional[np.array] = None):
        """Transform operators into the frame basis, and return two lists of
        operators: one with the 'frequency cutoff' and one with 'conjugate
        frequency cutoff' (explained below). This serves as a helper function
        for evaluating a time-dependent operator :math:`A(t)` specified as a
        linear combination of terms with carrier frequencies, in the frame
        :math:`F` with a cutoff frequency (in the frame basis).

        In particular, this function assumes the operator :math:`A(t)` is
        specified as:

        .. math::
            A(t) = \sum_j Re[f_j(t) e^{i 2 \pi \nu_j t}] A_j

        For some functions :math:`f_j`, carrier frequencies :math:`nu_j`,
        and operators :math:`A_j`.

        Assume we are already in a basis in which :math:`F` is diagonal, and
        let :math:`D=F`. As described elsewhere in the docstrings for this
        class, evaluating :math:`A(t)` in this frame at a time :math:`t`
        means computing :math:`\exp(-t D)A(t)\exp(tD)`. The benefit of working
        in the basis in which the frame is diagonal is that this computation
        simplifies to:

        .. math::
            [\exp( (-d_j + d_k) t)] \odot A(t),

        where above :math:`[\exp( (-d_j + d_k) t)]` denotes the matrix whose
        :math:`(j,k)` entry is :math:`\exp( (-d_j + d_k) t)`, and :math:`\odot`
        denotes entrywise multiplication.

        Evaluating the above with 'frequency cutoffs' requires expanding
        :math:`A(t)` into its linear combination. A single term in the sum
        (dropping the summation subscript) is:

        .. math::
            Re[f(t) e^{i 2 \pi \nu t}] [\exp( (-d_j + d_k) t)] \odot A.

        Next, we expand this further using

        .. math::
            Re[f(t) e^{i 2 \pi \nu t}] =
                \frac{1}{2}(f(t) e^{i 2 \pi \nu t} + \overline{f(t)} e^{-i 2 \pi \nu t})

        to get:

        .. math::
            \frac{1}{2}f(t) e^{i 2 \pi \nu t} [\exp( (-d_j + d_k) t)] \odot A +
                \frac{1}{2}\overline{f(t)} e^{-i 2 \pi \nu t} [\exp( (-d_j + d_k) t)] \odot A

        Examining the first term in the sum, the 'frequency' associated with
        matrix element :math:`(j,k)` is
        :math:`\nu + \frac{Im[-d_j + d_k]}{2 \pi}`, and similarly for the
        second term: :math:`-\nu + \frac{Im[-d_j + d_k]}{2 \pi}`.

        Evaluating the above expression with a 'frequency cutoff' :math:`\nu_*`
        means computing it, but setting all matrix elements in either term
        with a frequency above :math:`\nu_*` to zero. This can be achieved
        by defining two matrices :math:`A^\pm` to be equal to :math:`A`,
        except the :math:`(j,k)` is set to zero if
        :math:`\pm\nu + \frac{Im[-d_j + d_k]}{2 \pi} \geq \nu_*`.

        Thus, the above expression is evaluated with frequency cutoff via

        .. math::
            \frac{1}{2}f(t) e^{i 2 \pi \nu t} [\exp( (-d_j + d_k) t)] \odot A^+ +
                \frac{1}{2}\overline{f(t)} e^{-i 2 \pi \nu t} [\exp( (-d_j + d_k) t)] \odot A^-

        Relative to the initial list of operators :math:`A_j`, this function
        returns two lists of matrices as a 3d array: :math:`A_j^+` and
        :math:`A_j^-`, corresponding to :math:`A_j` with frequency cutoffs and
        'conjugate' frequency cutoffs, in the basis in which the frame has
        been diagonalized.

        To use the output of this function to evalute the original operator
        :math:`A(t)` in the frame, compute the linear combination

        .. math::
            \frac{1}{2} \sum_j f_j(t) e^{i 2 \pi \nu t} A_j^+ + \overline{f(t)} e^{-i 2 \pi \nu t} A_j^-

        then use `self.operator_into_frame` or `self.generator_into_frame`
        the frame transformation as required, using `operator_in_frame=True`.

        Args:
            operators: list of operators
            cutoff_freq: cutoff frequency
            carrier_freqs: list of carrier frequencies

        Returns:
            Tuple[np.array, np.array]: The operators with frequency cutoff
            and conjugate frequency cutoff.
        """


class Frame(BaseFrame):
    """Concrete implementation of BaseFrame using numpy arrays."""

    def __init__(self, frame_operator: Union[Operator, np.array]):
        """Initialize with a frame operator.

        Args:
            frame_operator: the frame operator, assumed to be anti-Hermitian.
        """

        self._frame_operator = frame_operator

        # if frame_operator is a 1d array, assume already diagonalized
        if frame_operator is None:
            self._dim = None
            self._frame_diag = None
            self._frame_basis = None
            self._frame_basis_adjoint = None
        elif isinstance(frame_operator, np.ndarray) and frame_operator.ndim == 1:

            # verify that it is anti-hermitian (i.e. purely imaginary)
            if np.linalg.norm(frame_operator + frame_operator.conj()) > 1e-10:
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")

            self._frame_diag = frame_operator
            self._frame_basis = np.eye(len(frame_operator))
            self._frame_basis_adjoint = self.frame_basis
            self._dim = len(self._frame_diag)
        # if not, diagonalize it
        else:
            # Ensure that it is an Operator object
            frame_operator = Operator(frame_operator)

            # verify anti-hermitian
            herm_part = frame_operator + frame_operator.adjoint()
            if herm_part != Operator(np.zeros(frame_operator.dim)):
                raise Exception("""frame_operator must be an
                                   anti-Hermitian matrix.""")

            # diagonalize with eigh, utilizing assumption of anti-hermiticity
            frame_diag, frame_basis = np.linalg.eigh(1j * frame_operator.data)

            self._frame_diag = -1j * frame_diag
            self._frame_basis = frame_basis
            self._frame_basis_adjoint = frame_basis.conj().transpose()
            self._dim = len(self._frame_diag)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def frame_operator(self) -> np.array:
        """The original frame operator."""
        return self._frame_operator

    @property
    def frame_diag(self) -> np.array:
        """Diagonal of the frame operator."""
        return self._frame_diag

    @property
    def frame_basis(self) -> np.array:
        """Array containing diagonalizing unitary."""
        return self._frame_basis

    @property
    def frame_basis_adjoint(self) -> np.array:
        """Adjoint of the diagonalizing unitary."""
        return self._frame_basis_adjoint

    def state_into_frame_basis(self, y: np.array) -> np.array:
        """Transform y into the frame basis.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """
        if self._frame_operator is None:
            return to_array(y)

        return self.frame_basis_adjoint @ y

    def state_out_of_frame_basis(self, y: np.array) -> np.array:
        """Transform y out of the frame basis.

        Args:
            y: the state
        Returns:
            np.array: the state in the frame basis
        """
        if self._frame_operator is None:
            return to_array(y)

        return self.frame_basis @ y

    def operator_into_frame_basis(self,
                                  op: Union[Operator, np.array]) -> np.array:
        """Transform operator into frame basis.

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """
        if self._frame_operator is None:
            return to_array(op)

        return self.frame_basis_adjoint @ to_array(op) @ self.frame_basis

    def operator_out_of_frame_basis(self,
                                    op: Union[Operator, np.array]) -> np.array:
        """Transform operator into frame basis.

        Args:
            op: the operator.
        Returns:
            np.array: the operator in the frame basis
        """
        if self._frame_operator is None:
            return to_array(op)

        return self.frame_basis @ to_array(op) @ self.frame_basis_adjoint

    def operators_into_frame_basis(self,
                                   operators: Union[List[Operator], np.array]) -> np.array:
        """Given a list of operators, perform a change of basis on all into
        the frame basis, and return as a 3d array.

        Args:
            operators: list of operators
        """

        return np.array([self.operator_into_frame_basis(o) for o in operators])

    def operators_out_of_frame_basis(self,
                                   operators: Union[List[Operator], np.array]) -> np.array:
        """Given a list of operators, perform a change of basis on all out of
        the frame basis, and return as a 3d array.

        Args:
            operators: list of operators
        """

        return np.array([self.operator_out_of_frame_basis(o) for o in operators])

    def state_into_frame(self,
                         t: float,
                         y: np.array,
                         y_in_frame_basis: Optional[bool] = False,
                         return_in_frame_basis: Optional[bool] = False):
        """Take a state into the frame, i.e. return exp(-tF) @ y.

        Args:
            t: time
            y: state (array of appropriate size)
            y_in_frame_basis: whether or not the array y is already in
                              the basis in which the frame is diagonal
            return_in_frame_basis: whether or not to return the result
                                   in the frame basis
        """
        if self._frame_operator is None:
            return to_array(y)

        out = y

        # if not in frame basis convert it
        if not y_in_frame_basis:
            out = self.state_into_frame_basis(out)

        # go into the frame
        out = np.diag(np.exp(- t * self.frame_diag)) @ out

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.state_out_of_frame_basis(out)

        return out

    def _conjugate_and_add(self,
                           t: float,
                           operator: np.array,
                           op_to_add_in_fb: Optional[np.array] = None,
                           operator_in_frame_basis: Optional[bool] = False,
                           return_in_frame_basis: Optional[bool] = False):
        """Concrete implementation of general helper function for computing
            exp(-tF)Gexp(tF) + B

        Note: B is added in the frame basis before any potential final change
        out of the frame basis.
        """
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
        frame_mat = np.outer(exp_freq.conj(), exp_freq)
        out = frame_mat * out

        if op_to_add_in_fb is not None:
            out = out + to_array(op_to_add_in_fb)

        # if output is requested to not be in the frame basis, convert it
        if not return_in_frame_basis:
            out = self.operator_out_of_frame_basis(out)

        return out

    def operators_into_frame_basis_with_cutoff(self,
                                               operators: Union[np.array, List[Operator]],
                                               cutoff_freq: Optional[float] = None,
                                               carrier_freqs: Optional[np.array] = None):
        """Transform operators into the frame basis, and return two lists of
        operators: one with the 'frequency cutoff' and one with 'conjugate
        frequency cutoff' (see base class documentation).

        Args:
            operators: list of operators
            cutoff_freq: cutoff frequency
            carrier_freqs: list of carrier frequencies

        Returns:
            Tuple[np.array, np.array]: The operators with frequency cutoff
            and conjugate frequency cutoff.
        """

        ops_in_frame_basis = self.operators_into_frame_basis(operators)

        # if no cutoff freq is specified, the two arrays are the same
        if cutoff_freq is None:
            return ops_in_frame_basis, ops_in_frame_basis

        # if no carrier frequencies set, set to 0
        if carrier_freqs is None:
            carrier_freqs = np.zeros(len(operators))

        # create difference matrix for diagonal elements
        dim = len(ops_in_frame_basis[0])
        D_diff = None
        if self._frame_operator is None:
            D_diff = np.zeros((dim, dim))
        else:
            D_diff = np.ones((dim, dim)) * self.frame_diag
            D_diff = D_diff - D_diff.transpose()

        # set up matrix encoding frequencies
        im_angular_freqs = 1j * 2 * np.pi * carrier_freqs
        freq_array = np.array([w + D_diff for w in im_angular_freqs])

        cutoff_array = ((np.abs(freq_array.imag) / (2 * np.pi))
                                < cutoff_freq).astype(int)

        return (cutoff_array * ops_in_frame_basis,
                cutoff_array.transpose([0, 2, 1]) * ops_in_frame_basis)


def to_array(op: Union[Operator, np.array]):
    """Convert an operator, either specified as an `Operator` or an array
    to an array.

    Args:
        op: the operator to represent as an array.
    Returns:
        np.array: op as an array
    """
    if isinstance(op, Operator):
        return op.data
    return op
