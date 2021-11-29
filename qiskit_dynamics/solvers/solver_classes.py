# -*- coding: utf-8 -*-

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

r"""
Solver classes.
"""


from typing import Optional, Union, Tuple, Any, Type, List
from copy import copy

import numpy as np

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit.circuit import Gate, QuantumCircuit
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info import SuperOp, Operator, Statevector, DensityMatrix

from qiskit_dynamics.models import (
    BaseGeneratorModel,
    HamiltonianModel,
    LindbladModel,
    RotatingFrame,
    rotating_wave_approximation,
)
from qiskit_dynamics.signals import Signal, SignalList
from qiskit_dynamics.array import Array

from .solver_functions import solve_lmde
from .solver_utils import is_lindblad_model_vectorized, is_lindblad_model_not_vectorized


class Solver:
    """Solver class for simulating both Hamiltonian and Lindblad dynamics, with high
    level type-handling of input states.

    Given the components of a Hamiltonian and optional dissipators, this class will
    internally construct either a :class:`HamiltonianModel` or :class:`LindbladModel`
    instance.

    Transformations on the model can be specified via the optional arguments:

    * ``rotating_frame``: Transforms the model into a rotating frame. Note that
      operator specifying the frame will be substracted from the static_hamiltonian.
      If supplied as a 1d array, ``rotating_frame`` is interpreted as the diagonal
      elements of a diagonal matrix. See :class:`~qiskit_dynamics.models.RotatingFrame` for details.
    * ``in_frame_basis``: Whether to represent the model in the basis in which the frame
      operator is diagonal, henceforth called the "frame basis".
      If ``rotating_frame`` is ``None`` or was supplied as a 1d array,
      this kwarg has no effect. If ``rotating_frame`` was specified as a 2d array,
      the frame basis is hte diagonalizing basis supplied by ``np.linalg.eigh``.
      If ``in_frame_basis==True``, calls to ``solve``, this objects behaves as if all
      operators were supplied in the frame basis: calls to ``solve`` will assume the initial
      state is supplied in the frame basis, and the results will be returned in the frame basis.
      If ``in_frame_basis==False``, the system will still be solved in the frame basis for
      efficiency, however the initial state (and final output states) will automatically be
      transformed into (and, respectively, out of) the frame basis.
    * ``rwa_cutoff_freq``: Performs a rotating wave approximation (RWA) on the model
      with cutoff frequency ``rwa_cutoff_freq``. See
      :func:`~qiskit_dynamics.models.rotating_wave_approximation`
      for details.

      .. note::
            When using the ``rwa_cutoff_freq`` optional argument,
            :class:`~qiskit_dynamics.solvers.solver_classes.Solver` cannot be instantiated within
            a JAX-transformable function. However, after construction, instances can
            still be used within JAX-transformable functions regardless of whether an
            ``rwa_cutoff_freq`` is set.

    .. note::
        Modifications to the underlying model after instantiation may be made
        directly via the ``model`` property of this class. However,
        the getting and setting of model signals should be done via the ``signals`` property
        of this class, which manages signal transformations required in
        the case that a rotating wave approximation is made.

    The evolution given by the model can be simulated by calling
    :meth:`~qiskit_dynamics.solvers.Solver.solve`, which calls
    calls :func:`~qiskit_dynamics.solve.solve_lmde`, and does various automatic
    type handling operations for :mod:`qiskit.quantum_info` state and super operator types.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[Array] = None,
        hamiltonian_operators: Optional[Array] = None,
        hamiltonian_signals: Optional[Union[List[Signal], SignalList]] = None,
        static_dissipators: Optional[Array] = None,
        dissipator_operators: Optional[Array] = None,
        dissipator_signals: Optional[Union[List[Signal], SignalList]] = None,
        rotating_frame: Optional[Union[Array, RotatingFrame]] = None,
        in_frame_basis: bool = False,
        evaluation_mode: str = "dense",
        rwa_cutoff_freq: Optional[float] = None,
        validate: bool = True,
    ):
        """Initialize solver with model information.

        Args:
            static_hamiltonian: Constant Hamiltonian term. If a ``rotating_frame``
                                is specified, the ``frame_operator`` will be subtracted from
                                the static_hamiltonian.
            hamiltonian_operators: Hamiltonian operators.
            hamiltonian_signals: Coefficients for the Hamiltonian operators.
            static_dissipators: Constant dissipation operators.
            dissipator_operators: Dissipation operators with time-dependent coefficients.
            dissipator_signals: Optional time-dependent coefficients for the dissipators. If
                                ``None``, coefficients are assumed to be the constant ``1.``.
            rotating_frame: Rotating frame to transform the model into. Rotating frames which
                            are diagonal can be supplied as a 1d array of the diagonal elements,
                            to explicitly indicate that they are diagonal.
            in_frame_basis: Whether to represent the model in the basis in which the rotating
                            frame operator is diagonalized. See class documentation for a more
                            detailed explanation on how this argument affects object behaviour.
            evaluation_mode: Method for model evaluation. See documentation for
                             ``HamiltonianModel.evaluation_mode`` or
                             ``LindbladModel.evaluation_mode``.
                             (if dissipators in model) for valid modes.
            rwa_cutoff_freq: Rotating wave approximation cutoff frequency. If ``None``, no
                             approximation is made.
            validate: Whether or not to validate Hamiltonian operators as being Hermitian.
        """

        model = None
        if static_dissipators is None and dissipator_operators is None:
            model = HamiltonianModel(
                static_operator=static_hamiltonian,
                operators=hamiltonian_operators,
                signals=hamiltonian_signals,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )
            self._signals = hamiltonian_signals
        else:
            model = LindbladModel(
                static_hamiltonian=static_hamiltonian,
                hamiltonian_operators=hamiltonian_operators,
                hamiltonian_signals=hamiltonian_signals,
                static_dissipators=static_dissipators,
                dissipator_operators=dissipator_operators,
                dissipator_signals=dissipator_signals,
                rotating_frame=rotating_frame,
                in_frame_basis=in_frame_basis,
                evaluation_mode=evaluation_mode,
                validate=validate,
            )
            self._signals = (hamiltonian_signals, dissipator_signals)

        self._rwa_signal_map = None
        if rwa_cutoff_freq is not None:
            model, rwa_signal_map = rotating_wave_approximation(
                model, rwa_cutoff_freq, return_signal_map=True
            )
            self._rwa_signal_map = rwa_signal_map

        self._model = model

    @property
    def model(self) -> Union[HamiltonianModel, LindbladModel]:
        """The model of the system, either a Hamiltonian or Lindblad model."""
        return self._model

    @property
    def signals(self) -> SignalList:
        """The signals used in the solver.

        These will be different from the signals in the model if a rotating wave approximation
        was made.
        """
        return self._signals

    @signals.setter
    def signals(self, new_signals: Union[List[Signal], SignalList]):
        """Set signals for the solver, and pass to the model."""
        self._signals = new_signals
        if self._rwa_signal_map is not None:
            new_signals = self._rwa_signal_map(new_signals)
        self.model.signals = new_signals

    def copy(self) -> "Solver":
        """Return a copy of self."""
        return copy(self)

    def solve(
        self, t_span: Array, y0: Union[Array, QuantumState, BaseOperator], **kwargs
    ) -> OdeResult:
        r"""Solve the dynamical problem.

        Calls :func:`~qiskit_dynamics.solvers.solve_lmde`, and returns an `OdeResult`
        object in the style of `scipy.integrate.solve_ivp`, with results
        formatted to be the same types as the input. See Additional Information
        for special handling of various input types.

        Args:
            t_span: Time interval to integrate over.
            y0: Initial state.
            kwargs: Keyword args passed to :func:`~qiskit_dynamics.solvers.solve_lmde`.

        Returns:
            OdeResult: object with formatted output types.

        Raises:
            QiskitError: Initial state ``y0`` is of invalid shape.

        Additional Information:

            The behaviour of this method is impacted by the input type of ``y0``:

             * If ``y0`` is an ``Array``, it is passed directly to
                :func:`~qiskit_dynamics.solve_lmde` as is. Acceptable array shapes are
                determined by the model type and evaluation mode.
             * If ``y0`` is a subclass of :class:`qiskit.quantum_info.QuantumState`:

                 * If ``self.model`` is a :class:`~qiskit_dynamics.models.LindbladModel`,
                    ``y0`` is converted to a :class:`DensityMatrix`. Further, if the model
                    evaluation mode is vectorized ``y0`` will be suitably reshaped for solving.
                 * If ``self.model`` is a :class:`~qiskit_dynamics.models.HamiltonianModel`,
                    and ``y0`` a :class:`DensityMatrix`, the full unitary will be simulated,
                    and the evolution of ``y0`` is attained via conjugation.

             * If ``y0`` is a subclass of :class`qiskit.quantum_info.QuantumChannel`, the full
                evolution map will be computed and composed with ``y0``; either the unitary if
                ``self.model`` is a :class:`~qiskit_dynamics.models.HamiltonianModel`, or the full
                Lindbladian ``SuperOp`` if the model is a
                :class:`~qiskit_dynamics.models.LindbladModel`.

        """

        # convert types
        if isinstance(y0, QuantumState) and isinstance(self.model, LindbladModel):
            y0 = DensityMatrix(y0)

        y0, y0_cls = initial_state_converter(y0, return_class=True)

        # validate types
        if (y0_cls is SuperOp) and is_lindblad_model_not_vectorized(self.model):
            raise QiskitError(
                """Simulating SuperOp for a LinbladModel requires setting
                vectorized evaluation. Set LindbladModel.evaluation_mode to a vectorized option.
                """
            )

        # modify initial state for some custom handling of certain scenarios
        y_input = y0

        # if Simulating density matrix or SuperOp with a HamiltonianModel, simulate the unitary
        if y0_cls in [DensityMatrix, SuperOp] and isinstance(self.model, HamiltonianModel):
            y0 = np.eye(self.model.dim, dtype=complex)
        # if LindbladModel is vectorized and simulating a density matrix, flatten
        elif (
            (y0_cls is DensityMatrix)
            and isinstance(self.model, LindbladModel)
            and "vectorized" in self.model.evaluation_mode
        ):
            y0 = y0.flatten(order="F")

        # validate y0 shape before passing to solve_lmde
        if isinstance(self.model, HamiltonianModel) and (
            y0.shape[0] != self.model.dim or y0.ndim > 2
        ):
            raise QiskitError("""Shape mismatch for initial state y0 and HamiltonianModel.""")
        if is_lindblad_model_vectorized(self.model) and (
            y0.shape[0] != self.model.dim ** 2 or y0.ndim > 2
        ):
            raise QiskitError(
                """Shape mismatch for initial state y0 and LindbladModel
                                 in vectorized evaluation mode."""
            )
        if is_lindblad_model_not_vectorized(self.model) and y0.shape[-2:] != (
            self.model.dim,
            self.model.dim,
        ):
            raise QiskitError("""Shape mismatch for initial state y0 and LindbladModel.""")

        results = solve_lmde(generator=self.model, t_span=t_span, y0=y0, **kwargs)

        # handle special cases
        if y0_cls is DensityMatrix and isinstance(self.model, HamiltonianModel):
            # conjugate by unitary
            out = Array(results.y)
            results.y = out @ y_input @ out.conj().transpose((0, 2, 1))
        elif y0_cls is SuperOp and isinstance(self.model, HamiltonianModel):
            # convert to SuperOp and compose
            out = Array(results.y)
            results.y = (
                np.einsum("nka,nlb->nklab", out.conj(), out).reshape(
                    out.shape[0], out.shape[1] ** 2, out.shape[1] ** 2
                )
                @ y_input
            )
        elif (y0_cls is DensityMatrix) and is_lindblad_model_vectorized(self.model):
            results.y = Array(results.y).reshape((len(results.y),) + y_input.shape, order="F")

        if y0_cls is not None:
            results.y = [final_state_converter(yi, y0_cls) for yi in results.y]

        return results


def initial_state_converter(
    obj: Any, return_class: bool = False
) -> Union[Array, Tuple[Array, Type]]:
    """Convert initial state object to an Array.

    Args:
        obj: An initial state.
        return_class: Optional. If True return the class to use
                      for converting the output y Array.

    Returns:
        Array: the converted initial state if ``return_class=False``.
        tuple: (Array, class) if ``return_class=True``.
    """
    # pylint: disable=invalid-name
    y0_cls = None
    if isinstance(obj, Array):
        y0, y0_cls = obj, None
    if isinstance(obj, QuantumState):
        y0, y0_cls = Array(obj.data), obj.__class__
    elif isinstance(obj, QuantumChannel):
        y0, y0_cls = Array(SuperOp(obj).data), SuperOp
    elif isinstance(obj, (BaseOperator, Gate, QuantumCircuit)):
        y0, y0_cls = Array(Operator(obj.data)), Operator
    else:
        y0, y0_cls = Array(obj), None
    if return_class:
        return y0, y0_cls
    return y0


def final_state_converter(obj: Any, cls: Optional[Type] = None) -> Any:
    """Convert final state Array to custom class.

    Args:
        obj: final state Array.
        cls: Optional. The class to convert to.

    Returns:
        Any: the final state.
    """
    if cls is None:
        return obj

    if issubclass(cls, (BaseOperator, QuantumState)) and isinstance(obj, Array):
        return cls(obj.data)

    return cls(obj)
