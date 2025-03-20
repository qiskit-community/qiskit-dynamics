# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
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
Quantum system model
"""

from typing import Optional, List, Union
from copy import copy

import numpy as np

from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.states.quantum_state import QuantumState

from qiskit_dynamics import ArrayLike
from qiskit_dynamics import Solver, Signal
from qiskit_dynamics.systems import Subsystem, DressedBasis, I, X, N
from qiskit_dynamics.systems.abstract_subsystem_operators import AbstractSubsystemOperator


class QuantumSystemModel:
    """Quantum system model class.

    This class represents an abstract quantum system model containing Hamiltonian and/or Lindblad
    terms, specified in terms of the abstract operator instances provided in this module. Once
    constructed, the :meth:`.get_Solver` method can be used to convert the model into a
    :class:`.Solver` instance with a concrete array representation to solve the system for a given
    initial state. Alternatively, the :meth:`.solve` method can be called to solve the system for an
    initial state without needing to work with the :class:`.Solver` directly. See the :mod:`.models`
    module for a concrete description of the Schrodinger and Lindblad master equations.

    Models can be summed together to build more complex models, e.g. for a system with multiple
    subsystems. See the :ref:`Systems Modelling Tutorial <systems modelling tutorial>` for an
    example of intended usage.
    """

    def __init__(
        self,
        static_hamiltonian: Optional[AbstractSubsystemOperator] = None,
        drive_hamiltonian_coefficients: Optional[List[str]] = None,
        drive_hamiltonians: Optional[List[AbstractSubsystemOperator]] = None,
        static_dissipators: Optional[List[AbstractSubsystemOperator]] = None,
        drive_dissipator_coefficients: Optional[List[str]] = None,
        drive_dissipators: Optional[List[AbstractSubsystemOperator]] = None,
    ):
        """Initialize.

        Args:
            static_hamiltonian: The static Hamiltonian.
            drive_hamiltonian_coefficients: A list of string labels for the drive Hamiltonian terms.
            drive_hamiltonians: The Hamiltonian terms with time-dependent coefficients. This is
                mapped to ``hamiltonian_operators`` in :class:`.Solver`.
            static_dissipators: The static dissipator terms.
            drive_dissipator_coefficients: A list of string labels for the drive dissipator terms.
            drive_dissipators: Dissipator terms with time-dependent rates. This is mapped to
                ``dissipator_operators`` in :class:`.Solver`.
        """

        drive_hamiltonians = drive_hamiltonians or []
        static_dissipators = static_dissipators or []
        drive_dissipators = drive_dissipators or []

        self._operators = {
            "static_hamiltonian": static_hamiltonian,
            "drive_hamiltonians": drive_hamiltonians,
            "static_dissipators": static_dissipators,
            "drive_dissipators": drive_dissipators,
        }

        self._drive_hamiltonian_coefficients = drive_hamiltonian_coefficients or []
        self._drive_dissipator_coefficients = drive_dissipator_coefficients or []

        # is this how we want to make this list?
        subsystems = []
        if static_hamiltonian is not None:
            subsystems = copy(static_hamiltonian.subsystems)
        for op in drive_hamiltonians + static_dissipators + drive_dissipators:
            for x in op.subsystems:
                if x not in subsystems:
                    subsystems.append(x)

        self._subsystems = subsystems

    @property
    def subsystems(self):
        """The model subsystems."""
        return self._subsystems

    @property
    def static_hamiltonian(self):
        """The model static Hamiltonian."""
        return self._operators["static_hamiltonian"]

    @property
    def drive_hamiltonians(self):
        """The model drive Hamiltonians."""
        return self._operators["drive_hamiltonians"]

    @property
    def static_dissipators(self):
        """The model static dissipators."""
        return self._operators["static_dissipators"]

    @property
    def drive_dissipators(self):
        """The model drive dissipators."""
        return self._operators["drive_dissipators"]

    @property
    def drive_hamiltonian_coefficients(self):
        """The drive Hamiltonian coefficients."""
        return self._drive_hamiltonian_coefficients

    @property
    def drive_dissipator_coefficients(self):
        """The drive dissipator coefficients."""
        return self._drive_dissipator_coefficients

    def dressed_basis(self, ordered_subsystems: Optional[List] = None, ordering: str = "default"):
        """Get the DressedBasis object for the system.

        Args:
            ordered_subsystems: Subsystems in the desired order.
            ordering: Ordering convention for the eigenvectors.
        """
        ordered_subsystems = ordered_subsystems or self.subsystems
        return DressedBasis.from_hamiltonian(
            self.static_hamiltonian, ordered_subsystems, ordering=ordering
        )

    def get_Solver(
        self,
        rotating_frame: Optional[Union[np.ndarray, AbstractSubsystemOperator]] = None,
        array_library: Optional[str] = None,
        vectorized: bool = False,
        validate: bool = False,
        ordered_subsystems: Optional[List[Subsystem]] = None,
    ):
        """Build concrete operators and instantiate solver.

        Note that the :meth:`.map_signal_dictionary` method can be used to map signals given in a
        dictionary format ``{drive_coefficient: s}``, where ``drive_coefficient`` is a string in
        ``drive_hamiltonian_coefficients + drive_dissipator_coefficients`` and ``s`` is a signal,
        to the required formatting of the ``signals`` argument in :meth:`.Solver.solve`.

        Args:
            rotating_frame: Rotating frame to define the solver in.
            array_library: array library to use (e.g. "numpy", "jax", "jax_sparse", "scipy_sparse")
            vectorized: If doing lindblad simulation, whether or not to vectorize.
            validate: Whether or not to validate the operators.
            ordered_subsystems: Chosen non-standard ordering for building the solver.
        """
        if ordered_subsystems is None:
            ordered_subsystems = self.subsystems

        if self.static_hamiltonian is None:
            static_hamiltonian = None
        else:
            static_hamiltonian = self.static_hamiltonian.matrix(ordered_subsystems)

        if len(self.drive_hamiltonians) == 0:
            drive_hamiltonians = None
        else:
            drive_hamiltonians = np.array(
                [op.matrix(ordered_subsystems) for op in self.drive_hamiltonians]
            )

        if len(self.static_dissipators) == 0:
            static_dissipators = None
        else:
            static_dissipators = np.array(
                [op.matrix(ordered_subsystems) for op in self.static_dissipators]
            )

        if len(self.static_dissipators) == 0:
            drive_dissipators = None
        else:
            drive_dissipators = np.array(
                [op.matrix(ordered_subsystems) for op in self.drive_dissipators]
            )

        if rotating_frame == "static_hamiltonian":
            rotating_frame = static_hamiltonian
        elif isinstance(rotating_frame, AbstractSubsystemOperator):
            rotating_frame = rotating_frame.matrix(ordered_subsystems)

        return Solver(
            static_hamiltonian=static_hamiltonian,
            hamiltonian_operators=drive_hamiltonians,
            static_dissipators=static_dissipators,
            dissipator_operators=drive_dissipators,
            rotating_frame=rotating_frame,
            validate=validate,
            array_library=array_library,
            vectorized=vectorized,
        )

    def map_signal_dictionary(self, signals: List[Union[ArrayLike, Signal]]):
        """Map labelled signal dictionary to the required format for for the signals argument of a
        :class:`.Solver` generated from the :meth:`.get_Solver` method.

        Args:
            signals: Signals in dictionary format ``{label: s}``, for ``label`` a string in
                ``drive_hamiltonian_coefficients + drive_dissipator_coefficients`` and ``s`` a
                signal.
        Returns:
            A container of signals formatted for the ``signals`` argument of the :class:`.Solver`
            method :meth:`.Solver.solve`.
        """
        # order coefficients
        hamiltonian_signals = [
            signals.get(label, 0.0) for label in self._drive_hamiltonian_coefficients
        ]
        dissipator_signals = [
            signals.get(label, 0.0) for label in self._drive_dissipator_coefficients
        ]

        return (
            hamiltonian_signals
            if len(dissipator_signals) == 0
            else (hamiltonian_signals, dissipator_signals)
        )

    def solve(
        self,
        signals: dict,
        t_span: ArrayLike,
        y0: Union[ArrayLike, QuantumState, BaseOperator],
        rotating_frame: Optional[Union[np.ndarray, AbstractSubsystemOperator]] = None,
        array_library: Optional[str] = None,
        vectorized: Optional[bool] = False,
        ordered_subsystems: Optional[List[Subsystem]] = None,
        **kwargs,
    ):
        """Solve the model.

        This method internally constructs a :class:`.Solver` instance with fully-formed arrays
        according to the abstract model specified in this instance, and then solves. Note that the
        ``signals`` argument for this method expects a dictionary format mapping the coefficient
        labels for the drive terms specified at instantiation to the desired coefficient.

        Args:
            signals: Signals in dictionary format ``{label: s}``, where ``label`` is a string in
                ``drive_hamiltonian_coefficients + drive_dissipator_coefficients``, and ``s`` is the
                corresponding singal.
            t_span: Time interval to integrate over.
            y0: Initial state.
            rotating_frame: Rotating frame to transform the model into. Rotating frames which are
                diagonal can be supplied as a 1d array of the diagonal elements, to explicitly
                indicate that they are diagonal.
            array_library: Array library to use for storing operators of underlying model. See the
                :ref:`model evaluation section of the Models API documentation <model evaluation>`
                for a more detailed description of this argument.
            vectorized: If including dissipator terms, whether or not to construct the
                :class:`.LindbladModel` in vectorized form. See the
                :ref:`model evaluation section of the Models API documentation <model evaluation>`
                for a more detailed description of this argument.
            ordered_subsystems: List of :class:`.Subsystem` instances explicitly specifying the
                ordering of the subsystems desired when building the concrete model.
            kwargs: Keyword arguments to pass through to :class:`.Solver.solve`.
        """

        solver = self.get_Solver(
            rotating_frame=rotating_frame,
            array_library=array_library,
            vectorized=vectorized,
            validate=False,
            ordered_subsystems=ordered_subsystems,
        )

        signals = self.map_signal_dictionary(signals)

        return solver.solve(t_span=t_span, y0=y0, signals=signals, **kwargs)

    def __add__(self, other):
        """Add two models."""

        new_operators = {key: op + other._operators[key] for key, op in self._operators.items()}

        return QuantumSystemModel(
            drive_hamiltonian_coefficients=self.drive_hamiltonian_coefficients
            + other.drive_hamiltonian_coefficients,
            drive_dissipator_coefficients=self.drive_dissipator_coefficients
            + other.drive_dissipator_coefficients,
            **new_operators,
        )

    def __str__(self):
        string = "QuantumSystemModel(\n"
        string += f"    static_hamiltonian={self._operators['static_hamiltonian']},\n"
        string += f"    drive_hamiltonian_coefficients={self.drive_hamiltonian_coefficients},\n"
        string += f"    drive_hamiltonians={self._operators['drive_hamiltonians']},\n"
        string += f"    static_dissipators={self._operators['static_dissipators']},\n"
        string += f"    drive_dissipator_coefficients={self.drive_dissipator_coefficients},\n"
        string += f"    drive_dissipators={self._operators['drive_dissipators']},\n"
        string += ")"

        return string

    def _map_model(self, f):
        """Apply a function to the underlying operators, returning a new QuantumSystemModel.

        If an operator or operator list is ``None``, it will remain ``None`` under the mapping.
        """

        static_hamiltonian = None
        if self.static_hamiltonian is not None:
            static_hamiltonian = f(self.static_hamiltonian)

        drive_hamiltonians = [f(x) for x in self.drive_hamiltonians]
        static_dissipators = [f(x) for x in self.static_dissipators]
        drive_dissipators = [f(x) for x in self.drive_dissipators]

        return QuantumSystemModel(
            static_hamiltonian=static_hamiltonian,
            drive_hamiltonian_coefficients=self.drive_hamiltonian_coefficients,
            drive_hamiltonians=drive_hamiltonians,
            static_dissipators=static_dissipators,
            drive_dissipator_coefficients=self.drive_dissipator_coefficients,
            drive_dissipators=drive_dissipators,
        )


class IdealQubit(QuantumSystemModel):
    r"""Simple dynamical model of a quantum system.

    Intended to represent a 2 level system, though can be constructed on higher dimensional
    subsystems. A model with Hamiltonian of the form :math:`H(t) = 2 \pi \nu Z + s(t) 2 \pi r X`,
    with :math:`\nu` being the frequency, :math:`s(t)` the drive term, and :math:`r` the drive
    strength.
    """

    def __init__(self, subsystem, frequency, drive_strength, drive_label=None):
        """Initialize.

        Args:
            subsystem: The subsystem to define the qubit on.
            frequency: The frequency of the qubit.
            drive_strength: The drive strength of the qubit.
            drive_label: The label for the drive term.
        """
        if drive_label is None:
            drive_label = f"d{subsystem.name}"
        super().__init__(
            static_hamiltonian=2 * np.pi * frequency * N(subsystem),
            drive_hamiltonian_coefficients=[drive_label],
            drive_hamiltonians=[2 * np.pi * drive_strength * X(subsystem) * 0.5],
            static_dissipators=[],
            drive_dissipator_coefficients=[],
            drive_dissipators=[],
        )


class DuffingOscillator(QuantumSystemModel):
    r"""Duffing oscillator.

    A model of a transmon with Hamiltonian:
    :math:`H(t) = 2 \pi \nu N + \pi \alpha N(N - I) + s(t) 2 \pi r X`, where :math:`\nu` is the
    frequency, :math:`\alpha` the anharmonicity, :math:`r` is the drive strength, and :math:`s(t)`
    is the drive signal.
    """

    def __init__(self, subsystem, frequency, anharm, drive_strength, drive_label=None):
        """Initialize.

        Args:
            subsystem: The subsystem to define the Duffing oscillator on.
            frequency: The frequency of the oscillator.
            anharm: The anharmonicity of the oscillator.
            drive_strength: The drive strength.
            drive_label: The label for the drive term.
        """
        if drive_label is None:
            drive_label = f"d{subsystem.name}"

        super().__init__(
            static_hamiltonian=2 * np.pi * frequency * N(subsystem)
            + np.pi * anharm * N(subsystem) @ (N(subsystem) + (-1 * I(subsystem))),
            drive_hamiltonian_coefficients=[drive_label],
            drive_hamiltonians=[2 * np.pi * drive_strength * X(subsystem)],
            static_dissipators=[],
            drive_dissipator_coefficients=[],
            drive_dissipators=[],
        )


class ExchangeInteraction(QuantumSystemModel):
    r"""An exchange interaction between two systems.

    Represents the Hamiltonian :math:`H = g X \otimes X`, where :math:`g` is the strength of the
    coupling, and the two :math:`X` operators act on the two subsystems.
    """

    def __init__(self, subsystems, g):
        """Initialize.

        Args:
            g: The coupling strength.
        """
        super().__init__(
            static_hamiltonian=2 * np.pi * g * (X(subsystems[0]) @ X(subsystems[1])),
            drive_hamiltonian_coefficients=[],
            drive_hamiltonians=[],
            static_dissipators=[],
            drive_dissipator_coefficients=[],
            drive_dissipators=[],
        )
