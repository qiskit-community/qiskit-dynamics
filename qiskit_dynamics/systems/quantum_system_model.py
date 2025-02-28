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

from qiskit_dynamics import Solver
from qiskit_dynamics.systems import Subsystem, DressedBasis, I, X, N
from qiskit_dynamics.systems.abstract_subsystem_operators import AbstractSubsystemOperator


class QuantumSystemModel:
    """Quantum system model class."""

    def __init__(
        self,
        static_hamiltonian=None,
        drive_hamiltonian_coefficients=None,
        drive_hamiltonians=None,
        static_dissipators=None,
        drive_dissipator_coefficients=None,
        drive_dissipators=None,
    ):
        """Initialize. Lists of operators are assumed to be"""

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

    def map_signal_dictionary(self, signals):
        """Map labelled signal dictionary to the required format for Solver."""
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
        signals,
        t_span,
        y0,
        rotating_frame=None,
        array_library=None,
        vectorized=False,
        ordered_subsystems=None,
        **kwargs,
    ):
        """Solve. Internally constructs a Solver."""

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
        """Add two models.

        To do: Merge operators with the same drive coefficients.
        """

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
    """Simple dynamical model of a quantum system. Intended to represent a 2 level system, though
    can be constructed on higher dimensional subsystems.
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
    """Duffing oscillator."""

    def __init__(self, subsystem, frequency, anharm, drive_strength, drive_label=None):
        """Initialize."""
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
    """Is this actually what's called "exchange interaction?"."""

    def __init__(self, subsystems, g):
        """Add validation of correct number of subsystems."""
        super().__init__(
            static_hamiltonian=2 * np.pi * g * (X(subsystems[0]) @ X(subsystems[1])),
            drive_hamiltonian_coefficients=[],
            drive_hamiltonians=[],
            static_dissipators=[],
            drive_dissipator_coefficients=[],
            drive_dissipators=[],
        )
