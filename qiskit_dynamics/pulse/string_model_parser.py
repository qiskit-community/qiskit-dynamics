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

"""
Functionality for importing qiskit.pulse model string representation.
"""

from typing import Tuple, List

from qiskit_dynamics.dispatch import Array

def parse_hamiltonian_dict(hamiltonian_dict: dict,
                           subsystem_list: Optional[List[int]] = None) -> Tuple[Array, Array, List[str]]:
    """Convert Hamiltonian string representation into concrete operators
    and an ordered list of channels corresponding to the operators.

    Args:
        hamiltonian_dict: Dictionary representation of Hamiltonian.
                            ********************************************************************************
                            should document this
        subsystem_list: List of qubits to include in the model. If ``None`` all are kept.

    Returns:
        Concrete Array representation of model: An Array for the static hamiltonian,
        an Array for the list of operators with time-dependent coefficients, and a list
        of channel names giving the time-dependent coefficients.
    """

    pass
