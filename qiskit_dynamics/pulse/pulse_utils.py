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

"""
Utility functions for pulse simulation.
"""


from typing import Optional, Union, List, Dict

import numpy as np
from qiskit import QiskitError
from qiskit_dynamics.array import Array
from qiskit_dynamics.models import HamiltonianModel, LindbladModel

from qiskit.quantum_info.operators.predicates import is_hermitian_matrix


def _get_dressed_state_decomposition(
    operator: np.ndarray, rtol=1e-8, atol=1e-8
) -> Union[Dict[str, np.ndarray], List[float], Dict[str, float]]:
    """Get the eigenvalues and eigenvectors of a nearly-diagonal hermitian operator, sorted
    according to overlap with the elementary basis.

    This function is essentially a wrapper around ``numpy.linalg.eigh``, but
    sorts the eigenvectors according to the value of ``np.argmax(np.abs(evec))``. It also
    validates that this is unique for each eigenvector.

    Args:
        operator: Hermitian operator.
        subsystem_dims: Dimensions of the subsystems composing the system.
        rtol: Relative tolerance for Hermiticity check.
        atol: Absolute tolerance for Hermiticity check.
    Returns:
        Tuple: a pair of arrays, one containing eigenvalues and one containing corresponding
        eigenvectors.
    Raises:
        QiskitError: If ``np.argmax(np.abs(evec))`` is non-unique across eigenvectors, or if
        operator is not Hermitian.
    """

    if not is_hermitian_matrix(operator, rtol=rtol, atol=atol):
        raise QiskitError("_get_dressed_state_decomposition received non-Hermitian operator.")

    evals, evecs = np.linalg.eigh(np.array(operator))

    dressed_evals = np.zeros_like(evals)
    dressed_states = np.zeros_like(evecs)

    found_positions = []
    for eval, evec in zip(evals, evecs.transpose()):

        position = np.argmax(np.abs(evec))
        if position in found_positions:
            raise QiskitError("Dressed-state sorting failed due to non-unique np.argmax(np.abs(evec)) for eigenvectors.")
        else:
            found_positions.append(position)

        dressed_states[:, position] = evec
        dressed_evals[position] = eval

    return dressed_evals, dressed_states


def _get_lab_frame_static_hamiltonian(model: Union[HamiltonianModel, LindbladModel]) -> np.ndarray:
    """Get the static Hamiltonian in the lab frame and standard basis.

    This function assumes that the model was constructed with operators specified in the lab frame
    (regardless of the rotating frame) and in the standard basis.

    Args:
        model: The model.
    Returns:
        np.ndarray
    """
    static_hamiltonian = None
    if isinstance(model, HamiltonianModel):
        static_hamiltonian = model.static_operator
    else:
        static_hamiltonian = model.static_hamiltonian

    static_hamiltonian = 1j * model.rotating_frame.generator_out_of_frame(
        t=0., operator=-1j * static_hamiltonian
    )

    return np.array(Array(static_hamiltonian).data)
