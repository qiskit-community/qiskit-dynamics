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
Utility functions for Dynamics Backend.
"""


from typing import Optional, Union, List, Dict

import numpy as np
from qiskit import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix

from qiskit_dynamics.array import Array
from qiskit_dynamics.models import HamiltonianModel, LindbladModel
from qiskit_dynamics.type_utils import to_array


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
    for eigval, evec in zip(evals, evecs.transpose()):

        position = np.argmax(np.abs(evec))
        if position in found_positions:
            raise QiskitError(
                """Dressed-state sorting failed due to non-unique np.argmax(np.abs(evec))
                for eigenvectors."""
            )

        found_positions.append(position)

        dressed_states[:, position] = evec
        dressed_evals[position] = eigval

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
        static_hamiltonian = to_array(model.static_operator)
    else:
        static_hamiltonian = to_array(model.static_hamiltonian)

    static_hamiltonian = 1j * model.rotating_frame.generator_out_of_frame(
        t=0.0, operator=-1j * static_hamiltonian
    )

    return Array(static_hamiltonian, backend="numpy").data


def _get_memory_slot_probabilities(
    probability_dict: Dict,
    memory_slot_indices: List[int],
    num_memory_slots: Optional[int] = None,
    max_outcome_value: Optional[int] = None,
) -> Dict:
    """Construct probability dictionary for memory slot outcomes from a probability dictionary for
    state level measurement outcomes.

    Args:
        probability_dict: A list of probabilities for the otucomes of state measurement. Keys
            are assumed to all be strings of integers of the same length.
        memory_slot_indices: Indices of which memory slots store the digits of the keys of
            probability_dict.
        num_memory_slots: Total number of memory slots for results. If None,
            defaults to the maximum index in memory_slot_indices. The default value
            of unused memory slots is 0.
        max_outcome_value: Maximum value that can be stored in a memory slot. All outcomes higher
            than this will be rounded down.

    Returns:
        Dict: Keys are memory slot outcomes, values are the probabilities of those outcomes.
    """
    num_memory_slots = num_memory_slots or (max(memory_slot_indices) + 1)
    memory_slot_probs = {}
    for level_str, prob in probability_dict.items():
        memory_slot_result = ["0"] * num_memory_slots

        for idx, level in zip(memory_slot_indices, reversed(level_str)):
            if max_outcome_value and int(level) > max_outcome_value:
                level = str(max_outcome_value)
            memory_slot_result[-(idx + 1)] = level

        memory_slot_result = "".join(memory_slot_result)
        if memory_slot_result in memory_slot_probs:
            memory_slot_probs[memory_slot_result] += prob
        else:
            memory_slot_probs[memory_slot_result] = prob

    return memory_slot_probs


def _sample_probability_dict(
    probability_dict: Dict, shots: int, seed: Optional[int] = None
) -> List[str]:
    """Sample outcomes based on probability dictionary.

    Args:
        probability_dict: Dictionary representing probability distribution, with keys being
            outcomes, values being probabilities.
        shots: Number of shots.
        seed: Seed to use in rng construction.

    Return:
        List: of entries of probability_dict, sampled according to the probabilities.
    """
    rng = np.random.default_rng(seed=seed)
    alphabet, probs = zip(*probability_dict.items())
    return rng.choice(alphabet, size=shots, replace=True, p=probs)


def _get_counts_from_samples(samples: list) -> Dict:
    """Count items in list."""
    return dict(zip(*np.unique(samples, return_counts=True)))
