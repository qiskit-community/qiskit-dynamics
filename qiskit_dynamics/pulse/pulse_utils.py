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
Pulse utils for computing dressed states and probabilities
"""


from typing import Optional, Union

import numpy as np
import numpy.linalg as la
from qiskit import QiskitError
from qiskit.quantum_info import DensityMatrix, Statevector


def labels_generator(
    subsystem_dims: list[int], array: Optional[bool] = False
) -> list[Union[str, list[int]]]:
    """Generate labels for a given system in a traditional order incrementing the
    first qubit through all its levels, then incrememnting the next qubit once and
    going back up the levels with the previous qubit, and so on. Can return either
    the string or array version of the labels. The result for a 2x 3 level qubit system
    is ['01', '02', '10', '11', '12', '20', '21', '22'].
    Args:
        subsystem_dims: The dimensions of each subsystem of the general system.
        array : Flag to determine type of label. Defaults to False.

    Returns:
        List of system labels either in a string or list format.
    """
    labels = [[0 for i in range(len(subsystem_dims))]]

    for subsys_ind, dim in enumerate(subsystem_dims):
        new_labels = []
        for state in range(dim)[1:]:
            if subsys_ind == 0:
                label = [0 for i in range(len(subsystem_dims))]
                label[subsys_ind] = state
                new_labels.append(label)
            else:
                for label in labels:
                    new_label = label.copy()
                    new_label[subsys_ind] = state
                    new_labels.append(new_label)
        labels += new_labels

    for l in labels:
        l.reverse()

    if not array:
        labels = [[str(x) for x in lab] for lab in labels]
        labels = ["".join(lab) for lab in labels]

    return labels


def convert_to_dressed(
    static_ham: np.ndarray, subsystem_dims: list[int]
) -> list[dict[str : np.ndarray], list[float], dict[str:float]]:
    """Generate the dressed states for a given static hamiltonian. For each eigenvalue
    of the hamiltonian, match it to an undressed state by finding the argmax of the
    eigenvalue and mapping it to a corresponding undressed label. In addition, calculate
    the dressed frequencies for each subsystem.

    Args:
        static_ham: Time-independent hamiltonian for the system.
        subsystem_dims: Dimensions of the subsystems composing the system.

    Raises:
        QiskitError: Multiple eigenvalues map to the same dressed state.
        QiskitError: No dressed state found for a first excited or base state.

    Returns:
        a dictionary of dressed states, a list of dressed frequencies, a
        dictionary of dressed eigenvalues
    """

    evals, estates = la.eigh(static_ham)

    labels = labels_generator(subsystem_dims)

    dressed_states = {}
    dressed_evals = {}
    dressed_freqs = {}

    for i, estate in enumerate(estates):

        pos = np.argmax(np.abs(estate))
        lab = labels[pos]

        if lab in dressed_states:
            raise QiskitError("Found overlap of dressed states")

        dressed_states[lab] = estate
        dressed_evals[lab] = evals[i]

    dressed_freqs = []
    for subsys, _ in enumerate(subsystem_dims):
        lab_excited = ["0" if i != subsys else "1" for i in range(len(subsystem_dims))]
        lab_excited.reverse()
        lab_excited = "".join(lab_excited)
        try:
            energy = dressed_evals[lab_excited] - dressed_evals[labels[0]]
        except KeyError as nokey:
            raise QiskitError("missing eigenvalue for a first excited or base state") from nokey

        dressed_freqs.append(energy / (2 * np.pi))

    # add testing for dressed frequencies
    # Should dressed frequencies be a dict as well? -- YES -- label by subsystem
    return dressed_states, dressed_freqs, dressed_evals


def compute_probabilities(
    state: Union[np.ndarray, list, Statevector, DensityMatrix], basis_states: dict
) -> dict[str:float]:
    """Compute the probabilities for each state occupation using the formula for each basis state:
        For each basis state d, given input state vector s, we have the probability
       .. math::
        P(d) = (d^* \\cdot s)^2

        Or for a density matrix s
       .. math::
        P(d) = (s * (matmul) d).conj() * d

    Args:
        state: State vector for current state of system.
        basis_states: Dressed state dictionary for system.

    Raises:
        QiskitError: state vector or density matrix has too many dimensions
    Returns:
        dict[str: float]: Dictionary of probabilities for each dressed state.
    """

    state = np.array(state)
    if state.ndim == 1:
        probs = {
            label: (np.abs(np.inner(basis_states[label].conj(), state) ** 2)).real
            for label in basis_states.keys()
        }
    elif state.ndim == 2:
        probs = {
            label: (
                np.abs(np.matmul(np.matmul(state, basis_states[label]).conj(), basis_states[label]))
            )
            for label in basis_states.keys()
        }
        # label: (np.abs(np.inner(basis_states[label].conj(), state) ** 2)).real()
        # for label in basis_states.keys()

    else:
        raise QiskitError("State has too many dimensions")

    sum_probs = sum(list(probs.values()))
    probs = {key: value / sum_probs for key, value in probs.items()}

    return probs


def sample_counts(probs: dict[str:float], n_shots: int, seed: Optional[int] = None) -> list[str]:
    """Sample the probability distribution `n_shot` times.

    Args:
        probs: Probability of dressed state occupation.
        n_shots: Number of samples
        seed: seed for random choice

    Returns:
        list of samples.
    """
    rng = np.random.default_rng(seed)
    return rng.choice(list(probs.keys()), size=n_shots, p=list(probs.values()))
