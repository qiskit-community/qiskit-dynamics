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

import scipy.linalg as la
import numpy as np

#%%


def labels_generator(subsystem_dims, array=False):
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

#%%


#%%
def convert_to_dressed(static_ham, subsystem_dims):

    # Remap eigenvalues and eigenstates
    evals, estates = la.eigh(static_ham)

    labels = labels_generator(subsystem_dims)

    dressed_states = {}
    dressed_evals = {}
    dressed_freqs = {}

    dressed_list = []

    for i, estate in enumerate(estates.T):

        pos = np.argmax(np.abs(estate))
        lab = labels[pos]

        if lab in dressed_states.keys():
            raise NotImplementedError("Found overlap of dressed states")

        dressed_states[lab] = estate
        dressed_evals[lab] = evals[i]
        dressed_list.append(lab)

    dressed_freqs = []
    print(dressed_states.keys())
    for subsys, dim in enumerate(subsystem_dims):
        lab_excited = "".join(["0" if i != subsys else "1" for i in range(len(subsystem_dims))])
        # could do lab ground if we don't want to assume labels[0] is '0000'
        # lab_ground = ''.join(['0' for i in range(len(subsystem_dims))])
        # energy = dressed_evals[lab_excited] - dressed_evals[lab_ground]
        try:
            energy = dressed_evals[lab_excited] - dressed_evals[labels[0]]
        except:
            raise Error("missing eigenvalue for excited or base state")

        dressed_freqs.append(energy / (2 * np.pi))

    return dressed_states, dressed_freqs, dressed_evals, dressed_list


def compute_probabilities(state, dressed_states: dict):

    return {
        label: (np.abs(np.inner(dressed_states[label].conj(), state))) ** 2
        for label in dressed_states.keys()
    }


def single_sample(sample, probs):
    last_val = 0
    for key, val in probs.items():
        new_val = last_val + val
        if sample <= new_val:
            return key
        last_val = new_val
    raise Exception(
        "Sample not within probability range, either numpy is broken or the probabilities do not sum to 1"
    )


def sample_counts(probs, n_shots):
    samples = np.random.random_sample(n_shots)
    results = np.vectorize(single_sample, excluded="probs")(samples, probs=probs)
    return results

#%%
# import numpy as np
# #%%
# subsystem_dims = [3]
def generate_ham(subsystem_dims):
    dim = subsystem_dims[0]
    if len(subsystem_dims) > 1:
        dim1 = subsystem_dims[1]
        ident2q = np.eye(dim * dim1)
        a1 = np.diag(np.sqrt(np.arange(1, dim1)), 1)
        adag1 = a1.transpose()
        N_1 = np.diag(np.arange(dim1))
        ident1 = np.eye(dim1)
    if len(subsystem_dims) == 3:
        dim2 = subsystem_dims[2]
        ident3q = np.eye(dim * dim1 * dim2)
        a2 = np.diag(np.sqrt(np.arange(1, dim2)), 1)
        adag2 = a2.transpose()
        N_2 = np.diag(np.arange(dim2))
        ident2 = np.eye(dim2)

    w_c = 2 * np.pi * 5.105
    w_t = 2 * np.pi * 5.033
    w_2 = 2 * np.pi * 5.53
    alpha_c = 2 * np.pi * (-0.33534)
    alpha_t = 2 * np.pi * (-0.33834)
    alpha_2 = 2 * np.pi * (-0.33234)
    J = 2 * np.pi * 0.002
    J2 = 2 * np.pi * 0.0021

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.transpose()
    N = np.diag(np.arange(dim))
    ident = np.eye(dim)

    if len(subsystem_dims) == 1:
        # operators on the control qubit (first tensor factor)
        N0 = N

        H0 = (
            w_c * N0
            + 0.5 * alpha_c * N0 @ (N0 - ident)
        )


    elif len(subsystem_dims) == 2:

        # operators on the control qubit (first tensor factor)
        a0 = np.kron(a, ident1)
        adag0 = np.kron(adag, ident1)
        N0 = np.kron(N, ident1)

        # operators on the target qubit (first tensor factor)
        a1 = np.kron(ident, a1)
        adag1 = np.kron(ident, adag1)
        N1 = np.kron(ident, N_1)
        H0 = (
            w_c * N0
            + 0.5 * alpha_c * N0 @ (N0 - ident2q)

            + w_t * N1
            + 0.5 * alpha_t * N1 @ (N1 - ident2q)

            + J * (a0 @ adag1 + adag0 @ a1)
        )
    elif len(subsystem_dims) == 3:

        # operators on the control qubit (first tensor factor)
        a0 = np.kron(a, ident1)
        a0 = np.kron(a0, ident2)
        adag0 = np.kron(adag, ident1)
        adag0 = np.kron(adag0, ident2)
        N0 = np.kron(N, ident1)
        N0 = np.kron(N0, ident2)

        # operators on the target qubit (first tensor factor)
        a1 = np.kron(ident, a1)
        a1 = np.kron(a1, ident2)
        adag1 = np.kron(ident, adag1)
        adag1 = np.kron(adag1, ident2)
        N1 = np.kron(ident, N_1)
        N1 = np.kron(N_2, N1)

        # operators on the third qubit (first tensor factor)
        a2 = np.kron(ident1, a2)
        a2 = np.kron(ident, a2)
        adag2 = np.kron(ident1, adag2)
        adag2 = np.kron(ident, adag2)
        N2 = np.kron(ident1, N_2)
        N2 = np.kron(ident, N2)

        H0 = (
            w_c * N0
            + 0.5 * alpha_c * N0 @ (N0 - ident3q)

            + w_t * N1
            + 0.5 * alpha_t * N1 @ (N1 - ident3q)

            + w_2 * N2
            + 0.5 * alpha_2 * N2 @ (N2 - ident3q)

            + J * (a0 @ adag1 + adag0 @ a1)
            + J2 * (a1 @ adag2 + adag1 @ a2)
        )
    return H0
#     # Hd0 = 2 * np.pi * (a0 + adag0)
#     # Hd1 = 2 * np.pi * (a1 + adag1)

# # %%
# subsystem_dims=[3,3]
# H0 = generate_ham(subsystem_dims)
# dressed_states, dressed_freqs, dressed_evals, dressed_list= convert_to_dressed(H0, subsystem_dims)

# q1 = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
# q2 = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
# state = np.kron(q1, q2)
# probs = compute_probabilities(state, dressed_states=dressed_states)
# samples = sample_counts(probs, 100)
# samples
# probs
# #%%
# def test1():
#     subsystem_dims=[3,3]
#     H0 = generate_ham(subsystem_dims)
#     dressed_states, dressed_freqs, dressed_evals, dressed_list= convert_to_dressed(H0, subsystem_dims)
#     q1 = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
#     q2 = [0, 1/np.sqrt(2), 1/np.sqrt(2)]
#     state = np.kron(q1, q2)
#     probs = compute_probabilities(state, dressed_states=dressed_states)
#     return probs
# #%%
# test1()
#     # assert(probs) 
# #%%
# # %%
# a = [0,1/np.sqrt(2),1/np.sqrt(2)]
# b = [1,0,0,0]
# c = [1,0,0,0,0]
# first = np.kron(a,b)
# second = np.kron(first, c)
# first
# state = second
# second
# #%%
# probs = compute_probabilities(a, dressed_states=dressed_states)
# samples = sample_counts(probs, 100)
# samples
# # %%
# H0.shape

# # %%


# dim = 3
# evals, estates = np.linalg.eigh(H0)
# E00, dressed00 = get_dressed_state_and_energy(0, 0, dim, B.transpose())
# E01, dressed01 = get_dressed_state_and_energy(0, 1, dim, B.transpose())
# E10, dressed10 = get_dressed_state_and_energy(1, 0, dim, B.transpose())
# E11, dressed11 = get_dressed_state_and_energy(1, 1, dim, B.transpose())

# assert(np.max(dressed00 - dressed_states['00'] < 1e-12))
# assert(np.max(dressed01 - dressed_states['01'] < 1e-12))
# assert(np.max(dressed10 - dressed_states['10'] < 1e-12))
# assert(np.max(dressed11 - dressed_states['11'] < 1e-12))
# # %%

# labels = labels_generator(subsystem_dims, array=True)
# str_labels = labels_generator(subsystem_dims, array=False)
# for str_label, label in zip(str_labels, labels):
#     id = np.argmax(np.abs(dressed_states[str_label]))
#     labels[id]
#     assert((labels[id] == label))
# %%
def dressed_tester(dressed_states, subsystem_dims):
    print(subsystem_dims)
    labels = labels_generator(subsystem_dims, array=True)
    str_labels = labels_generator(subsystem_dims, array=False)
    for str_label, label in zip(str_labels, labels):
        id = np.argmax(np.abs(dressed_states[str_label]))
        labels[id]
        # self.assertTrue((labels[id] == label))


subsystem_dims = [3]
H0 = generate_ham(subsystem_dims)
dressed_states, dressed_freqs, dressed_evals, dressed_list= convert_to_dressed(H0, subsystem_dims)
dressed_tester(dressed_states, subsystem_dims)

# %%
