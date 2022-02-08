#%%
from lib2to3.pytree import convert
# from msilib.schema import Error
import scipy.linalg as la
import numpy as np


ham_type = '3_4_5'
#%%
if ham_type == 'base':
    subsystem_dims = [3,3]
    T = 300.0
    risefall = 2.0
    sigma = 15.0
    dim=3

    w_c = 2 * np.pi * 5.105
    w_t = 2 * np.pi * 5.033
    alpha_c = 2 * np.pi * (-0.33534)
    alpha_t = 2 * np.pi * (-0.33834)
    J = 2 * np.pi * 0.002

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.transpose()
    N = np.diag(np.arange(dim))
    ident = np.eye(dim)
    ident2 = np.eye(dim ** 2)

    # operators on the control qubit (first tensor factor)
    a0 = np.kron(a, ident)
    adag0 = np.kron(adag, ident)
    N0 = np.kron(N, ident)

    # operators on the target qubit (first tensor factor)
    a1 = np.kron(ident, a)
    adag1 = np.kron(ident, adag)
    N1 = np.kron(ident, N)

    H0 = (
        w_c * N0
        + 0.5 * alpha_c * N0 @ (N0 - ident2)
        + w_t * N1
        + 0.5 * alpha_t * N1 @ (N1 - ident2)
        + J * (a0 @ adag1 + adag0 @ a1)
    )
    Hdc = 2 * np.pi * (a0 + adag0)
    Hdt = 2 * np.pi * (a1 + adag1)
elif ham_type == '3_4':
    subsystem_dims=[3,4]
    T = 300.0
    risefall = 2.0
    sigma = 15.0
    dim=3
    dim1=4
    dim2=5

    w_c = 2 * np.pi * 5.105
    w_t = 2 * np.pi * 5.033
    w_2 = 2 * np.pi * 5.153
    alpha_c = 2 * np.pi * (-0.33534)
    alpha_t = 2 * np.pi * (-0.33834)
    alpha_2 = 2 * np.pi * (-0.33234)
    J = 2 * np.pi * 0.002
    Jt2 = 2 * np.pi * 0.0022

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.transpose()
    N = np.diag(np.arange(dim))
    ident = np.eye(dim)
    ident2 = np.eye(dim * dim1 * dim2)

    a1 = np.diag(np.sqrt(np.arange(1, dim1)), 1)
    adag1 = a1.transpose()
    N_1 = np.diag(np.arange(dim1))
    ident1 = np.eye(dim1)

    a2 = np.diag(np.sqrt(np.arange(1, dim2)), 1)
    adag2 = a2.transpose()
    N_2 = np.diag(np.arange(dim2))
    ident2 = np.eye(dim2)


    # operators on the control qubit (first tensor factor)
    a0 = np.kron(a, np.kron(ident1, ident2))
    adag0 = np.kron(adag, np.kron(ident1, ident2))
    N0 = np.kron(N, np.kron(ident1, ident2))

    # operators on the target qubit (first tensor factor)
    a1 = np.kron(ident, a1)
    adag1 = np.kron(ident, adag1)
    N1 = np.kron(ident, N_1)

    # operators on the target qubit (first tensor factor)
    a1 = np.kron(ident, a1)
    adag1 = np.kron(ident, adag1)
    N1 = np.kron(ident, N_1)

    H0 = (
        w_c * N0
        + 0.5 * alpha_c * N0 @ (N0 - ident2)
        + w_t * N1
        + 0.5 * alpha_t * N1 @ (N1 - ident2)
        + J * (a0 @ adag1 + adag0 @ a1)
    )
    Hdc = 2 * np.pi * (a0 + adag0)
    Hdt = 2 * np.pi * (a1 + adag1)
elif ham_type == '3_4_5':
    subsystem_dims=[3,4]
    T = 300.0
    risefall = 2.0
    sigma = 15.0
    dim=3
    dim1=4

    w_c = 2 * np.pi * 5.105
    w_t = 2 * np.pi * 5.033
    alpha_c = 2 * np.pi * (-0.33534)
    alpha_t = 2 * np.pi * (-0.33834)
    J = 2 * np.pi * 0.002

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = a.transpose()
    N = np.diag(np.arange(dim))
    ident = np.eye(dim)
    ident2 = np.eye(dim * dim1)

    a1 = np.diag(np.sqrt(np.arange(1, dim1)), 1)
    adag1 = a1.transpose()
    N_1 = np.diag(np.arange(dim1))
    ident1 = np.eye(dim1)


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
        + 0.5 * alpha_c * N0 @ (N0 - ident2)
        + w_t * N1
        + 0.5 * alpha_t * N1 @ (N1 - ident2)
        + J * (a0 @ adag1 + adag0 @ a1)
    )
    Hdc = 2 * np.pi * (a0 + adag0)
    Hdt = 2 * np.pi * (a1 + adag1)
#%%
# Hamiltonian 2 -- 3 level and 4 level qubit
#%%

def labels_generator(subsystem_dims):
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
    labels = [[str(x) for x in lab] for lab in labels]
    labels = [''.join(lab) for lab in labels]
    return labels
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
        lab_excited = ''.join(['0' if i != subsys else '1' for i in range(len(subsystem_dims))])
        # could do lab ground if we don't want to assume labels[0] is '0000'
        # lab_ground = ''.join(['0' for i in range(len(subsystem_dims))])
        # energy = dressed_evals[lab_excited] - dressed_evals[lab_ground]
        try:
            energy = dressed_evals[lab_excited] - dressed_evals[labels[0]]
        except:
            raise Error("missing eigenvalue for excited or base state")

        dressed_freqs.append(energy/(2 * np.pi))

    return dressed_states, dressed_freqs, dressed_evals, dressed_list

#%%

dressed_states, dressed_freqs, dressed_evals, dressed_list = convert_to_dressed(H0, subsystem_dims)
#%%
la.eigh(H0)[1].shape
estates = la.eigh(H0)[1]
# %%
estates[:,1]
dressed_states

# %%

# May want to handle more general formats of the state
# There are state classis in quantum info which have some measurement function probabilities
# may not want them to be quantum info
# handle state vector or density matrix
# maybe don't do it in the dressed state?

# if vector -- np.dot(value.conj())
# elif state.ndim == 2: (density matrix) -> <value, state @ value>
# density matrix rho, and pi is projection operator in some measurement
# probability given density matrix rho of observing pi = <pi, p>
# <A,B> = Trace(A star B)
# pi is projection onto space from dressed space
# so if dressed state is v then pi is v v star
# If you write <pi,rho> = <v v star, p>
# then you get < v, rho v>

#If state is in dressed basis, then each element directly corresponds 
# but since you already know ordering, can just look at the entry of the vector
# dressed state is in computational basis
# so for now keep in standard basis
# if everything starts in matrix can continue just turn everything into dressed basis
# 

def compute_probabilities(state, dressed_states: dict):

    # return probs_dict


    # How do I compute probabilities of a quantum state: 
    # Given a state a0 and vector state |q0>
    # <a0|q0>
    # q0 = vert[0,0,1]
    # a0 = dressed states?
    # so for each state in the dressed state, the probability of it is <a0|q0>
    return {label: (np.abs(np.inner(dressed_states[label].conj(), state)))**2 for label in dressed_states.keys()}
    # return {label: np.inner(state, dressed_states[label]) for label in dressed_states.keys()}
#%%
# state = [1/np.sqrt(2),0,0,0,0,1/np.sqrt(2),0,0,0]
state = [1/np.sqrt(2),0,0,0,0,0,0,0,1/np.sqrt(2),0,0,0]
compute_probabilities(state, dressed_states)
#%%
    
probs = compute_probabilities(state, dressed_states)

def single_sample(sample, probs):
    last_val = 0
    for key, val in probs.items():
        new_val = last_val + val
        if sample <= new_val:
            return key
        last_val = new_val
    # print(new)
    raise Exception("Sample not within probability range, either numpy is broken or the probabilities do not sum to 1")

   

def sample_counts(probs, n_shots):
    samples = np.random.random_sample(n_shots)
    results = np.vectorize(single_sample, excluded='probs')(samples, probs=probs)
    return results
#%%
sample_counts(probs, 100)



#%%


def _first_excited_state(qubit_idx, subsystem_dims):
    """
    Returns the vector corresponding to all qubits in the 0 state, except for
    qubit_idx in the 1 state.
    Parameters:
        qubit_idx (int): the qubit to be in the 1 state
        subsystem_dims (dict): a dictionary with keys being subsystem index, and
                        value being the dimension of the subsystem
    Returns:
        vector: the state with qubit_idx in state 1, and the rest in state 0
    """
    vector = np.array([1.])
    # iterate through qubits, tensoring on the state
    qubit_indices = [int(qubit) for qubit in subsystem_dims]
    qubit_indices.sort()
    for idx in qubit_indices:
        new_vec = np.zeros(subsystem_dims[idx])
        if idx == qubit_idx:
            new_vec[1] = 1
        else:
            new_vec[0] = 1
        vector = np.kron(new_vec, vector)

    return vector