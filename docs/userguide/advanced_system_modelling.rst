.. _systems modelling userguide:

How-to use advanced system modelling functionality
==================================================

The :mod:`.systems` module contains tools for building descriptions of systems...

1. Subsystems and building operators acting on tensor product spaces
--------------------------------------------------------------------

Define some subsystems we will work with.

.. jupyter-execute::

    from qiskit_dynamics.systems import Subsystem

    Q1 = Subsystem("Q1", dim=2)
    Q2 = Subsystem("Q2", dim=2)
    Q3 = Subsystem("Q3", dim=2)

    print(Q1)

Import operators and define some instances.

.. jupyter-execute::

    from qiskit_dynamics.systems import I, X, Y, Z, A, Adag, N

    # X acting on Q1
    X1 = X(Q1)

    # Y acting on Q2
    Y2 = Y(Q2)

    # Z acting on Q3
    Z3 = Z(Q3)

    print(X1)


.. jupyter-execute::

    X1.matrix()

.. jupyter-execute::

    X1.matrix([Q1, Q2])

.. jupyter-execute::

    X1.matrix([Q2, Q1])


We can do algebraic operations to build composite operators.

Add two operators together.

.. jupyter-execute::

    X1 + Y2

.. jupyter-execute::
    
    type(X1 + Y2)

This new composite operator now acts on the two subsystems that ``X1`` and ``Y2`` act on.

.. jupyter-execute::

    (X1 + Y2).subsystems


.. jupyter-execute::

    (X1 + Y2).matrix()

.. jupyter-execute::

    (X1 + Y2).matrix([Q1, Q2, Q3])

Multiply.

.. jupyter-execute::

    X1 @ Z3

Scalar addition and multiplication. Scalars under addition are treated as multiples of the identity.

.. jupyter-execute::

    1 + 2 * X1


2. Define an operator acting on a subspace
------------------------------------------

Task: Define a :math:`2`-dimensional matrix on the first two levels of a system with :math:`4`
levels, with zeroes everywhere else. Effectively, we want to construct :math:`X \oplus 0`, where the
:math:`0` is the :math:`2 \times 2` zero matrix.

.. jupyter-execute::

    # define subsystem for the subspace
    C2 = Subsystem("C2", dim=2)

    # define the higher dimensional space
    C4 = Subsystem("C4", dim=4)

Define a basis for the subspace of ``C4`` spanned by the first 2 standard basis elements.

.. jupyter-execute::

    from qiskit_dynamics.systems import ONBasis
    import numpy as np

    basis = ONBasis(
        basis_vectors=np.eye(4, 2),
        subsystems=[C4]
    )

    # view the basis vectors
    basis.basis_vectors

Construct :math:`X \oplus 0` by expanding ``X(C2)`` into an operator acting on ``C4`` via the above
basis.

.. jupyter-execute::

    from qiskit_dynamics.systems import SubsystemMapping

    injection = SubsystemMapping(
        matrix=basis.basis_vectors,
        in_subsystems=[C2],
        out_subsystems=[C4]
    )

    logical_X = injection(X(C2))
    logical_X

Observe the desired matrix:

.. jupyter-execute::

    logical_X.matrix()


3. Define an operator acting on the logical subspace of a transmon model
------------------------------------------------------------------------

Task: Given a 2 transmon system, construct the opertor "X on qubit 0 in the computational subspace".

Mathematically, this means the matrix :math:`A(X \otimes I)A^\dagger`, where :math:`X`` and
:math:`I` are :math:`2 \times 2` matrices, and is the isometry mapping the two qubit logical space
into the two transmon physical space. Note that in applications, will be defined in terms of the
dressed basis of a Hamiltonian.

To do this, we:
- Define subsystems for both the logical/computational spaces, and the physical spaces.
- Compute the dressed basis of an example 2 transmon Hamiltonian.
- Restrict this basis to the computational states.
- Define the operator :math:`X` acting on the logical qubit :math:`0`.
- "Expand" this operator into the full physical space, creating the desired operator :math:`A(X \otimes I)A^\dagger`
    
.. jupyter-execute::

    # logical subsystems
    L0 = Subsystem("L0", dim=2)
    L1 = Subsystem("L1", dim=2)

    # physical subsystems
    Q0 = Subsystem("Q0", dim=3)
    Q1 = Subsystem("Q1", dim=3)

Define a 2 qubit Hamiltonian, compute the dressed basis, and get the computational states.

.. jupyter-execute::

    from qiskit_dynamics.systems import DressedBasis

    # define a standard Hamiltonian
    H = (2 * np.pi * 5. * N(Q0) +(- 0.33) * np.pi * N(Q0) @ (N(Q0) + (-1 * I(Q0))) +
        2 * np.pi * 5.5 * N(Q1) +(- 0.33) * np.pi * N(Q1) @ (N(Q1) + (-1 * I(Q1))) +
        2 * np.pi * 0.002 * X(Q0) @ X(Q1))

    # Get the dressed basis and the computational states
    dressed_basis = DressedBasis.from_hamiltonian(H, [Q0, Q1])
    computational_states = dressed_basis.computational_states

Define ``X`` acting on the logical states of qubit ``0``.

.. jupyter-execute::

    op = X(L0)


Expand this into an operator on the combined physical system ``[Q0, Q1]``.

.. jupyter-execute::

    injection = SubsystemMapping(
        matrix=computational_states.basis_vectors,
        in_subsystems=[L0, L1],
        out_subsystems=[Q0, Q1]
    )

    injected_X0 = injection(op)

    injected_X0


4. Restrict an operator to a low energy subspace
------------------------------------------------

When defining models on many subsystems, we may want to restrict the model to a low energy subspace.
Here, we:
- Build the static Hamiltonian of a 3 transmon system.
- Restrict it to the at-most-2-excitation subspace.
- Restrict the X operator acting on one of the transmons to the same subspace.

Define a 3 transmon Hamiltonian:

.. jupyter-execute::

    # physical subsystems
    Q0 = Subsystem("Q0", dim=3)
    Q1 = Subsystem("Q1", dim=3)
    Q2 = Subsystem("Q2", dim=3)

    # define a standard Hamiltonian
    H = (
        2 * np.pi * 5. * N(Q0) +(- 0.33) * np.pi * N(Q0) @ (N(Q0) + (-1 * I(Q0))) +
        2 * np.pi * 5.5 * N(Q1) +(- 0.33) * np.pi * N(Q1) @ (N(Q1) + (-1 * I(Q1))) +
        2 * np.pi * 5.3 * N(Q2) +(- 0.33) * np.pi * N(Q2) @ (N(Q2) + (-1 * I(Q2))) +
        2 * np.pi * 0.002 * X(Q0) @ X(Q1) +
        2 * np.pi * 0.002 * X(Q1) @ X(Q2)
    )

Construct the dressed basis and view eigenvalues.

.. jupyter-execute::

    from qiskit_dynamics.systems import DressedBasis

    # Get the dressed basis
    dressed_basis = DressedBasis.from_hamiltonian(H, [Q0, Q1, Q2])
    dressed_basis.evals


Restrict to low energy states below a given cutoff.

.. jupyter-execute::

    low_energy_states = dressed_basis.low_energy_states(cutoff_energy=70.)
    low_energy_states.evals


Observe standard basis labelling.

.. jupyter-execute::

    low_energy_states.labels

Restrict the Hamiltonian to this low energy space. Note that we need to define a ``Subsystem`` on
which this restriction acts.

.. jupyter-execute::

    LESpace = Subsystem("LES", dim=len(low_energy_states))

    restriction = SubsystemMapping(
        matrix=low_energy_states.basis_vectors_adj,
        in_subsystems=[Q0, Q1, Q2],
        out_subsystems=[LESpace]
    )

    low_energy_H = restriction(H)

    np.diag(low_energy_H.matrix())


It is a diagonal matrix whose entries are the low energy eigenvalues.

We can also restrict other operators, e.g. the :math:`X` operator acting on the physical ``Q1``
system. This operator is implicitly expanded into the input space of the ``restriction`` map
before applying the restriction.

.. jupyter-execute::

    drive_op = restriction(X(Q1))
    drive_op