.. _systems modelling userguide:

How-to use advanced system modelling functionality
==================================================

The :mod:`.systems` module contains tools for building and representing models of systems on tensor
product spaces. The high level usage of this module is demonstrated in the :ref:`Building and solving
models of quantum systems <systems modelling tutorial>` tutorial, in which pre-built models are used
to simulate transmon systems. This userguide walks through how to use the underlying operator 
construction tools to build new models, or to modify models in non-trivial ways: e.g. restriction to
a subspace of interest.

This user-guide walks through the following tasks:

1. How-to define operators acting on tensor product spaces.
2. How-to define an operator acting on a subspace.
3. How-to define an operator acting on the logical subspace of a multi-transmon model.
4. How-to restrict an operator to a low energy subspace of a 3-transmon model.

1. How-to build operators acting on tensor product spaces
---------------------------------------------------------

Here we will walk-through the construction of operators acting on a tri-partite system. First, we 
define three 2-dimensional subsystems:

.. jupyter-execute::

    from qiskit_dynamics.systems import Subsystem

    Q1 = Subsystem("Q1", dim=2)
    Q2 = Subsystem("Q2", dim=2)
    Q3 = Subsystem("Q3", dim=2)

Define several operators acting on the individual subsystems.

.. jupyter-execute::

    from qiskit_dynamics.systems import I, X, Y, Z, N

    # X acting on Q1
    X1 = X(Q1)

    # Y acting on Q2
    Y2 = Y(Q2)

    # Z acting on Q3
    Z3 = Z(Q3)

    print(X1)

Note that while these operators are defined only on individual subsystems, if they are used in the
context of multi-subsystem models, the operators implicitly act as the identity on the subsystems
lying outside the operator's definition. E.g. ``X(Q1)`` is defined on ``Q1`` only, but when thought
of as an operator on :math:`Q_1 \otimes Q_2`, it represents the operator :math:`X \otimes I`.

Once an operator is construct, the :meth:`matrix` method returns its matrix form. By default, the
matrix is constructed only on the subsystems the operator explicitly acts on:

.. jupyter-execute::

    X1.matrix()

To construct the matrix corresponding to ``X1`` when viewed as an operator acting on the combined
tensor product space :math:`Q_1 \otimes Q_2`, explicitly pass the list of subsystems to the
:meth:`matrix` method:

.. jupyter-execute::

    X1.matrix([Q1, Q2])

We can also reverse the ordering of the tensor product representation by reversing the order of the
subsystem list:

.. jupyter-execute::

    X1.matrix([Q2, Q1])


More complicated operators can be built through algebraic operations, e.g. addition:

.. jupyter-execute::

    X1 + Y2

This new composite operator acts on the combined set of subsystems that ``X1`` and ``Y2`` act on:

.. jupyter-execute::

    (X1 + Y2).subsystems

When calling the :meth:`matrix` method, a matrix will be constructed on the tensor product system
in the above order.

.. jupyter-execute::

    (X1 + Y2).matrix()

Similarly, we can build the matrix for ``X1 + Y2`` when viewed as an operator on the tripartite
system :math:`Q_1 \otimes Q_2 \otimes Q_3`.

.. jupyter-execute::

    (X1 + Y2).matrix([Q1, Q2, Q3])

Matrix multiplication can also be performed:

.. jupyter-execute::

    X1 @ Z3

In the above case, as ``X1`` acts on ``Q1`` and ``Z3`` acts on ``Q3``, ``X1 @ Z3`` represents the
operator :math:`X \otimes Z` acting on the space :math:`Q_1 \otimes Q_3`.

Lastly, we can multiply and add scalars to operators. Scalars under addition are treated as
multiples of the identity.

.. jupyter-execute::

    1 + 2 * X1


2. How-to define an operator acting on a subspace
-------------------------------------------------

It is common in quantum information to define operators on *subspaces*, e.g. the computational
subspace of a physical system. Here we walk through constructing an :math:`X` operator on the
first two levels of a :math:`4`-dimensional system. In mathematical notation, we want to construct
the operator :math:`X \oplus 0`, where both :math:`X` and :math:`0` are :math:`2 \times 2` matrices.

First, define :class:`Subsystem` instances representing both the :math:`2`-dimensional subspace and
the full :math:`4`-dimensional space.

.. jupyter-execute::

    # define subsystem for the subspace
    C2 = Subsystem("C2", dim=2)

    # define the higher dimensional space
    C4 = Subsystem("C4", dim=4)

Next, define a :class:`ONBasis` instance for the subspace of ``C4`` representing how the standard
basis elements of ``C2`` are mapped into ``C4``. Here we will use the first two standard basis
elements of ``C4``, representing the first two levels of ``C4``.

.. jupyter-execute::

    from qiskit_dynamics.systems import ONBasis
    import numpy as np

    basis = ONBasis(
        basis_vectors=np.eye(4, 2),
        subsystems=[C4]
    )

    # view the basis vectors
    basis.basis_vectors

Using the basis, we explicitly construct the injection of ``C2`` into ``C4`` using the
:class:`SubsystemMapping` class. This class represents a linear map between vector spaces
:math:`V \rightarrow W` of the form :math:`v \mapsto Av` for an operator :math:`A`. Acting on an
operator :math:`X`, a :class:`SubsystemMapping` will perform the transformation
:math:`X \mapsto A X A^\dagger`.

Here, the operator :math:`A` is defined as a matrix given by the first two basis elements of ``C4``:

.. jupyter-execute::

    from qiskit_dynamics.systems import SubsystemMapping

    injection = SubsystemMapping(
        matrix=basis.basis_vectors,
        in_subsystems=[C2],
        out_subsystems=[C4]
    )

:math:`X` acting on the first two levels of ``C4`` is then constructed as:

.. jupyter-execute::

    subspace_X = injection(X(C2))
    subspace_X

Observe the desired matrix:

.. jupyter-execute::

    subspace_X.matrix()


3. How-to define an operator acting on the logical subspace of a multi-transmon model
-------------------------------------------------------------------------------------

In this section we work through a more advanced version of the previous example. Here, we consider
the problem of constructing the operator ":math:`X` acting on the computional subspace of the first
qubit in a two-transmon system". Mathematically, this means the matrix
:math:`A(X \otimes I)A^\dagger`, where :math:`X` and :math:`I` are :math:`2 \times 2` matrices, and
:math:`A` is the isometry mapping the two qubit computational subspace (the first 4 energy levels)
into the two transmon physical space.

For this, we walk through the following steps:

- Define subsystems for both the logical/computational spaces, and the physical spaces.
- Construct the standard static Hamiltonian for a 2 transmon model, and compute the dressed basis
  (the basis of energy eigenstates).
- Construct a basis for the computational subspace within the physical space.
- Define the operator :math:`X` acting on the logical qubit :math:`0`.
- "Expand" this operator into the full physical space, creating the desired operator
  :math:`A(X \otimes I)A^\dagger`

First, construct the :class:`Subsystem` instances we will work with:

.. jupyter-execute::

    # logical subsystems
    L0 = Subsystem("L0", dim=2)
    L1 = Subsystem("L1", dim=2)

    # physical subsystems
    Q0 = Subsystem("Q0", dim=3)
    Q1 = Subsystem("Q1", dim=3)

Define the 2 transmon Hamiltonian.

.. jupyter-execute::

    # define a standard Hamiltonian
    H = (2 * np.pi * 5. * N(Q0) +(- 0.33) * np.pi * N(Q0) @ (N(Q0) + (-1 * I(Q0))) +
        2 * np.pi * 5.5 * N(Q1) +(- 0.33) * np.pi * N(Q1) @ (N(Q1) + (-1 * I(Q1))) +
        2 * np.pi * 0.002 * X(Q0) @ X(Q1))

Compute the dressed basis, and retrieve the :class:`ONBasis` instance corresponding to the
computational states.

.. jupyter-execute::

    from qiskit_dynamics.systems import DressedBasis

    # Get the dressed basis with an explicit tensor product ordering
    dressed_basis = DressedBasis.from_hamiltonian(H, [Q0, Q1])

    # retrieve the computational states
    computational_states = dressed_basis.computational_states

Define the mapping of the logical space :math:`L_0 \otimes L_1` into the computational subspace of
the physical space :math:`Q_0 \otimes Q_1` specified by the matrix of basis vectors for the
computational subspace.

.. jupyter-execute::

    injection = SubsystemMapping(
        matrix=computational_states.basis_vectors,
        in_subsystems=[L0, L1],
        out_subsystems=[Q0, Q1]
    )


Finally, define ``X`` acting on ``L0``, and inject it into the full two-transmon physical space
using ``injection``. Note that as the injection acts on the combined :math:`L_0 \otimes L_1` system,
``X(L0)`` will be treated as ``X(L0) @ I(L1)`` when performing the injection (i.e. with implicit
identity on :math:`L_1`).

.. jupyter-execute::

    op = X(L0)

    injected_X0 = injection(op)

    injected_X0


4. How-to restrict an operator to a low energy subspace of a 3-transmon model
-----------------------------------------------------------------------------

Similarly to defining an operator on a subspace and expanding it into the full space, we may want to
restrict on operator or model to a subspace. For example, restricting a model to a low energy
subspace is a common technique to reduce the dimension of a model.

Here, we walk through the problem of restricting an operator to a low energy subspace of a 3
transmon system with the following steps:

- Build the static Hamiltonian of a 3 transmon system.
- Restrict it to a subspace with bounded energy.
- Restrict the X operator acting on one of the transmons to the same subspace.

Define a 3 transmon static Hamiltonian:

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


Construct a basis for a low energy subspace below a given cutoff.

.. jupyter-execute::

    low_energy_states = dressed_basis.low_energy_states(cutoff_energy=70.)
    low_energy_states.evals


Observe the standard basis labelling, and note that this energy cutoff happens to correspond to the
subspace with at most 2-excitations in the full system.

.. jupyter-execute::

    low_energy_states.labels

Restrict the Hamiltonian to this low energy space. Note that we first need to define a
:class:`Subsystem` instance representing this subspace in isolation.

.. jupyter-execute::

    LESpace = Subsystem("LES", dim=len(low_energy_states))

    restriction = SubsystemMapping(
        matrix=low_energy_states.basis_vectors_adj,
        in_subsystems=[Q0, Q1, Q2],
        out_subsystems=[LESpace]
    )

    low_energy_H = restriction(H)


Looking at the diagonal of ``low_energy_H``, we can confirm that the entries are the eigenvalues
below the cutoff.

.. jupyter-execute::

    np.diag(low_energy_H.matrix())

With this ``restriction`` mapping, we can also restrict other operators to this subspace, e.g. the
:math:`X` operator acting on the physical ``Q1`` system. Note that when the ``restriction`` map is
applied to ``X(Q1)``, the operator is interpreted as ``I(Q0) @ X(Q1) @ I(Q2)``, i.e. :math:`X`
acting on ``Q1``, and the identity on the remaining subsystems in the input space of ``restriction``
that ``X(Q1)`` does not explicitly act on.

.. jupyter-execute::

    drive_op = restriction(X(Q1))
    drive_op