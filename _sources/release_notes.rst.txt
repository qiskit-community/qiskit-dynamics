=============
Release Notes
=============

.. _Release Notes_0.4.1-12:

0.4.1-12
========

.. _Release Notes_0.4.1-12_Prelude:

Prelude
-------

.. releasenotes/notes/patch-0.4.2-6a7c7bf380e54187.yaml @ None

Qiskit Dynamics 0.4.2 is an incremental release with minor bug fixes and additional warnings to help guide users through issues.


.. _Release Notes_0.4.1-12_Upgrade Notes:

Upgrade Notes
-------------

.. releasenotes/notes/subsystem_labels-removal-9fcc71c310eff220.yaml @ b'cf256192ce1c0ef7c2f4c696d9be64234b48b68f'

- The ``subsystem_labels`` option has been removed from the :class:`.DynamicsBackend`. This
  removal impacts some technical aspects of the backend returned by
  :meth:`.DynamicsBackend.from_backend` when the ``subsystem_list`` argument is used. Using the
  ``subsystem_list`` argument with :meth:`.DynamicsBackend.from_backend` restricts the internally
  constructed model to the qubits in ``subsystem_list``. When doing so previously, the option
  ``subsystem_labels`` would be set to ``subsystem_labels``, and ``subsystem_dims`` would record
  only the dimensions for the systems in ``subsystem_labels``. To account for the fact that
  ``subsystem_labels`` no longer exists, :meth:`.DynamicsBackend.from_backend` now constructs
  ``subsystem_dims`` to list a dimension for all of the qubits in the original backend, however
  now the dimensions of the removed systems are given as 1 (i.e. they are treated as trivial
  quantum systems with a single state). This change is made only for technical bookkeping
  purposes, and has no impact on the core simulation behaviour.


.. _Release Notes_0.4.1-12_Bug Fixes:

Bug Fixes
---------

.. releasenotes/notes/carrier-freq-0-19ad4362c874944f.yaml @ None

- In the case that ``envelope`` is a constant, the :meth:`.Signal.__init__` method has been
  updated to not attempt to evaluate ``carrier_freq == 0.0`` if ``carrier_freq`` is a JAX tracer.
  In this case, it is not possible to determine if the :class:`.Signal` instance is constant. This
  resolves an error that was being raised during JAX tracing if ``carrier_freq`` is abstract.

.. releasenotes/notes/classical-registers-9bb117398a4d21d5.yaml @ None

- Fixes bug in :meth:`.DynamicsBackend.run` that caused miscounting of the number of classical
  registers in a :class:`~qiskit.circuit.QuantumCircuit` (issue #251).

.. releasenotes/notes/normalize-probabilities-d729245bb3fe5f10.yaml @ b'6ede10a2bc8c61e8640db9085d4d1d9423341550'

- ``DynamicsBackend.options.normalize_states`` now also controls whether or not the probability
  distribution over outcomes is normalized before sampling outcomes. 


.. _Release Notes_0.4.1-12_Other Notes:

Other Notes
-----------

.. releasenotes/notes/patch-0.4.2-6a7c7bf380e54187.yaml @ None

- For users that have JAX installed, a warning has been added upon import of Qiskit Dynamics to
  notify the user of issues with certain versions: JAX versions newer than ``0.4.6`` break the
  ``perturbation`` module, and to use ``perturbation`` module with versions ``0.4.4``, ``0.4.5``,
  or ``0.4.6``, it is necessary to set ``os.environ['JAX_JIT_PJIT_API_MERGE'] = '0'`` before
  importing JAX or Dynamics.

.. releasenotes/notes/patch-0.4.2-6a7c7bf380e54187.yaml @ None

- A warning has been added to :class:`.InstructionToSignals` class when converting pulse schedules
  to signals to notify the user if the usage of ``SetFrequency`` or ``ShiftFrequency`` commands 
  result in a digital carrier frequency larger than the Nyquist frequency of the envelope sample
  size ``dt``.


.. _Release Notes_0.4.1:

0.4.1
=====

.. _Release Notes_0.4.1_Prelude:

Prelude
-------

.. releasenotes/notes/0.4/patch-0.4.1-d339aa8669341341.yaml @ b'd6e280259d120d31723e0220a91cbd7dd8099298'

Qiskit Dynamics 0.4.1 is an incremental release with minor bug fixes, documentation updates, and usability features.

.. _Release Notes_0.4.1_New Features:

New Features
------------

.. releasenotes/notes/measurement_property_bug_fix-12461088823a943c.yaml @ b'807edf92d7f5d6f34715fff9d21614d77cd096d3'

- The :meth:`DynamicsBackend.from_backend` method has been updated to automatically populate the 
  ``control_channel_map`` option based on the supplied backend if the user does not supply one.


.. _Release Notes_0.4.1_Known Issues:

Known Issues
------------

.. releasenotes/notes/0.4/diffrax-bound-0bd80c01b7f4b48f.yaml @ b'd6e280259d120d31723e0220a91cbd7dd8099298'

- Due to a bug in JAX, Dynamics can only be used with jax<=0.4.6. As they depend on newer versions
  of JAX, Dynamics is also now only compatible with diffrax<=0.3.1 and equinox<=0.10.3.


.. _Release Notes_0.4.1_Bug Fixes:

Bug Fixes
---------

.. releasenotes/notes/0.4/multiset-order-bug-fix-1f1603ee1e230cba.yaml @ b'd6e280259d120d31723e0220a91cbd7dd8099298'

- Fixes a bug in the perturbation module with internal sorting of ``Multiset`` instances, which
  caused incorrect computation of perturbation theory terms when ``>10`` perturbations are
  present.

.. releasenotes/notes/measurement_property_bug_fix-12461088823a943c.yaml @ b'807edf92d7f5d6f34715fff9d21614d77cd096d3'

- A bug in :meth:`DynamicsBackend.__init__` causing existing measurement instructions for a
  user-supplied :class:`Target` to be overwritten has been fixed.


.. _Release Notes_0.4.1_Other Notes:

Other Notes
-----------

.. releasenotes/notes/0.4/move-repo-c0b48ba3b0ced8db.yaml @ b'd6e280259d120d31723e0220a91cbd7dd8099298'

- The repository has been moved from 
  [github.com/Qiskit/qiskit-dynamics](https://github.com/Qiskit/qiskit-dynamics) to 
  [github.com/Qiskit-Extensions/qiskit-dynamics](https://github.com/Qiskit-Extensions/qiskit-dynamics), 
  and the documentation has been moved from
  [qiskit.org/documentation/dynamics](https://qiskit.org/documentation/dynamics) to 
  [qiskit.org/ecosystem/dynamics](https://qiskit.org/ecosystem/dynamics/).


