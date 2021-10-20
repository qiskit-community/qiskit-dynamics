#############################
Qiskit Dynamics documentation
#############################

Qiskit Dynamics is an open-source project for building, transforming, and solving
models of quantum systems in Qiskit.

The goal of Qiskit Dynamics is to provide access to different numerical
methods, and to automate common processes typically performed by hand,
e.g. entering rotating frames, or doing the rotating wave approximation.

Qiskit Dynamics can be configured to use either
`NumPy <https://github.com/numpy/numpy>`_ or `JAX <https://github.com/google/jax>`_
as the backend for array operations. `NumPy <https://github.com/numpy/numpy>`_ is the default,
and `JAX <https://github.com/google/jax>`_ is an optional dependency, which enables
just-in-time compilation, automatic differentiation, and GPU execution of Qiskit Dynamics code.

.. warning::

   This package is still in the early stages of development and it is very likely
   that there will be breaking API changes in future releases.
   If you encounter any bugs please open an issue on
   `Github <https://github.com/Qiskit/qiskit-dynamics/issues>`_


.. toctree::
  :maxdepth: 1

  Tutorials <tutorials/index>
  User Guide <userguide/index>
  API References <apidocs/index>
  Discussions <discussions/index>
  Release Notes <release_notes>

.. Hiding - Indices and tables
   :ref:`genindex`
   :ref:`modindex`
   :ref:`search`
