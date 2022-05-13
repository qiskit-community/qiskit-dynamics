How-to use Dyson and Magnus based solvers
=========================================

In this tutorial we walk through how to use perturbation-theory based solvers.

.. note::

    This is an advanced topic --- utilizing perturbation-theory based solvers
    requires detailed knowledge of the structure of the differential equations
    involved, as well as manual tuning of the solver parameters.
    See the :class:`~qiskit_dynamics.solvers.DysonSolver` and
    :class:`~qiskit_dynamics.solvers.MagnusSolver` documentation for API details. Also, see
    [:footcite:`puzzuoli_sensitivity_2022`] for a detailed explanation of the solvers,
    which varies and builds on the core idea introduced in [:footcite:`shillito_fast_2020`].

    We note further that the circumstances under which perturbative solvers outperform
    traditional solvers, and which parameter sets to use, is nuanced.
    Perturbative solvers executed with JAX are setup to use more parallelization within a
    single solver run than typical solvers, and thus it is circumstance-specific whether
    the trade-off between speed of a single run and resource consumption is advantageous.
    Due to the parallelized nature, the comparison of execution times demonstrated in this
    userguide are highly hardware-dependent.


In this tutorial we use a simple transmon model:

.. math:: H(t) = 2 \pi \nu N + \pi \alpha N(N-I) + s(t) \times 2 \pi r (a + a^\dagger)

where:

-  :math:`N`, :math:`a`, and :math:`a^\dagger` are, respectively, the
   number, annihilation, and creation operators.
-  :math:`\nu` is the qubit frequency and :math:`r` is the drive
   strength.
-  :math:`s(t)` is the drive signal, which we will take to be on
   resonance with envelope :math:`f(t) = A \frac{4t (T - t)}{T^2}`
   for a given amplitude :math:`A` and total time :math:`T`.

We will walk through the following steps:

1. Configure ``qiskit-dynamics`` to work with JAX.
2. Construct the model.
3. How-to construct and simulate using the Dyson-based perturbative solver.
4. Simulate using a traditional ODE solver, comparing speed.
5. How-to construct and simulate using the Magnus-based perturbative solver.

1. Configure to use JAX
-----------------------

These simulations will be done with JAX array backend to enable
compilation. See the user guide entry on using JAX for a detailed
breakdown of using JAX with dynamics.

.. jupyter-execute::

    from qiskit_dynamics.array import Array

    # configure jax to use 64 bit mode
    import jax
    jax.config.update("jax_enable_x64", True)

    # tell JAX we are using CPU
    jax.config.update('jax_platform_name', 'cpu')

    # set default backend
    Array.set_default_backend('jax')

2. Construct the model
----------------------

First, construct the model described in the introduction. Here we use a
higher dimension to observe a difference between the solvers.

.. jupyter-execute::

    import numpy as np

    dim = 10

    v = 5.
    anharm = -0.33
    r = 0.02

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))

    # static part
    static_hamiltonian = 2 * np.pi * v * N + np.pi * anharm * N * (N - np.eye(dim))
    # drive term
    drive_hamiltonian = 2 * np.pi * r * (a + adag)

    # total simulation time
    T = 1. / r

    # envelope function
    envelope_func = lambda t: t * (T - t) / (T**2 / 4)

3. How-to construct and simulate using the Dyson-based perturbative solver
--------------------------------------------------------------------------

Constructing the Dyson-based perturbative solver requires specifying several
configuration parameters, as well as specifying the structure of the
differential equation more explicitly than using the standard
:class:`~qiskit_dynamics.solvers.Solver` object in qiskit-dynamics, which automatically builds
either the Schrodinger or Lindblad equation based on the inputs.

See the API docs for :class:`~qiskit_dynamics.solvers.DysonSolver` for a more detailed
explanation, but some general comments on its instantiation and usage:

- :class:`~qiskit_dynamics.solvers.DysonSolver` requires direct specification of the LMDE to the
  solver. As we are simulating the Schrodinger equation, we need to
  multiply the Hamiltonian terms by ``-1j``.
- :class:`~qiskit_dynamics.solvers.DysonSolver` is a fixed step solver, with the step size
  being fixed at instantiation. This step size must be chosen in conjunction
  with the ``expansion_order``, to ensure that a suitable accuracy is attained.
- Over each fixed time-step:

  - :class:`~qiskit_dynamics.solvers.DysonSolver` solves by computing a truncated perturbative
    expansion.
  - To compute the truncated perturbative expansion, the signal envelopes are
    approximated as a linear combination of Chebyshev polynomials.
  - The order of the Chebyshev approximations, along with central carrier frequencies
    for defining the “envelope” of each ``Signal``, must be provided at instantiation.


.. jupyter-execute::

    %%time

    from qiskit_dynamics import DysonSolver

    dt = 0.1
    dyson_solver = DysonSolver(
        operators=[-1j * drive_hamiltonian],
        rotating_frame=-1j * static_hamiltonian,
        dt=dt,
        carrier_freqs=[v],
        chebyshev_orders=[1],
        expansion_order=7,
        integration_method='jax_odeint',
        atol=1e-12,
        rtol=1e-12
    )


Construct a function that simulates the system for the pulse sequence
with a given amplitude.

.. jupyter-execute::

    from qiskit_dynamics import Signal

    def dyson_sim(amp):
        drive_signal = Signal(lambda t: Array(amp) * envelope_func(t), carrier_freq=v)
        return dyson_solver.solve(
            signals=[drive_signal],
            y0=np.eye(dim, dtype=complex),
            t0=0.,
            n_steps=int(T // dt)
        ).y[-1]

    from jax import jit

    jit_dyson_sim = jit(dyson_sim)

First run includes compile time.

.. jupyter-execute::

    %time yf_dyson = jit_dyson_sim(1.).block_until_ready()


Second run demonstrates the speed of the solver.

.. jupyter-execute::

    %time yf_dyson = jit_dyson_sim(1.).block_until_ready()


4. Comparison to traditional ODE solver
---------------------------------------

We now construct the same simulation using a standard solver to compare
accuracy and simulation speed.

.. jupyter-execute::

    from qiskit_dynamics import Solver

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=[drive_hamiltonian],
        rotating_frame=static_hamiltonian
    )

    # specify tolerance as an argument to run the simulation at different tolerances
    def ode_sim(amp, tol):
        drive_signal = Signal(lambda t: Array(amp) * envelope_func(t), carrier_freq=v)
        solver_copy = solver.copy()
        solver_copy.signals = [drive_signal]
        res = solver_copy.solve(
            t_span=[0., int(T // dt) * dt],
            y0=np.eye(dim, dtype=complex),
            method='jax_odeint',
            atol=tol,
            rtol=tol
        )
        return res.y[-1]

Simulate with low tolerance for comparison to high accuracy solution.

.. jupyter-execute::

    yf_low_tol = ode_sim(1., 1e-13)
    np.linalg.norm(yf_low_tol - yf_dyson)


For speed comparison, compile at a tolerance with similar accuracy.

.. jupyter-execute::

    jit_ode_sim = jit(lambda amp: ode_sim(amp, 1e-8))

    %time yf_ode = jit_ode_sim(1.).block_until_ready()

Measure compiled time.

.. jupyter-execute::

    %time yf_ode = jit_ode_sim(1.).block_until_ready()


Confirm simular accuracy solution.

.. jupyter-execute::

    np.linalg.norm(yf_low_tol - yf_ode)

Here we see that, once compiled, the Dyson-based solver has a
significant speed advantage over the traditional solver, at the expense
of the initial compilation time and the technical aspect of using the
solver.

5. How-to construct and simulate using the Magnus-based perturbation solver
---------------------------------------------------------------------------

Next, build the Magnus-based perturbative solver.
The :class:`~qiskit_dynamics.solvers.MagnusSolver` uses the same scheme as
:class:`~qiskit_dynamics.solvers.DysonSolver`, but uses the Magnus expansion and
matrix exponentiation to simulate over each fixed time step.
Note that the Magnus expansion typically requires going to fewer orders to achieve accuracy,
with the trade-off being that, after construction, the solving step itself is more expensive.

.. jupyter-execute::

    %%time

    from qiskit_dynamics import MagnusSolver

    dt = 0.1
    magnus_solver = MagnusSolver(
        operators=[-1j * drive_hamiltonian],
        rotating_frame=-1j * static_hamiltonian,
        dt=dt,
        carrier_freqs=[v],
        chebyshev_orders=[1],
        expansion_order=3,
        integration_method='jax_odeint',
        atol=1e-12,
        rtol=1e-12
    )


Setup simulation function.

.. jupyter-execute::

    def magnus_sim(amp):
        drive_signal = Signal(lambda t: Array(amp) * envelope_func(t), carrier_freq=v)
        return magnus_solver.solve(
            signals=[drive_signal],
            y0=np.eye(dim, dtype=complex),
            t0=0.,
            n_steps=int(T // dt)
        ).y[-1]

    jit_magnus_sim = jit(magnus_sim)


First run includes compile time.

.. jupyter-execute::

    %time yf_magnus = jit_magnus_sim(1.).block_until_ready()

Second run demonstrates speed of the simulation.

.. jupyter-execute::

    %time yf_magnus = jit_magnus_sim(1.).block_until_ready()


.. jupyter-execute::

    np.linalg.norm(yf_magnus - yf_low_tol)


Observe comparable accuracy at a lower order in the expansion, albeit
with a modest speed up as compared to the Dyson-based solver.

.. footbibliography::
