Using ``PerturbativeSolver`` to accelerate simulation
=====================================================

In this tutorial we walk through how to use perturbation-theory based
solvers to accelerate simulation. Note that this is an advanced topic
involving low level functionalty, and therefore requires detailed
knowledge of the structure of the differential equations involved.

In this tutorial we use a simple transmon model:

.. math:: H(t) = 2 \pi \nu N + \pi \alpha N(N-I) + s(t) \times 2 \pi r (a + a^\dagger)

where:

-  :math:`N`, :math:`a`, and :math:`a^\dagger` are, respectively, the
   number, annihilation, and creation operators.
-  :math:`\nu` is the qubit frequency and :math:`r` is the drive
   strength.
-  :math:`s(t)` is the drive signal, which we will take to be on
   resonance with constant envelope :math:`1`.

We will consider simulation of this system with

.. math:: s(t) = A \frac{4t (T - t)}{T^2}

for a given amplitude :math:`A` and total time :math:`T`.

We will walk through the following steps: 1. Configure
``qiskit-dynamics`` to work with JAX. 2. Construct the model. 3.
Simulate using the Dyson-based perturbative solver. 4. Simulate using a
traditional ODE solver, comparing speed. 5. Simulate using the
Magnus-based perturbative solver.

1. Configure to use JAX
-----------------------

These simulations will be done with JAX array backend to enable
compilation. See the user guide entry on using JAX for a detailed
breakdown of using JAX with dynamics.

.. code:: ipython3

    from qiskit_dynamics.array import Array

    # configure jax to use 64 bit mode
    import jax
    jax.config.update("jax_enable_x64", True)

    # tell JAX we are using CPU
    jax.config.update('jax_platform_name', 'cpu')

    # set default backend
    Array.set_default_backend('jax')

2. Construct the model
======================

First, construct the model described in the introduction. Here we use a
higher dimension to observe a difference between the solvers.

.. code:: ipython3

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

3. Construct Dyson-based perturbative solver
============================================

Constructing the perturbative solver requires specifying several
configuration parameters, as well as specifying the structure of the
differential equation more explicitly than using the standard ``Solver``
object in qiskit-dynamics.

See the API docs for ``PerturbativeSolver`` for a more detailed
explanation, but some general comments on its instantiation and usage: -
``PerturbativeSolver`` requires direct specification of the LMDE to the
solver. As we are simulating the Schrodinger equation, we need to
multiply the Hamiltonian terms by ``-1j``. - ``PerturbativeSolver`` is a
fixed step solver, with the step size being fixed at instantiation. This
step size must be chosen in conjunction with the ``expansion_order``, to
ensure that a suitable accuracy is attained. - Over each fixed
time-step: - ``PerturbativeSolver`` solves by computing a truncated
perturbative expansion. - To compute the truncated perturbative
expansion, the signal envelopes are approximated as a linear combination
of Chebyshev polynomials. - The order of the Chebyshev approximations,
along with central carrier frequencies for defining the “envelope” of
each ``Signal``, must be provided at instantiation.

.. code:: ipython3

    %%time

    from qiskit_dynamics.perturbation import PerturbativeSolver

    dt = 0.1
    dyson_solver = PerturbativeSolver(operators=[-1j * drive_hamiltonian],
                                      rotating_frame=-1j * static_hamiltonian,
                                      dt=dt,
                                      carrier_freqs=[v],
                                      chebyshev_orders=[1],
                                      expansion_method='dyson',
                                      expansion_order=7,
                                      integration_method='jax_odeint',
                                      atol=1e-12,
                                      rtol=1e-12)


.. parsed-literal::

    CPU times: user 2.61 s, sys: 65.5 ms, total: 2.68 s
    Wall time: 2.52 s


Construct a function that simulates the system for the pulse sequence
with a given amplitude.

.. code:: ipython3

    from qiskit_dynamics import Signal

    def dyson_sim(amp):
        drive_signal = Signal(lambda t: Array(amp) * envelope_func(t), carrier_freq=v)
        return dyson_solver.solve(signals=[drive_signal], y0=np.eye(dim, dtype=complex), t0=0., n_steps=int(T // dt))

    from jax import jit

    jit_dyson_sim = jit(dyson_sim)

First run includes compile time.

.. code:: ipython3

    %time yf_dyson = jit_dyson_sim(1.).block_until_ready()


.. parsed-literal::

    CPU times: user 1.72 s, sys: 39.5 ms, total: 1.76 s
    Wall time: 1.73 s


Second run demonstrates the speed of the solver.

.. code:: ipython3

    %time yf_dyson = jit_dyson_sim(1.).block_until_ready()


.. parsed-literal::

    CPU times: user 19.1 ms, sys: 1.89 ms, total: 21 ms
    Wall time: 4.79 ms


4. Comparison to traditional ODE solver
=======================================

We now construct the same simulation using a standard solver to compare
accuracy and simulation speed.

.. code:: ipython3

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

.. code:: ipython3

    yf_low_tol = ode_sim(1., 1e-13)

.. code:: ipython3

    np.linalg.norm(yf_low_tol - yf_dyson)




.. parsed-literal::

    6.531763838979631e-07



For speed comparison, compile at a tolerance with similar accuracy.

.. code:: ipython3

    jit_ode_sim = jit(lambda amp: ode_sim(amp, 1e-8))

    %time yf_ode = jit_ode_sim(1.).block_until_ready()


.. parsed-literal::

    CPU times: user 815 ms, sys: 14.4 ms, total: 829 ms
    Wall time: 821 ms


Measure compiled time.

.. code:: ipython3

    %time yf_ode = jit_ode_sim(1.).block_until_ready()


.. parsed-literal::

    CPU times: user 65.1 ms, sys: 954 µs, total: 66.1 ms
    Wall time: 64.8 ms


Confirm simular accuracy solution.

.. code:: ipython3

    np.linalg.norm(yf_low_tol - yf_ode)




.. parsed-literal::

    8.672110534785226e-07



Here we see that, once compiled, the Dyson-based solver has a
significant speed advantage over the traditional solver, at the expense
of the initial compilation time and the technical aspect of using the
solver.

5. Construct and compare Magnus-based perturbation solver
=========================================================

Next, build the solver with the Magnus expansion. Note that the Magnus
expansion typically requires going to fewer orders to achieve accuracy,
with the trade-off being that, after construction, the solving step
itself is more expensive.

.. code:: ipython3

    %%time

    dt = 0.1
    magnus_solver = PerturbativeSolver(operators=[-1j * drive_hamiltonian],
                                      rotating_frame=-1j * static_hamiltonian,
                                      dt=dt,
                                      carrier_freqs=[v],
                                      chebyshev_orders=[1],
                                      expansion_method='magnus',
                                      expansion_order=3,
                                      integration_method='jax_odeint',
                                      atol=1e-12,
                                      rtol=1e-12)


.. parsed-literal::

    CPU times: user 2.49 s, sys: 38.7 ms, total: 2.53 s
    Wall time: 2.52 s


.. code:: ipython3

    def magnus_sim(amp):
        drive_signal = Signal(lambda t: Array(amp) * envelope_func(t), carrier_freq=v)
        return magnus_solver.solve(signals=[drive_signal], y0=np.eye(dim, dtype=complex), t0=0., n_steps=int(T // dt))

    jit_magnus_sim = jit(magnus_sim)

.. code:: ipython3

    %time yf_magnus = jit_magnus_sim(1.).block_until_ready()


.. parsed-literal::

    CPU times: user 2.69 s, sys: 99.1 ms, total: 2.79 s
    Wall time: 2.78 s


.. code:: ipython3

    %time yf_magnus = jit_magnus_sim(1.).block_until_ready()


.. parsed-literal::

    CPU times: user 26.6 ms, sys: 3.1 ms, total: 29.7 ms
    Wall time: 20.3 ms


.. code:: ipython3

    np.linalg.norm(yf_magnus - yf_low_tol)




.. parsed-literal::

    6.679766721814084e-07



Observe comparable accuracy at a lower order in the expansion, albeit
with a modest speed up as compared to the Dyson-based solver.
