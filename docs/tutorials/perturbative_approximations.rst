Approximating quantum system dynamics with time-dependent perturbation theory
=============================================================================

The Dyson series and Magnus expansions are commonly used
to quantify how quantum system evolution changes with small
variations to model parameters. This information can be utilized in the
study of the dynamics itself, or to quantify things like robustness when
designing control sequences. Numerical computation of these expansion
can aid in understanding things like their domain of convergence, or as
a component of an objective function in optimization.

In this tutorial we walk through the numerical computation of a multi-variable
Magnus expansion, and investigate the quality of the approximation for
various expansion orders.

.. note::

  This is an advanced topic --- utilizing time-dependent perturbation theory
  requires detailed knowledge of the underlying mathematics.

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

Here we walk through:

1. Configure dynamics to use JAX.
2. Construct the operators for the model.
3. Construct a function that solves the model for given frequency :math:`\nu` and
   pulse amplitude :math:`A`. The model parameters are specified to this function as
   *perturbations* from *unperturbed* values, in the form of percentage deviations.
4. Compute multi-variable Magnus expansions truncated to various orders in the perturbation
   parameters.
5. For each truncation order, construct a function that approximates the system evolution for
   specified perturbation parameters. Observe the speed of a full simulation v.s. Magnus-based
   approximation.
6. Investigate accuracies of the evolution approximation to various orders in the Magnus expansion.

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

.. jupyter-execute::

    import numpy as np

    dim = 5

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

3. Define full simulation function in the parameter values
----------------------------------------------------------

.. jupyter-execute::

    from jax import jit
    from qiskit_dynamics import Solver, Signal

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=[2 * np.pi * v * N, drive_hamiltonian],
        rotating_frame=static_hamiltonian
    )

    @jit
    def ode_sim(params):
        d_freq = params[0]
        d_amp = params[1]
        drive_signal = Signal(lambda t: Array(1. + d_amp) * envelope_func(t), carrier_freq=v)
        solver_copy = solver.copy()
        solver_copy.signals = [Array(d_freq), drive_signal]
        res = solver_copy.solve(
            t_span=[0., T],
            y0=np.eye(dim, dtype=complex),
            method='jax_odeint',
            atol=1e-8,
            rtol=1e-8
        )
        return res.y[-1]

Compile and run once.

.. jupyter-execute::

    %time yf_ode = ode_sim(np.array([0., 0.])).block_until_ready()


Run a second time to observe compiled speed.

.. jupyter-execute::

    %time yf_ode = ode_sim(np.array([0., 0.])).block_until_ready()


4. Compute multi-variable Magnus expansion
------------------------------------------

First set up the perturbation parameters.

.. jupyter-execute::

    from qiskit_dynamics import RotatingFrame
    from qiskit_dynamics.models import HamiltonianModel
    from qiskit_dynamics.perturbation import (solve_lmde_perturbation,
                                              ArrayPolynomial)

    signal = Signal(lambda t: Array(1.) * envelope_func(t), carrier_freq=v)

    full_hamiltonian = HamiltonianModel(
        static_operator=static_hamiltonian,
        operators=[drive_hamiltonian],
        signals=[signal],
        rotating_frame=static_hamiltonian
    )
    rotating_frame = RotatingFrame(static_hamiltonian)

    perturb0 = lambda t: -1j * 2 * np.pi * v * rotating_frame.operator_into_frame(t, N)
    perturb1 = lambda t: -1j * signal(t) * rotating_frame.operator_into_frame(t, drive_hamiltonian)

Compute the solution at zero perturbation, and the perturbative terms.

.. jupyter-execute::

    %%time

    results = []

    max_order = 5

    for k in range(1, max_order + 1):
        result = solve_lmde_perturbation(
            perturbations=[perturb0, perturb1],
            t_span=[0, T],
            expansion_method='magnus',
            expansion_order=k,
            generator=full_hamiltonian,
            integration_method='jax_odeint',
            atol=1e-8,
            rtol=1e-8,
        )
        results.append(result)


Set up an ``ArrayPolynomial`` object from the results to evaluate the
Magnus expansion.

.. jupyter-execute::

    magnus_expansions = []
    for result in results:
        magnus_terms = result.perturbation_results.expansion_terms[:, -1]
        labels = result.perturbation_results.expansion_labels

        magnus_expansion = ArrayPolynomial(
            array_coefficients=magnus_terms,
            monomial_labels=labels
        )
        magnus_expansions.append(magnus_expansion)

Construct the perturbation-based simulation function.

.. jupyter-execute::

    from jax.scipy.linalg import expm as jexpm


    # necessary when constructing functions in loops to avoid
    def get_magnus_sim(k):
        @jit
        def magnus_sim(c):
            return results[k].y[-1] @ jexpm(magnus_expansions[k](c).data)

        return magnus_sim


    magnus_sims = []
    for k in range(max_order):
        magnus_sims.append(get_magnus_sim(k))

5. Compare speed of ODE-based simulation and Magnus simulation
--------------------------------------------------------------

Compile and run once.

.. jupyter-execute::

    %time yf_magnus = magnus_sims[-1](np.array([0., 0.])).block_until_ready()

Run again to observe compiled speed.

.. jupyter-execute::

    %time yf_magnus = magnus_sims[-1](np.array([0., 0.])).block_until_ready()


Verify agreement of the no-perturbation solution.

.. jupyter-execute::

    def fidelity(U, V):
        return np.abs((U.conj() * V).sum() / dim) ** 2

    fidelity(yf_magnus, yf_ode)

6. Empirically observe accuracy over parameter ranges at various orders
-----------------------------------------------------------------------

First, observe the range of valid approximation

.. jupyter-execute::

    import matplotlib.pyplot as plt

    direction = np.array([1., 0.])

    fidelities = [[] for _ in range(max_order)]

    perturb_vals = np.linspace(-0.005, 0.005, 50)
    for d in perturb_vals:
        c = d * direction

        y_ode = ode_sim(c)

        for fidelity_list, magnus_sim in zip(fidelities, magnus_sims):
            y_magnus = magnus_sim(c)
            fidelity_list.append(fidelity(y_ode, y_magnus))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()

Zoom in to observe parameter range over which the solutions agree with a
fidelity above ``0.999``.

.. jupyter-execute::

    plt.ylim((0.999, 1.))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()

Next do perturbations to the drive strength.

.. jupyter-execute::

    fidelities = [[] for _ in range(max_order)]

    direction = np.array([0., 1.])

    perturb_vals = np.linspace(-3.0, 3.0, 50)
    for d in perturb_vals:
        c = d * direction

        y_ode = ode_sim(c)

        for fidelity_list, magnus_sim in zip(fidelities, magnus_sims):
            y_magnus = magnus_sim(c)
            fidelity_list.append(fidelity(y_ode, y_magnus))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()

Zoom in again to observe parameter range above which a fidelity of ``0.999`` is achieved.

.. jupyter-execute::

    plt.ylim((0.999, 1.))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()

Finally, observe on the diagonal.

.. jupyter-execute::

    fidelities = [[] for _ in range(max_order)]

    direction = np.array([0.002, 2.])

    perturb_vals = np.linspace(-1.0, 1.0, 50)
    for d in perturb_vals:
        c = d * direction

        y_ode = ode_sim(c)

        for fidelity_list, magnus_sim in zip(fidelities, magnus_sims):
            y_magnus = magnus_sim(c)
            fidelity_list.append(fidelity(y_ode, y_magnus))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()

Zoom in to observe range for fidelity above ``0.999``.

.. jupyter-execute::

    plt.ylim((0.999, 1.))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()
