Constructing approximations to quantum system dynamics using time-dependent perturbation theory
===============================================================================================

The Dyson series and Magnus expansions are commonly used in the study of
the dynamics of quantum systems, both as an analytical tool and as a
numerical tool to quantify how the evolution changes with small
variations to model parameters. This information can be utilized in the
study of the dynamics itself, or to quantify things like robustness when
designing control sequences. Numerical computation of these expansion
can aid in understanding things like their domain of convergence, or as
a component of an objective function in optimization.

In this tutorial we walk through using time-dependent perturbation
theory to construct approximations of quantum system dynamics.

Note: - This is an advanced topic… etc

Here we walk through:

1. Configure dynamics to use JAX.
2. Define a model.
3. Construct a function that computes the solution of the model, whose
   inputs are perturbations to the parameter values.
4. Compute Magnus expansion in the perturbation parameters.
5. Compare speed of a full simulation v.s. Magnus-based approximation.
6. Investigate accuracy of approximation.

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
----------------------

.. code:: ipython3

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

.. code:: ipython3

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

.. code:: ipython3

    %time yf_ode = ode_sim(np.array([0., 0.])).block_until_ready()


.. parsed-literal::

    CPU times: user 679 ms, sys: 15.8 ms, total: 694 ms
    Wall time: 684 ms


Run a second time to observe compiled speed.

.. code:: ipython3

    %time yf_ode = ode_sim(np.array([0., 0.])).block_until_ready()


.. parsed-literal::

    CPU times: user 26.6 ms, sys: 1.54 ms, total: 28.1 ms
    Wall time: 26.1 ms


4. Compute multi-variable Magnus expansion
------------------------------------------

First set up the perturbation parameters.

.. code:: ipython3

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

.. code:: ipython3

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


.. parsed-literal::

    CPU times: user 15.4 s, sys: 138 ms, total: 15.5 s
    Wall time: 15.4 s


Set up an ``ArrayPolynomial`` object from the results to evaluate the
Magnus expansion.

.. code:: ipython3

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

.. code:: ipython3

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

.. code:: ipython3

    %time yf_magnus = magnus_sims[-1](np.array([0., 0.])).block_until_ready()


.. parsed-literal::

    CPU times: user 557 ms, sys: 11.9 ms, total: 569 ms
    Wall time: 553 ms


Run again to observe compiled speed.

.. code:: ipython3

    %time yf_magnus = magnus_sims[-1](np.array([0., 0.])).block_until_ready()


.. parsed-literal::

    CPU times: user 63 µs, sys: 13 µs, total: 76 µs
    Wall time: 71.8 µs


Verify agreement of the no-perturbation solution.

.. code:: ipython3

    def fidelity(U, V):
        return np.abs((U.conj() * V).sum() / dim) ** 2

.. code:: ipython3

    fidelity(yf_magnus, yf_ode)




.. parsed-literal::

    1.0000002789977782



6. Empirically observe accuracy over parameter ranges at various orders
-----------------------------------------------------------------------

First, observe the range of valid approximation

.. code:: ipython3

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




.. parsed-literal::

    <matplotlib.legend.Legend at 0x144d669b0>




.. image:: constructing_solution_approximations_files/constructing_solution_approximations_27_1.png


.. code:: ipython3

    plt.ylim((0.999, 1.))
    
    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')
    
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x14543ea70>




.. image:: constructing_solution_approximations_files/constructing_solution_approximations_28_1.png


.. code:: ipython3

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




.. parsed-literal::

    <matplotlib.legend.Legend at 0x141785630>




.. image:: constructing_solution_approximations_files/constructing_solution_approximations_29_1.png


.. code:: ipython3

    plt.ylim((0.999, 1.))
    
    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')
    
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x145489840>




.. image:: constructing_solution_approximations_files/constructing_solution_approximations_30_1.png


.. code:: ipython3

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




.. parsed-literal::

    <matplotlib.legend.Legend at 0x145465990>




.. image:: constructing_solution_approximations_files/constructing_solution_approximations_31_1.png


.. code:: ipython3

    plt.ylim((0.999, 1.))
    
    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')
    
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x144e63820>




.. image:: constructing_solution_approximations_files/constructing_solution_approximations_32_1.png


