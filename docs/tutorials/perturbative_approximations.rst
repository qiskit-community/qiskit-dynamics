Approximating quantum system dynamics with time-dependent perturbation theory
=============================================================================

The Dyson series and Magnus expansions are commonly used
perturbative expansions that quantify how quantum system evolution changes with small
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
  requires an understanding of the mathematics of the expansions, as well as
  their limitations. For the Dyson series see [:footcite:`dyson_radiation_1949`],
  for the Magnus expansion see
  [:footcite:`magnus_exponential_1954`, :footcite:`blanes_magnus_2009`],
  and for the notation and conventions used in this package, see the perturbation module
  API documentation.

In this tutorial we use a simple transmon model:

.. math::

  H(t) = 2 \pi \nu (1 + p_{\nu}) N + \pi \alpha N(N-I)
    + s(t) \times 2 \pi r (1 + p_{r}) (a + a^\dagger)

where:

- :math:`N`, :math:`a`, and :math:`a^\dagger` are, respectively, the
  number, annihilation, and creation operators.
- :math:`\nu` is the qubit frequency and :math:`r` is the drive
  strength.
- :math:`s(t)` is the drive signal, which we will take to be on
  resonance with envelope :math:`f(t) = A \frac{4t (T - t)}{T^2}`
  for a given amplitude :math:`A` and total time :math:`T`.
- :math:`p_{\nu}` and :math:`p_{r}` are, respectively, *perturbations* of the qubit
  frequency and drive from their *unperturbed* values :math:`\nu` and :math:`r`, specified
  as percentage deviations. These parameters model *uncertainties* in the
  qubit frequency and drive strength, which may be due to characterization uncertainty or parameter
  drift.

Here we walk through how to use time-dependent perturbation theory to generate approximations
to the evolution for small values of :math:`p_{\nu}` and :math:`p_r`, and investigate
performance, both in terms of the time it takes to generate the approximation, as well as
the quality of the approximation. We walk through the following steps:

1. Configure dynamics to use JAX.
2. Construct the operators for the model.
3. Construct a function that solves the model for given frequency :math:`\nu` and
   pulse amplitude :math:`A`. The model parameters are specified to this function as
   *perturbations* from *unperturbed* values, in the form of percentage deviations.
4. Compute multi-variable Magnus expansions truncated to various orders in the perturbation
   parameters.
5. For each truncation order, construct a function that approximates the system evolution for
   specified perturbation parameters. For a given value of the perturbed parameters, observe the
   speed of performing a new full simulation v.s. generating a perturbative approximation.
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

Build the operators used in the model. For this model we use a truncation dimension of :math:`5`,
and parameters :math:`\nu = 5`, :math:`\alpha = -0.33`, and :math:`r = 0.02`.

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

Next, define a function that solves the system using traditional ODE methods over
the length of the pulse :math:`T`, using the above parameters, with inputs being
the peturbations :math:`p_{\nu}` and :math:`p_r`. We will solve in the rotating frame
of the unperturbed static Hamiltonian.

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
        p_v = params[0]
        p_r = params[1]

        drive_signal = Signal(lambda t: Array(1. + p_r) * envelope_func(t), carrier_freq=v)
        solver_copy = solver.copy()
        solver_copy.signals = [Array(p_v), drive_signal]
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

Next, compute the multi-variable Magnus expansion for this problem. As we are solving in
the rotating frame, we have that the unperturbated Hamiltonian is:

.. math::

    H_0(t) = e^{i H_s t}(s(t) \times 2 \pi r (a + a^\dagger))e^{-i H_s t}

where :math:`H_s` is the unperturbed static Hamiltonian, and the two perturbation are:

.. math::

    H_{\nu}(t) = e^{i H_s t}(2 \pi \nu N)e^{-i H_s t}

and

.. math::

    H_{r}(t) = e^{i H_s t}(s(t) \times 2 \pi r (a + a^\dagger))e^{-i H_s t},


so that :math:`H(t) = H_0(t) + p_{\nu} H_{\nu}(t) + p_r H_r(t)`.

First, we construct functions for computing
:math:`-iH_0(t)`, :math:`-iH_{\nu}(t)`, and :math:`-iH_r(t)`.


.. jupyter-execute::

    from qiskit_dynamics import RotatingFrame
    from qiskit_dynamics.perturbation import (solve_lmde_perturbation,
                                              ArrayPolynomial)

    signal = Signal(lambda t: Array(1.) * envelope_func(t), carrier_freq=v)
    rotating_frame = RotatingFrame(static_hamiltonian)

    unperturbed_hamiltonian = lambda t: (-1j * signal(t)
        * rotating_frame.operator_into_frame(t, drive_hamiltonian)
    )

    perturb0 = lambda t: -1j * 2 * np.pi * v * rotating_frame.operator_into_frame(t, N)
    perturb1 = lambda t: -1j * signal(t) * rotating_frame.operator_into_frame(t, drive_hamiltonian)


Next, call ``solve_lmde_perturbation`` to compute both the solution to the unperturbed system,
as well as the Magnus expansion for various orders. Here, we consider orders 1 through 5.

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
            generator=unperturbed_hamiltonian,
            integration_method='jax_odeint',
            atol=1e-8,
            rtol=1e-8,
        )
        results.append(result)


For each order of the Magnus expansion, set up an ``ArrayPolynomial`` object from the results
to be used for evaluation of the expansion.

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

For each order in the Magnus expansion, construct a function that, given an array of
perturbation parameters ``c = [p_v, p_r]``, computes the approximate evolution using
the Magnus expansion.

.. jupyter-execute::

    from jax.scipy.linalg import expm as jexpm

    # necessary when constructing functions in loops to avoid referencing issues
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

Run the highest order Magnus expansion-based approximation function (which is the most expensive)
once to compile:

.. jupyter-execute::

    %time yf_magnus = magnus_sims[-1](np.array([0., 0.])).block_until_ready()

Run again to observe compiled speed.

.. jupyter-execute::

    %time yf_magnus = magnus_sims[-1](np.array([0., 0.])).block_until_ready()

Observe that, at the cost of the initial construction, generating an approximation for
some perturbed values is significantly faster than re-solving the system using a
traditional ODE method.

Using the fidelity metric for comparing unitaries:

.. math::

    f(U, V) = \frac{|Tr(U^{\dagger} V)|^2}{d^2},

with :math:`d` is the dimension of the system, validate the perturbative at no perturbation
is equivalent to the standard ODE simulation:

.. jupyter-execute::

    def fidelity(U, V):
        return np.abs((U.conj() * V).sum() / dim) ** 2

    fidelity(yf_magnus, yf_ode)

6. Empirically observe accuracy over parameter ranges at various orders
-----------------------------------------------------------------------

Next, we will investigate the accuracy of the approximation at various orders of the expansion
in various directions in parameter space, by computing fidelity between the approximation
and a traditional ODE solution at each point in parameter space.

First, consider variations in qubit frequency, given by non-zero :math:`p_v`,
holding :math:`p_r=0`. We plot the fidelity of the approximations at various
Magnus orders, for :math:`p_v \in [-0.005, 0.005]`.

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

We plot the same curves with y-axis restricted to :math:`[0.999, 1.]` to observe the
regions with high quality approximations.

.. jupyter-execute::

    plt.ylim((0.999, 1.))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()

Observe widening ranges of high quality approximation as the Magnus order increases.

Next, we perform the same computation, however now holding :math:`p_{\nu}=0`, and varying
:math:`p_r \in [-3, 3]`.

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

Again, plot the same curves with restricted y-axis, now zooming in
to the range :math:`[0.99999, 1.]`.

.. jupyter-execute::

    plt.ylim((0.99999, 1.))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()

Finally, we generate the same plots but now varying *both* :math:`p_{\nu}` and :math:`p_{r}`.
We plot again over a single parameter, :math:`p`, with


.. math::

    (p_{\nu}, p_r) = p \times (0.002, 2.).


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

Zoom in to observe fidelity range :math:`[0.999, 1]`.

.. jupyter-execute::

    plt.ylim((0.999, 1.))

    for order, fidelity_list in enumerate(fidelities):
        plt.plot(perturb_vals, fidelity_list, label=f'order={order + 1}')

    plt.legend()


.. footbibliography::
