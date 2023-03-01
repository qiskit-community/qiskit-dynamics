Gradient optimization of a pulse sequence
=========================================

Here, we walk through an example of optimizing a single-qubit gate using
``qiskit_dynamics``. This tutorial requires JAX - see the user guide
on :ref:`How-to use JAX with qiskit-dynamics <how-to use jax>` to
work with JAX.

We will optimize an :math:`X`-gate on a model of a qubit system using
the following steps. Here we introduce optimization of two patterns.
A. construct a signal from samples for a piecewise-constant envelope
B. construct a signal from pulse sequences using `qiskit-pulse`

1. Configure ``qiskit-dynamics`` to work with the JAX backend.

2-A. Setup a ``Solver`` instance with the model of the system.
3-A. Define a pulse sequence parameterization to optimize over.
4-A. Define a gate fidelity function.
5-A. Define an objective function for optimization.
6-A. Use JAX to differentiate the objective, then do the gradient optimization.

2-B. Define a transmon model with Hamiltonian and set up the solver.
3-B. Define DRAG pulse.
4-B. Define a gate fidelity function.
5-B. Define an objective function for optimization.
6-B. Perform JAX transformations and optimize.




1. Configure to use JAX
-----------------------

First, set JAX to operate in 64-bit mode, and set JAX as the default
backend using ``Array`` for performing array operations.
This is necessary to enable automatic differentiation of the Qiskit Dynamics code
in this tutorial. See the user guide entry on using JAX
for a more detailed explanation of why this step is necessary.

.. jupyter-execute::

    import jax
    jax.config.update("jax_enable_x64", True)

    # tell JAX we are using CPU
    jax.config.update('jax_platform_name', 'cpu')

    from qiskit_dynamics.array import Array
    Array.set_default_backend('jax')


2-A. Setup the solver
---------------------

Here we will setup a ``Solver`` with a simple model of a qubit. The
Hamiltonian is:

.. math:: H(t) = 2 \pi \nu \frac{Z}{2} + 2 \pi r s(t) \frac{X}{2}

In the above:

- :math:`\nu` is the qubit frequency,
- :math:`r` is the drive strength,
- :math:`s(t)` is the drive signal which we will optimize, and
- :math:`X` and :math:`Z` are the Pauli X and Z operators.

We will setup the problem to be in the rotating frame of the drift term.

Also note: The ``Solver`` is initialized *without* signals, as we will
update these and optimize over this later.

.. jupyter-execute::

    import numpy as np
    from qiskit.quantum_info import Operator
    from qiskit_dynamics import Solver

    v = 5.
    r = 0.02

    static_hamiltonian = 2 * np.pi * v * Operator.from_label('Z') / 2
    drive_term = 2 * np.pi * r * Operator.from_label('X') / 2

    ham_solver = Solver(
        hamiltonian_operators=[drive_term],
        static_hamiltonian=static_hamiltonian,
        rotating_frame=static_hamiltonian,
    )


3-A. Define a pulse sequence parameterization to optimize over
--------------------------------------------------------------

We will optimize over signals that are:

-  On resonance with piecewise constant envelopes,
-  Envelopes bounded between :math:`[-1, 1]`,
-  Envelopes are smooth, in the sense that the change between adjacent
   samples is small, and
-  Envelope starts and ends at :math:`0`.

In setting up our parameterization, we need t keep in mind that we will
use the BFGS optimization routine, and hence:

-  Optimization parameters must be *unconstrained*.
-  Parameterization must be JAX-differentiable.

We implement a parameterization as follows:

-  Input: Array ``x`` of real values.
-  “Normalize” ``x`` by applying a JAX-differentiable function from
   :math:`\mathbb{R} \rightarrow [-1, 1]`.
-  Pad the normalized ``x`` with a :math:`0.` to start.
-  “Smoothen” the above via convolution.
-  Construct the signal using the above as the samples for a
   piecewise-constant envelope, with carrier frequency on resonance.

We remark that there are many other parameterizations that may achieve
the same ends, and may have more efficient strategies for achieving a
value of :math:`0` at the beginning and end of the pulse. This is only
meant to demonstrate the need for such an approach, and one simple
example of one.

.. jupyter-execute::

    from qiskit_dynamics import DiscreteSignal
    from qiskit_dynamics.array import Array
    from qiskit_dynamics.signals import Convolution

    # define convolution filter
    def gaus(t):
        sigma = 15
        _dt = 0.1
        return 2.*_dt/np.sqrt(2.*np.pi*sigma**2)*np.exp(-t**2/(2*sigma**2))

    convolution = Convolution(gaus)

    # define function mapping parameters to signals
    def signal_mapping(params):
        samples = Array(params)

        # map samples into [-1, 1]
        bounded_samples = np.arctan(samples) / (np.pi / 2)

        # pad with 0 at beginning
        padded_samples = np.append(Array([0], dtype=complex), bounded_samples)

        # apply filter
        output_signal = convolution(DiscreteSignal(dt=1., samples=padded_samples))

        # set carrier frequency to v
        output_signal.carrier_freq = v

        return output_signal

Observe, for example, the signal generated when all parameters are
:math:`10^8`:

.. jupyter-execute::

    signal = signal_mapping(np.ones(80) * 1e8)
    signal.draw(t0=0., tf=signal.duration * signal.dt, n=1000, function='envelope')


4-A. Define gate fidelity
-------------------------

We will optimize an :math:`X` gate, and define the fidelity of the unitary :math:`U`
implemented by the pulse via the standard fidelity measure:

.. math:: f(U) = \frac{|\text{Tr}(XU)|^2}{4}

.. jupyter-execute::

    X_op = Array(Operator.from_label('X'))

    def fidelity(U):
        U = Array(U)

        return np.abs(np.sum(X_op * U))**2 / 4.

5-A. Define the objective function
----------------------------------

The function we want to optimize consists of:

-  Taking a list of input samples and applying the signal mapping.
-  Simulating the Schrodinger equation over the length of the pulse
   sequence.
-  Computing and return the infidelity (we minimize :math:`1-f(U)`).

.. jupyter-execute::

    def objective(params):

        # apply signal mapping and set signals
        signal = signal_mapping(params)
        
        # Simulate
        results = ham_solver.solve(
            y0=np.eye(2, dtype=complex),
            t_span=[0, signal.duration * signal.dt],
            signals=[signal],
            method='jax_odeint',
            atol=1e-8,
            rtol=1e-8
        )
        U = results.y[-1]

        # compute and return infidelity
        fid = fidelity(U)
        return 1. - fid.data

6-A. Perform JAX transformations and optimize
---------------------------------------------

Finally, we gradient optimize the objective:

-  Use ``jax.value_and_grad`` to transform the objective into a function
   that computes both the objective and the gradient.
-  Use ``jax.jit`` to just-in-time compile the function into optimized
   `XLA <https://www.tensorflow.org/xla>`__ code. For the initial cost of
   performing the compilation, this speeds up each call of the function,
   speeding up the optimization.
-  Call ``scipy.optimize.minimize`` with the above, with
   ``method='BFGS'`` and ``jac=True`` to indicate that the passed
   objective also computes the gradient.

.. jupyter-execute::

    from jax import jit, value_and_grad
    from scipy.optimize import minimize

    jit_grad_obj = jit(value_and_grad(objective))

    initial_guess = np.random.rand(80) - 0.5

    opt_results = minimize(fun=jit_grad_obj, x0=initial_guess, jac=True, method='BFGS')
    print(opt_results.message)
    print('Number of function evaluations: ' + str(opt_results.nfev))
    print('Function value: ' + str(opt_results.fun))


The gate is optimized to an :math:`X` gate, with deviation within the
numerical accuracy of the solver.

We can draw the optimized signal, which is retrieved by applying the
``signal_mapping`` to the optimized parameters.

.. jupyter-execute::

    opt_signal = signal_mapping(opt_results.x)

    opt_signal.draw(
        t0=0,
        tf=opt_signal.duration * opt_signal.dt,
        n=1000,
        function='envelope',
        title='Optimized envelope'
    )


Summing the signal samples yields approximately :math:`\pm 50`, which is
equivalent to what one would expect based on a rotating wave
approximation analysis.

.. jupyter-execute::

    opt_signal.samples.sum()


2-B. Define a transmon model with Hamiltonian and set up the solver
-------------------------------------------------------------------

A transmon model with Hamiltonian we will simulate is here.

.. math:: H(t) = 2 \pi \nu N + \pi \alpha N(N-I) + s(t) \times 2 \pi r (a + a^\dagger)


- :math:`N`, :math:`a`, and :math:`a^\dagger` are, respectively, the number, annihilation, and creation operators.
- :math:`\nu` is the qubit frequency,
- :math:`r` is the drive strength,
- :math:`s(t)` is the drive signal which we will optimize.

The following used values such as ``v``, ``anharm``, ``r``, ``dt``, and ``w`` is determined as typical ones.
We note that `dim` is set to ``3`` since DRAG pulse considers the leakage to 2-state.


.. jupyter-execute::

    import numpy as np
    from qiskit.quantum_info import Operator
    from qiskit_dynamics import Solver
    from qiskit_dynamics.pulse import InstructionToSignals

    dim = 3
    v = 5.
    anharm = -0.33
    r = 0.1
    dt = 0.222
    w = 5.

    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))


    static_hamiltonian = 2 * np.pi * v * N + np.pi * anharm * N * (N - np.eye(dim))
    drive_hamiltonian = 2 * np.pi * r * (a + adag)


    ham_solver = Solver(
        hamiltonian_operators=[drive_hamiltonian],
        static_hamiltonian=static_hamiltonian,
        rotating_frame=static_hamiltonian,
    )


3-B. Define DRAG pulse
----------------------

Although qiskit provides a ``DRAG`` class that generates a DRAG pulse, which is a subclass of ``ScalableSymbolicPulse``, 
this class is currently not JAX-supported.

We construct the DRAG pulse directly from ``ScalableSymbolicPulse``.

.. jupyter-execute::

    from qiskit import pulse
    import sympy as sym

    def lifted_gaussian(
        t: sym.Symbol,
        center,
        t_zero,
        sigma,
    ) -> sym.Expr:
        t_shifted = (t - center).expand()
        t_offset = (t_zero - center).expand()

        gauss = sym.exp(-((t_shifted / sigma) ** 2) / 2)
        offset = sym.exp(-((t_offset / sigma) ** 2) / 2)

        return (gauss - offset) / (1 - offset)

    def drag(params):
        amp, beta = params
        _t, _duration, _amp, _sigma, _beta, _angle = sym.symbols(
            "t, duration, amp, sigma, beta, angle"
        )
        _center = _duration / 2
        _gauss = lifted_gaussian(_t, _center, _duration + 1, _sigma)
        _deriv = -(_t - _center) / (_sigma**2) * _gauss

        envelope_expr = _amp * sym.exp(sym.I * _angle) * (_gauss + sym.I * _beta * _deriv)
        
        return pulse.ScalableSymbolicPulse(
                pulse_type="Drag",
                duration=160,
                amp=amp,
                angle=0,
                parameters={"sigma": 40, "beta": beta},
                envelope=envelope_expr,
                constraints=_sigma > 0,
                valid_amp_conditions=sym.And(sym.Abs(_amp) <= 1.0, sym.Abs(_beta) < _sigma),
            )


4-B. Define a gate fidelity function.
-------------------------------------

We want to optimize :math:`X` gate, and define the fidelity of the unitary :math:`U`
implemented by the pulse. :

.. math:: f(U) = \frac{|\text{Tr}(XU)|_2|}{2}

.. jupyter-execute::

    X_op = Array(Operator(
        [[0., 1., 0.],
         [1., 0., 0.], 
         [0., 0., 1.]]))


    def fidelity(U):
        U = Array(U)
        V = Array(Operator(
        [[1., 0., 0.],
         [0., 1., 0.], 
         [0., 0., 0.]]))

        return np.abs(np.trace(X_op@(V@U@V))) / 2


5-B. Define an objective function for optimization
--------------------------------------------------

The role of the function we want to optimize is:

- Setting params we want to optimze. In this tutorial, we optimize amplifier and beta.
- Constructing qiskit-pulse using parametrized drag pulse and converting to signal.
- Simulating the equation over the length of the pulse sequence.
- Computing and return the infidelity (we minimize :math:`1-f(U)`).

.. jupyter-execute::

    def objective(params):

        instance = drag(params)

        # build a pulse schedule
        with pulse.build() as Xp:
            pulse.play(instance, pulse.DriveChannel(0))

        # convert from a pulse schedule to a list of signals
        converter = InstructionToSignals(dt, carriers={"d0": w})

        # get signals for the converter
        signal = converter.get_signals(Xp)

        result = ham_solver.solve(
            y0=np.eye(3, dtype=complex),
            t_span=[0, instance.duration * dt],
            signals=[signal],
            method='jax_odeint',
            atol=1e-8,
            rtol=1e-8
        )

        return 1. - fidelity(Array(result[0].y[-1])).data

6-B. Perform JAX transformations and optimize
---------------------------------------------

We set amplifier and beta as :math:`initial_params = np.array([0.2, 10,])`.
Before the optimization, the shape of the pulse is here.

.. jupyter-execute::

    initial_params = np.array([0.2, 10,])
    drag(initial_params).draw()

.. jupyter-execute::

    from jax import jit, value_and_grad
    from scipy.optimize import minimize

    jit_grad_obj = jit(value_and_grad(objective))

    opt_results = minimize(fun=jit_grad_obj, x0=initial_params, jac=True, method='L-BFGS-B',
    bounds=((0.,1.), (None, None)))

    print(opt_results.message)
    print(f"Optimized Amp is {opt_results.x[0]} and beta is {opt_results.x[1]}")
    print('Number of function evaluations: ' + str(opt_results.nfev))
    print('Function value: ' + str(opt_results.fun))



We can draw the optimized pulse, whose parameter is retrieved by :math:`opt_results.x`.

.. jupyter-execute::

    drag(opt_results.x).draw()