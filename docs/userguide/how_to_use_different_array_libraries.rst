.. _how-to use different array libraries:

How-to use different array libraries and types with Qiskit Dynamics
===================================================================

Main points:
- You can use JAX or numpy
- For models/solving you can use JAX, JAX sparse, numpy, scipy sparse

The simulations and computations in Qiskit Dynamics can be executed with different array libraries
and types. A user can choose to use either NumPy or JAX to define their models, and the code in
Qiskit Dynamics will execute as if the array operations had been natively written in either library.
Additionally, a user can specify that the operators in a model be stored in sparse types offered by
SciPy or JAX (see :ref:`configuring simulations for performance <configuring simulations>`).
Internally, Qiskit Dynamics utilizes `Arraylias <https://qiskit-extensions.github.io/arraylias/>`_
to dispatch computations on different array types to the appropriate library function.

This guide addresses the following topics:

1. Example: How-to use either NumPy or JAX when building a :class:`.Signal`.
2. How-to use the Qiskit Dynamics NumPy and SciPy aliased libraries.


1. Example: How-to use either NumPy or JAX when building a :class:`.Signal`
---------------------------------------------------------------------------

First, configure JAX and import array libraries.

.. jupyter-execute::

    # configure jax to use 64 bit mode
    import jax
    jax.config.update("jax_enable_x64", True)

    # tell JAX we are using CPU
    jax.config.update('jax_platform_name', 'cpu')

    import numpy as np
    import jax.numpy as jnp


Defining equivalent :class:`.Signal` instances, with envelope implemented in either NumPy or JAX.

.. jupyter-execute::

    from qiskit_dynamics import Signal

    def envelope_numpy(t):
        return np.exp((t - 0.5)**2 / 0.025)
    
    def envelope_jax(t):
        return jnp.exp((t - 0.5)**2 / 0.025)
    
    signal_numpy = Signal(envelope=envelope_numpy)
    signal_jax = Signal(envelope=envelope_jax)


Evaluation of ``signal_numpy`` is executed with NumPy:

.. jupyter-execute::

    type(signal_numpy(0.1))

Evaluation of ``signal_jax`` is executed with JAX:

.. jupyter-execute::

    type(signal_jax(0.1))

JAX transformations can be applied to ``signal_jax``, e.g. just-in-time compilation:

.. jupyter-execute::

    from jax import jit

    jit_signal_jax = jit(signal_jax)
    jit_signal_jax(0.1)


2. How-to use the Qiskit Dynamics NumPy and SciPy aliased libraries
-------------------------------------------------------------------

Internally, Qiskit Dynamics uses an extension of the default NumPy and SciPy array libraries offered
by `Arraylias <https://qiskit-extensions.github.io/arraylias/>`_. These can be imported as:

.. jupyter-execute::
    
    # alias for NumPy and corresponding aliased library
    from qiskit_dynamics import DYNAMICS_NUMPY_ALIAS
    from qiskit_dynamics import DYNAMICS_NUMPY

    # alias for SciPy and corresponding aliased library
    from qiskit_dynamics import DYNAMICS_SCIPY_ALIAS
    from qiskit_dynamics import DYNAMICS_SCIPY

See the `Arraylias documentation <https://qiskit-extensions.github.io/arraylias/>`_ for how the
general library aliasing framework works, as well as the Qiskit Dynamics submodule :mod:`.arraylias`
for a description of how the default NumPy and SciPy aliases have been extended for use in this
package.

################################
# OLD
################################


This guide addresses the following topics:

1. How do I configure dynamics to run with JAX?
2. How do I write code using dispatch that can be executed with either
   ``numpy`` or JAX?
3. How do I write JAX-transformable functions using the objects and
   functions in ``qiskit-dynamics``?
4. Gotchas when using JAX with dynamics.

1. How do I configure dynamics to run with JAX?
-----------------------------------------------

The :class:`.Array` class provides a means of controlling whether array
operations are performed using ``numpy`` or ``jax.numpy``. In many
cases, the “default backend” is used to determine which of the two
options is used.

.. jupyter-execute::

    ################################################################################# 
    # Remove this
    #################################################################################
    import warnings
    warnings.filterwarnings("ignore")
    
    # configure jax to use 64 bit mode
    import jax
    jax.config.update("jax_enable_x64", True)

    # tell JAX we are using CPU
    jax.config.update('jax_platform_name', 'cpu')

    import jax.numpy as jnp


2. How do I write code using Array that can be executed with either ``numpy`` or JAX?
-------------------------------------------------------------------------------------

The ``Array`` class wraps both ``numpy`` and ``jax.numpy``
arrays. The particular type is indicated by the ``backend`` property,
and ``numpy`` functions called on an :class:`.Array` will automatically be
dispatched to ``numpy`` or ``jax.numpy`` based on the :class:`.Array`\ ’s
backend. See the API documentation for ``qiskit_dynamics.array`` for
details.

3. How do I write JAX-transformable functions using the objects and functions in ``qiskit-dynamics``?
-----------------------------------------------------------------------------------------------------

JAX-transformable functions must be:

  - JAX-executable.
  - Take JAX arrays as input and output (see the
    `JAX documentation <https://jax.readthedocs.io/en/latest/>`__
    for more details on accepted input and output types).
  - Pure, in the sense that they have no side-effects.

The previous section shows how to handle the first two points using
:class:`.Array`. The last point further restricts the type of
code that can be safely transformed. Qiskit Dynamics uses various objects which
can be updated by setting properties (models, solvers). If a function to
be transformed requires updating an already-constructed object of this
form, it is necessary to first make a *copy*.

We demonstrate this process for both just-in-time compilation and
automatic differentiation in the context of an anticipated common
use-case: parameterized simulation of a model of a quantum system.

3.1 Just-in-time compiling a parameterized simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"Just-in-time compiling" a function means to compile it at run time. Just-in-time compilation
incurs an initial cost associated with the construction of the compiled function,
but subsequent calls to the function will generally be faster than the uncompiled version.
In JAX, just-in-time compilation is performed using the ``jax.jit`` function,
which transforms a JAX-compatible function into optimized code using
`XLA <https://www.tensorflow.org/xla>`__. We demonstrate here how, using the JAX backend,
functions built using Qiskit Dynamics can be
just-in-time compiled, resulting in faster simulation times.

For convenience, the ``wrap`` function can be used to transform
``jax.jit`` to also work on functions that have :class:`.Array` objects as
inputs and outputs.

Construct a :class:`.Solver` instance with a model that will be used to solve.

.. jupyter-execute::

    import numpy as np
    from qiskit.quantum_info import Operator
    from qiskit_dynamics import Solver, Signal

    r = 0.5
    w = 1.
    X = Operator.from_label('X')
    Z = Operator.from_label('Z')

    static_hamiltonian = 2 * np.pi * w * Z/2
    hamiltonian_operators = [2 * np.pi * r * X/2]

    solver = Solver(
        static_hamiltonian=static_hamiltonian,
        hamiltonian_operators=hamiltonian_operators,
        rotating_frame=static_hamiltonian
    )


Next, define the function to be compiled:

  - The input is the amplitude of a constant-envelope signal on resonance, driven over time
    :math:`[0, 3]`.
  - The output is the state of the system, starting in the ground state, at
    ``100`` points over the total evolution time.

Note, as described at the beginning of this section, we need to make a copy of ``solver``
before setting the signals, to ensure the simulation function remains pure.

.. jupyter-execute::

    def sim_function(amp):

        # define a constant signal
        signals = [Signal(amp, carrier_freq=w)]

        # simulate and return results
        results = solver.solve(
            t_span=[0, 3.],
            y0=np.array([0., 1.], dtype=complex),
            signals=signals,
            t_eval=np.linspace(0, 3., 100),
            method='jax_odeint'
        )

        return results.y

Compile the function.

.. jupyter-execute::

    from jax import jit
    fast_sim = jit(sim_function)

The first time the function is called, JAX will compile an
`XLA <https://www.tensorflow.org/xla>`__ version of the function, which is then executed.
Hence, the time taken on the first call *includes* compilation time.

.. jupyter-execute::

    %time ys = fast_sim(1.).block_until_ready()


On subsequent calls the compiled function is directly executed,
demonstrating the true speed of the compiled function.

.. jupyter-execute::

    %timeit fast_sim(1.).block_until_ready()


We use this function to plot the :math:`Z` expectation value over a
range of input amplitudes.

.. jupyter-execute::

    import matplotlib.pyplot as plt

    for amp in np.linspace(0, 1, 10):
        ys = fast_sim(amp)
        plt.plot(np.linspace(0, 3., 100), np.real(np.abs(ys[:, 0])**2-np.abs(ys[:, 1])**2))


3.2 Automatically differentiating a parameterized simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, we use ``jax.grad`` to automatically differentiate a parameterized
simulation. In this case, ``jax.grad`` requires that the output be a
real number, so we specifically compute the population in the excited
state at the end of the previous simulation

.. jupyter-execute::

    def excited_state_pop(amp):
        yf = sim_function(amp)[-1]
        return jnp.abs(yf[0])**2

Wrap ``jax.grad`` in the same way, then differentiate and compile
``excited_state_pop``.

.. jupyter-execute::

    from jax import grad
    excited_pop_grad = jit(grad(excited_state_pop))

As before, the first execution includes compilation time.

.. jupyter-execute::

    %time excited_pop_grad(1.).block_until_ready()


Subsequent runs of the function reveal the execution time once compiled.

.. jupyter-execute::

    %timeit excited_pop_grad(1.).block_until_ready()


3. Pitfalls when using JAX with Dynamics
----------------------------------------

4.1 JAX must be set as the default backend before building any objects in Qiskit Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To get dynamics to run with JAX, it is necessary to configure dynamics
to run with JAX *before* building any objects or running any functions.
The internal behaviour of some objects is modified by what the default
backend is *at the time of instantiation*. For example, at instantiation
the operators in a model or :class:`.Solver` instance will be wrapped in an
:class:`.Array` whose backend is the current default backend, and changing the
default backend after building the object won’t change this.

4.2 Running Dynamics with JAX on CPU vs GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Certain JAX-based features in Dynamics are primarily recommended for use only with CPU
or only with GPU. In such cases, a warning is raised if non-recommended hardware is used,
however users are not prevented from configuring Dynamics and JAX in whatever way they choose.

Instances of such features are:
  * Setting ``evaluation_mode='sparse'`` for solvers and models is only recommended for use on CPU.
  * Parallel fixed step solver options in ``solve_lmde`` are recommended only for use on GPU.
