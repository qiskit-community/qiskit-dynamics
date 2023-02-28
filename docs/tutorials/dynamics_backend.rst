Simulating backends at the pulse-level with ``DynamicsBackend``
===============================================================

In this tutorial we walk through how to use the ``DynamicsBackend`` class as a Qiskit
Dynamics-backed, pulse-level simulator of a real backend. In particular, we demonstrate how to
configure a ``DynamicsBackend`` to simulate pulse schedules, circuits whose gates have pulse
definitions, and calibration and characterization experiments from Qiskit Experiments.

The sections of this tutorial are as follows: 

1. Configure Dynamics to use JAX.
2. Instantiating a minimally-configured ``DynamicsBackend`` with a 2 qubit model.
3. Simulating pulse schedules on the ``DynamicsBackend``.
4. Simulating circuits at the pulse level using the ``DynamicsBackend``.
5. Simulating single-qubit calibration processes via Qiskit Experiments.
6. Simulating 2 qubit interaction characterization via the ``CrossResonanceHamiltonian`` experiment.

1. Configure Dynamics to use JAX
--------------------------------

Note that the ``DynamicsBackend`` internally performs just-in-time compilation automatically when
configured to use JAX.

.. jupyter-execute::

    # Configure to use JAX internally
    import jax
    jax.config.update("jax_enable_x64", True)
    jax.config.update('jax_platform_name', 'cpu')
    from qiskit_dynamics.array import Array
    Array.set_default_backend('jax')

2. Instantiating a ``DynamicsBackend``
--------------------------------------

To create the ``DynamicsBackend``, first specify a ``Solver`` instance using the model details. For
the model we will use a 2 qubit transmon model, with Hamiltonian:

.. math:: 
    
    H(t) = 2 \pi \nu_0 N_0 + 2 \pi \alpha_0 N_0 (N_0 - I) + 2 \pi \nu_1 N_1
    + 2 \pi \alpha_1 N_1(N_1 - I)\\ + 2 \pi J (a_0 + a_0^\dagger)(a_1 + a_1^\dagger) \\ 
    + 2 \pi r_0 s_0(t)(a_0 + a_0^\dagger) + 2 \pi r_1 s_1(t)(a_1 + a_1^\dagger),

where 

- :math:`\nu_0` and :math:`\nu_1` are the qubit frequencies, 
- :math:`\alpha_0` and :math:`\alpha_1` are the qubit anharmonicities, 
- :math:`J` is the coupling strength, 
- :math:`r_0` and :math:`r_1` are the Rabi strengths, and :math:`s_0(t)` and :math:`s_1(t)` are the
  drive signals, 
- :math:`a_j` and :math:`a_j^\dagger` are the lowering and raising operators for qubit :math:`j`,
  and - :math:`N_0` and :math:`N_1` are the number operators.

.. jupyter-execute::

    import numpy as np
    
    dim = 3
    
    v0 = 4.86e9
    anharm0 = -0.32e9
    r0 = 0.22e9
    
    v1 = 4.97e9
    anharm1 = -0.32e9
    r1 = 0.26e9
    
    J = 0.002e9
    
    a = np.diag(np.sqrt(np.arange(1, dim)), 1)
    adag = np.diag(np.sqrt(np.arange(1, dim)), -1)
    N = np.diag(np.arange(dim))
    
    ident = np.eye(dim, dtype=complex)
    full_ident = np.eye(dim**2, dtype=complex)
    
    N0 = np.kron(ident, N)
    N1 = np.kron(N, ident)
    
    a0 = np.kron(ident, a)
    a1 = np.kron(a, ident)
    
    a0dag = np.kron(ident, adag)
    a1dag = np.kron(adag, ident)
    
    
    static_ham0 = 2 * np.pi * v0 * N0 + np.pi * anharm0 * N0 * (N0 - full_ident)
    static_ham1 = 2 * np.pi * v1 * N1 + np.pi * anharm1 * N1 * (N1 - full_ident)
    
    static_ham_full = static_ham0 + static_ham1 + 2 * np.pi * J * ((a0 + a0dag) @ (a1 + a1dag))
    
    drive_op0 = 2 * np.pi * r0 * (a0 + a0dag)
    drive_op1 = 2 * np.pi * r1 * (a1 + a1dag)

Construct the ``Solver`` using the model details, including parameters necessary for pulse
simulation.

**To do: add reference to pulse simulation tutorial**

.. jupyter-execute::

    from qiskit_dynamics import Solver
    
    # build solver
    dt = 1/4.5e9
    
    solver = Solver(
        static_hamiltonian=static_ham_full,
        hamiltonian_operators=[drive_op0, drive_op1, drive_op0, drive_op1],
        rotating_frame=static_ham_full,
        hamiltonian_channels=['d0', 'd1', 'u0', 'u1'],
        channel_carrier_freqs={'d0': v0, 'd1': v1, 'u0': v1, 'u1': v0},
        dt=dt,
    )

Next, instantiate the ``DynamicsBackend``. The ``solver`` is used for simulation, ``subsystem_dims``
indicates how the full system decomposes for measurement data computation, and ``solver_options``
are consistent options used by ``Solver.solve`` when simulating the differential equation.

Note that, to enable the internal automatic jit-compilation, we choose a JAX integration method.

.. jupyter-execute::

    from qiskit_dynamics import DynamicsBackend
    
    # Consistent solver option to use throughout notebook
    solver_options = {'method': 'jax_odeint', 'atol': 1e-6, 'rtol': 1e-8}
    
    backend = DynamicsBackend(
        solver=solver,
        subsystem_dims=[dim, dim], # for computing measurement data
        solver_options=solver_options, # to be used every time run is called
    )

Alternatively to the above, the ``DynamicsBackend.from_backend`` method can be used to build the
``DymamicsBackend`` from an existing backend. The above model, which was built manually, was taken
from qubit :math:`0` and :math:`1` of ``almaden``.

3. Simulate a list of schedules
-------------------------------

With the above backend, we can already simulate a list of pulse schedules. The code below generates
a list of schedules specifying experiments on qubit :math:`0`. The schedule is chosen to demonstrate
that the usual instructions work on the ``DynamicsBackend``.

.. jupyter-execute::

    %%time
    
    from qiskit import pulse
    
    sigma = 128
    num_samples = 256
    
    schedules = []
    
    for amp in np.linspace(0., 1., 10):
        gauss = pulse.library.Gaussian(
            num_samples, amp, sigma, name="Parametric Gauss"
        )
    
        with pulse.build() as schedule:
            with pulse.align_right():
                pulse.play(gauss, pulse.DriveChannel(0))
                pulse.shift_phase(0.5, pulse.DriveChannel(0))
                pulse.shift_frequency(0.1, pulse.DriveChannel(0))
                pulse.play(gauss, pulse.DriveChannel(0))
                pulse.acquire(duration=1, qubit_or_channel=0, register=pulse.MemorySlot(0))
            
        schedules.append(schedule)
        
    job = backend.run(schedules, shots=100)
    result = job.result()

Retrieve the counts for one of the experiments as would be done using the results object from a real
backend.

.. jupyter-execute::

    result.get_counts(3)

4. Simulating circuits at the pulse level using ``DynamicsBackend``
-------------------------------------------------------------------

For the ``DynamicsBackend`` to simulate a circuit, each circuit element must have a corresponding
pulse schedule. These schedules can either be specified in the gates themselves, by attaching
calibrations, or by adding instructions to the ``Target`` contained in the ``DynamicsBackend``.

4.1 Simulating circuits with attached calibrations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build a simple circuit. Here we build one consisting of a single Hadamard gate on qubit 0, followed
by measurement.

.. jupyter-execute::

    from qiskit import QuantumCircuit
    
    circ = QuantumCircuit(1, 1)
    circ.h(0)
    circ.measure([0], [0])
    
    circ.draw("mpl")

Next, attach a calibration for the Hadamard gate on qubit 0 to the circuit. Note that here are only
demonstrating the mechanics of adding a calibration; we have not actually chosen the pulse to
implement a Hadamard gate.

.. jupyter-execute::

    with pulse.build() as h_q0:
        pulse.play(
            pulse.library.Gaussian(duration=256, amp=0.2, sigma=50, name='custom'),
            pulse.DriveChannel(0)
        )
    
    circ.add_calibration('h', [0], h_q0)

Call run on the circuit, and get counts as usual.

.. jupyter-execute::

    %time res = backend.run(circ).result()
    
    res.get_counts(0)

4.2 Simulating circuits via gate definitions in the backend ``Target``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Alternatively to the above work flow, add the above schedule as the pulse-level definition of the
Hadamard gate on qubit 0.

.. jupyter-execute::

    from qiskit.circuit.library import HGate
    from qiskit.transpiler import InstructionProperties
    
    backend.target.add_instruction(HGate(), {(0,): InstructionProperties(calibration=h_q0)})

Rebuild the same circuit, however this time we do not need to add the calibration for the Hadamard
gate to the circuit object.

.. jupyter-execute::

    circ2 = QuantumCircuit(1, 1)
    circ2.h(0)
    circ2.measure([0], [0])
    
    %time result = backend.run(circ2).result()

.. jupyter-execute::

    result.get_counts(0)

5. Simulating calibration of single qubit gates using Qiskit Experiments
------------------------------------------------------------------------

Next, we calibrate ``X`` and ``SX`` gates on both qubits modeled in the ``DynamicsBackend``.

**To do: add reference to the single qubit calibration tutorial for
Qiskit Experiments and say that we’re walking through this.**

5.1 Configure the ``Target`` to include single qubit instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable running of the single qubit experiments, we add the following to the target: - Qubit
frequency properties (needed by the ``RoughFrequencyCal`` experiment). - ``X`` and ``SX`` gate
instructions, which the transpiler needs to check are supported by the backend. - Add definitions of
``RZ`` gates as phase shifts.

.. jupyter-execute::

    from qiskit.circuit.library import XGate, SXGate, RZGate
    from qiskit.circuit import Parameter
    from qiskit.providers.backend import QubitProperties
    
    target = backend.target
    
    # qubit properties
    target.qubit_properties = [QubitProperties(frequency=v0), QubitProperties(frequency=v1)]
    
    # add instructions
    target.add_instruction(XGate())
    target.add_instruction(SXGate())
    
    # Add RZ instruction as phase shift for drag cal
    phi = Parameter('phi')
    with pulse.build() as rz0:
        pulse.shift_phase(phi, pulse.DriveChannel(0))
    
    with pulse.build() as rz1:
        pulse.shift_phase(phi, pulse.DriveChannel(1))
    
    target.add_instruction(
        RZGate(phi),
        {(0,): InstructionProperties(calibration=rz0), (1,): InstructionProperties(calibration=rz1)}
    )

5.2 Prepare ``Calibrations`` object
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Following **insert link to tutorial**, we prepare the ``Calibrations`` object.

**TO DO: insert link above**

.. jupyter-execute::

    import pandas as pd
    from qiskit_experiments.calibration_management.calibrations import Calibrations
    
    cals = Calibrations()
    
    dur = Parameter("dur")
    sigma = Parameter("σ")
    drive = pulse.DriveChannel(Parameter("ch0"))
    
    # Define and add template schedules.
    with pulse.build(name="x") as x:
        pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("β")), drive)
    
    with pulse.build(name="sx") as sx:
        #pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("β")), drive)
        pulse.play(pulse.Drag(dur, Parameter("amp"), sigma, Parameter("β")), drive)
    
    cals.add_schedule(x, num_qubits=1)
    cals.add_schedule(sx, num_qubits=1)
    
    # add parameter guesses
    for sched in ["x", "sx"]:
        cals.add_parameter_value(80, "σ", schedule=sched)
        cals.add_parameter_value(0.5, "β", schedule=sched)
        cals.add_parameter_value(320, "dur", schedule=sched)
        cals.add_parameter_value(0.5, "amp", schedule=sched)
    
    pd.DataFrame(**cals.parameters_table(qubit_list=[0, ()]))

5.3 Rough frequency cals
~~~~~~~~~~~~~~~~~~~~~~~~

Run frequency calibration experiments. We perturb the frequency estimate to imitate not knowing the
frequency ahead of time.

.. jupyter-execute::

    from qiskit_experiments.library.calibration.rough_frequency import RoughFrequencyCal
    
    # experiment for qubit 0
    freq0_estimate = v0 + 0.5e7
    frequencies = np.linspace(freq0_estimate -15e6, freq0_estimate + 15e6, 27)
    spec0 = RoughFrequencyCal(0, cals, frequencies, backend=backend)
    spec0.set_experiment_options(amp=0.005)
    
    # experiment for qubit 1
    freq1_estimate = v1 + 1e7
    frequencies = np.linspace(freq1_estimate -15e6, freq1_estimate + 15e6, 27)
    spec1 = RoughFrequencyCal(1, cals, frequencies, backend=backend)
    spec1.set_experiment_options(amp=0.005)

Visualize the first circuit for qubit 0.

.. jupyter-execute::

    spec0.circuits()[0].draw(output="mpl")

Run the spectroscopy experiments.

.. jupyter-execute::

    %%time
    spec0_data = spec0.run().block_for_results()
    spec1_data = spec1.run().block_for_results()


Plot the simulated data for both qubits.

.. jupyter-execute::

    spec0_data.figure(0)

.. jupyter-execute::

    spec1_data.figure(0)

5.4 Rough amplitude calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, run a rough amplitude calibration for ``X`` and ``SX`` gates for both qubits. First, build the
experiments.

.. jupyter-execute::

    from qiskit_experiments.library.calibration import RoughXSXAmplitudeCal
    
    # rabi experiments for qubit 0
    rabi0 = RoughXSXAmplitudeCal(0, cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 27))
    
    # rabi experiments for qubit 1
    rabi1 = RoughXSXAmplitudeCal(1, cals, backend=backend, amplitudes=np.linspace(-0.2, 0.2, 27))

Run the Rabi experiments.

.. jupyter-execute::

    %%time
    rabi0_data = rabi0.run().block_for_results()
    rabi1_data = rabi1.run().block_for_results()

Plot the results.

.. jupyter-execute::

    rabi0_data.figure(0)

.. jupyter-execute::

    rabi1_data.figure(0)

Observe the updated parameters for qubit 0.

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[0, ()], parameters="amp"))

5.5 Rough Drag parameter calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run rough Drag parameter calibration for the ``X`` and ``SX`` gates.
This follows the same procedure as above.

.. jupyter-execute::

    from qiskit_experiments.library.calibration import RoughDragCal
    
    cal_drag0 = RoughDragCal(0, cals, backend=backend, betas=np.linspace(-20, 20, 15))
    cal_drag1 = RoughDragCal(1, cals, backend=backend, betas=np.linspace(-20, 20, 15))
    
    cal_drag0.set_experiment_options(reps=[3, 5, 7])
    cal_drag1.set_experiment_options(reps=[3, 5, 7])
    
    cal_drag0.circuits()[5].draw(output='mpl')

.. jupyter-execute::

    %%time
    drag0_data = cal_drag0.run().block_for_results()
    drag1_data = cal_drag1.run().block_for_results()

.. jupyter-execute::

    drag0_data.figure(0)


.. jupyter-execute::

    drag1_data.figure(0)


5.6 Fine amplitude calibration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, run fine amplitude calibration for both qubits. Start with the
``X`` gate.

.. jupyter-execute::

    from qiskit_experiments.library.calibration.fine_amplitude import FineXAmplitudeCal
    
    amp_x_cal0 = FineXAmplitudeCal(0, cals, backend=backend, schedule_name="x")
    amp_x_cal1 = FineXAmplitudeCal(1, cals, backend=backend, schedule_name="x")
    
    amp_x_cal0.circuits()[5].draw(output="mpl")


.. jupyter-execute::

    %%time
    data_fine0 = amp_x_cal0.run().block_for_results()
    data_fine1 = amp_x_cal1.run().block_for_results()

.. jupyter-execute::

    data_fine0.figure(0)


.. jupyter-execute::

    data_fine1.figure(0)

Next, run fine calibration on the ``SX`` gates.

.. jupyter-execute::

    # Do SX Cal
    from qiskit_experiments.library.calibration.fine_amplitude import FineSXAmplitudeCal
    
    amp_sx_cal0 = FineSXAmplitudeCal(0, cals, backend=backend, schedule_name="sx")
    amp_sx_cal1 = FineSXAmplitudeCal(1, cals, backend=backend, schedule_name="sx")
    
    amp_sx_cal0.circuits()[5].draw(output="mpl")


.. jupyter-execute::

    %%time
    data_fine_sx0 = amp_sx_cal0.run().block_for_results()
    data_fine_sx1 = amp_sx_cal1.run().block_for_results()

.. jupyter-execute::

    pd.DataFrame(**cals.parameters_table(qubit_list=[0, ()], parameters="amp"))

6. Simulating a cross resonance characterization experiment
-----------------------------------------------------------

Finally, simulate the ``CrossResonanceHamiltonian`` characterization
experiment.

First, we further configure the backend to run this experiment. This
requires: - Adding the custom gate used in the experiment as a valid
instruction in the ``Target``. - Defining the control channel map, which
the experiment requires.

.. jupyter-execute::

    # add the gate to the target
    from qiskit_experiments.library import CrossResonanceHamiltonian
    backend.target.add_instruction(
        instruction=CrossResonanceHamiltonian.CRPulseGate(width=Parameter("width")), 
        properties={(0, 1): None, (1, 0): None}
    )
    
    # set the control channel map
    backend.set_options(control_channel_map={(0, 1): 0, (1, 0): 1})

Build the characterization experiment object, and set the instruction
map in the transpilation options to use the single qubit gates
calibrated above.

.. jupyter-execute::

    cr_ham_experiment = CrossResonanceHamiltonian(
        qubits=(0, 1), 
        flat_top_widths=np.linspace(0, 5000, 17), 
        backend=backend
    )
    
    cr_ham_experiment.set_transpile_options(inst_map=cals.default_inst_map)

.. jupyter-execute::

    cr_ham_experiment.circuits()[10].draw("mpl")

Run the simulation.

.. jupyter-execute::

    %time data_cr = cr_ham_experiment.run().block_for_results()


.. jupyter-execute::

    data_cr.figure(0)
