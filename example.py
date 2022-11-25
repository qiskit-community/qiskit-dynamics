from functools import partial
import jax
from qiskit import pulse
from qiskit_dynamics.pulse import InstructionToSignals
import numpy as np
import sympy as sym
from qiskit_dynamics.array import Array
Array.set_default_backend('jax')

def _lifted_gaussian(
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

@partial(jax.jit)
def run_simulation(amp, sigma):
    # angle = 0.1
    # # pulse.Gaussian(duration=160, amp=amp, sigma=sigma, angle=np.pi)
    # parameters = {"amp": amp, "sigma": sigma, "angle": angle}

    #     # Prepare symbolic expressions
    # _t, _duration, _amp, _sigma, _angle = sym.symbols("t, duration, amp, sigma, angle")
    # _center = _duration / 2

    # envelope_expr = _amp * _lifted_gaussian(_t, _center, _duration + 1, _sigma)
    # # To conform with some old tests, the angle part is inserted only when needed.
    # if angle != 0:
    #     envelope_expr *= sym.exp(1j * _angle)

    # consts_expr = _sigma > 0
    # valid_amp_conditions_expr = sym.Abs(_amp) <= 1.0

    # gaussian_pulse = pulse.SymbolicPulse(
    #     pulse_type="Gaussian",
    #     duration=100,
    #     parameters=parameters,
    #     name="GP1",
    #     envelope=envelope_expr,
    #     constraints=consts_expr,
    #     valid_amp_conditions=valid_amp_conditions_expr,
    # )
    converter = InstructionToSignals(dt=1, carriers=None)
    with pulse.build() as schedule:
        pulse.play(pulse.Gaussian(duration=160, amp=amp, sigma=sigma, angle=0), pulse.DriveChannel(0))
    signals = converter.get_signals(schedule)
run_simulation(0.983,40)

# amp=0.983
# sigma=2
# print(pulse.Gaussian(duration=160, amp=amp, sigma=sigma, angle=np.pi))