from copy import deepcopy

from qiskit import QiskitError
from qiskit.transpiler import Target

from .dynamics_backend import DynamicsBackend
from ..systems import (
    ExchangeInteraction,
    T1Relaxation,
    T2Relaxation,
    DuffingOscillator,
    Subsystem,
    IdealQubit,
    QuantumSystemModel,
)
from typing import List, Optional, Dict, Tuple
from qiskit.pulse import DriveChannel, ControlChannel


class FixedFrequencyTransmonBackend(DynamicsBackend):
    def __init__(
        self,
        dims: List[int],
        freqs: List[float],
        drive_freqs: List[float],
        couplings: Optional[Dict[Tuple[int, int], float]] = None,
        anharmonicities: Optional[List[float]] = None,
        t1s: Optional[List[Optional[float]]] = None,
        t2s: Optional[List[Optional[float]]] = None,
        array_library: str = "numpy",
        dt: Optional[float] = None,
        target: Optional[Target] = None,
        **options,
    ):
        """
        Template class for a fixed frequency transmon backend.

        Args:
            dims: List of dimensions for each qubit.
            freqs: List of frequencies for each qubit.
            drive_freqs: List of drive frequencies for each qubit.
            couplings: Dictionary of coupling strengths between qubits.
            anharmonicities: List of anharmonicities for each qubit.
            t1s: List of T1 relaxation times for each qubit.
            t2s: List of T2 relaxation times for each qubit.
            dt: Sample rate for simulating pulse schedules.
            target: Target object for the DynamicsBackend.
            options: Additional options for the DynamicsBackend.
        """

        if not len(dims) == len(freqs) == len(drive_freqs):
            raise QiskitError("Number of dimensions and frequencies must match.")
        if anharmonicities is None:
            anharmonicities = [None for _ in range(len(freqs))]
        if couplings is None:
            couplings = {}
        if t1s is None:
            t1s = [None for _ in range(len(freqs))]
        if t2s is None:
            t2s = [None for _ in range(len(freqs))]

        subsystems = [Subsystem(f"Q{i}", dim) for i, dim in enumerate(dims)]
        model_list = []
        for i in range(len(subsystems)):
            if anharmonicities[i] is not None:
                model_list.append(
                    DuffingOscillator(
                        subsystems[i],
                        freqs[i],
                        anharmonicities[i],
                        drive_freqs[i],
                        drive_channel=DriveChannel(i),
                    )
                )
            else:
                model_list.append(
                    IdealQubit(
                        subsystems[i], freqs[i], drive_freqs[i], drive_channel=DriveChannel(i)
                    )
                )
            if t1s[i] is not None:
                model_list.append(T1Relaxation(subsystems[i], t1s[i]))
            if t2s[i] is not None:
                model_list.append(T2Relaxation(subsystems[i], t2s[i]))

        control_channel_map = {}
        for idx, (qubits, coupling) in enumerate(couplings.items()):
            i, j = qubits
            model_list.append(ExchangeInteraction(subsystems[i], subsystems[j], coupling))
            # Add cross resonance term for the drive
            cross_resonance_tone = deepcopy(model_list[i].drive_hamiltonians[0])
            model_list[i].drive_hamiltonians.append(cross_resonance_tone)
            model_list[i].hamiltonian_channels.append(ControlChannel(idx).name)
            model_list[i].channel_carrier_freqs[ControlChannel(idx).name] = freqs[j]
            control_channel_map[(i, j)] = idx

        model: QuantumSystemModel = model_list[0]
        for m in model_list[1:]:
            model += m
        solver = model.get_Solver(
            rotating_frame="static_hamiltonian", array_library=array_library, dt=dt, validate=True
        )
        
        if 'control_channel_map' not in options:
            options['control_channel_map'] = control_channel_map
        if 'subsystem_dims' not in options:
            options['subsystem_dims'] = dims
        else:
            raise QiskitError("subsystem_dims option not consistent with dims argument.")
        
        super().__init__(solver, target=target, **options)
