---
title: 'Qiskit Dynamics: A Python package for simulating the time dynamics of quantum systems'
tags:
  - Python
  - quantum
  - quantum computer
  - pulse
  - control
authors:
  - name: Daniel Puzzuoli
    corresponding: true
    affiliation: 1
  - name: Christopher J. Wood
    affiliation: 2
  - name: Daniel J. Egger
    affiliation: 3
  - name: Benjamin Rosand
    affiliation: 4
  - name: Kento Ueda
    affiliation: 5
  - name: Ian Hincks
    affiliation: 6
  - name: Haggai Landa
    affiliation: 7
  - name: Moein Malekakhlagh
    affiliation: 2
  - name: Avery Parr
    affiliation: 8
  - name: Rupesh R. K.
    affiliation: 9
  - name: Matthew Treinish
    affiliation: 2
affiliations:
 - name: IBM Quantum, IBM Canada, Vancouver, BC, Canada
   index: 1
 - name: IBM Quantum, IBM T.J. Watson Research Center, Yorktown Heights, NY, USA
   index: 2
 - name: IBM Quantum, IBM Research Europe - Zurich, Ruschlikon, Switzerland
   index: 3
 - name: Department of Physics, Yale University, New Haven, CT, USA
   index: 4
 - name: IBM Quantum, IBM Research Tokyo, Tokyo, Japan
   index: 5
 - name: IBM Quantum, IBM Canada, Markham, ON, Canada
   index: 6
 - name: IBM Quantum, IBM Research Israel, Haifa, Israel
   index: 7
 - name: Department of Physics, Harvard University, Cambridge, MA, USA
   index: 8
 - name: School of Physics, University of Hyderabad, Hyderabad, India
   index: 9


date: 31 July 2023
bibliography: paper.bib
---

# Summary

Qiskit Dynamics is an open source Python library for numerically simulating the time dynamics of finite-dimensional quantum systems. The goal of the package is to provide flexible configuration of the numerical methods used for simulation: general tools for transforming models of quantum systems for more efficient simulation (rotating frames and the rotating wave approximation), choice of array representations (dense v.s. sparse, and different array libraries), and access to different types of underlying solvers (standard ODE v.s. geometric solvers). The package also contains advanced functionality for computing time-dependent perturbation theory expressions used in robust quantum control optimization `[@perturb1; @perturb2]`. 

As part of the Qiskit Ecosystem (https://qiskit.org/ecosystem), the package interfaces with other parts of Qiskit `[@Qiskit]`. Most notably, Qiskit Dynamics provides tools for simulating control sequences specified by Qiskit Pulse `[@alexander_qiskit_2020]`, which is used to specify hardware-level control of quantum computers. Higher level interfaces allow users to build and interact with simulation-based objects that target the same constraints (coupling map, timing, etc.) as a specified IBM Quantum computer.

Lastly, to facilitate high-perfomance applications, Qiskit Dynamics is compatible with the JAX array library `[@jax2018github]`. As such, all core computations are just-in-time compilable, automatically differentiable, and executable on GPU.

# Statement of need

Numerical simulation of time-dependent quantum systems is a useful tool in quantum device characterization and design, as well as control optimization. As these applications often involve the expensive process of repeatedly simulating a system across different parameters (e.g. in exploratory parameter scans, or in optimizations), it is important for users to be able to easily select the numerical methods that are most performant for their specific problem. The ability to automatically differentiate and compile simulations is also critical for flexible control optimization research. 

Furthermore, having a simulation-based drop-in replacement for real quantum computing systems is useful for developers building software tools for low-level control of experiments, such as Qiskit Pulse `[@alexander_qiskit_2020]` and Qiskit Experiments `[@kanazawa_qiskit_2023]`.

# Related open source packages

Due to its importance, many open source packages contain time-dependent quantum system simulation tools. In Python, these include QuTiP `[@qutip]`, TorchQuantum `[@torchquantum]`, and C3 `[@C3]`. C++ packages (also with Python interfaces) include lindbladmpo `[@lindbladmpo]` and Quandary `[@quandary]`. Packages also exist in other languages, such as the Hamiltonian open quantum system toolkit (HOQST) `[@hoqst]` and a Framework for Quantum Optimal Control `[@julia_qc]` in Julia, and Spinach `[@spinach]` in MATLAB.

# Documentation and community

Qiskit Dynamics documentation, including API docs and tutorials, is available at https://qiskit.org/ecosystem/dynamics/. A public slack channel for community discussion can be found here https://qiskit.slack.com/archives/C03E7UVCDEV.

# Acknowledgements

We would like to thank Helena Zhang, Naoki Kanazawa, Will Shanks, and Arthur Strauss for helpful discussions, reviews, and bug fixes.

# References