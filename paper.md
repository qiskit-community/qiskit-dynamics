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
    affiliation: 2
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    affiliation: 3
affiliations:
 - name: IBM Quantum, IBM T.J. Watson Research Center, Yorktown Heights, NY, USA
   index: 1
 - name: IBM Quantum, IBM Canada, Vancouver, BC, Canada
   index: 2
 - name: IBM Quantum, IBM Research Tokyo, Tokyo, Japan
   index: 3
date: 31 July 2023
bibliography: paper.bib
---

# Summary

Qiskit Dynamics is an open source Python library for numerically simulating the time dynamics of finite-dimensional quantum systems. The goal of the package is to provide flexible configuration of the numerical methods used for simulation: choice of array representations (dense v.s. sparse, and different array libraries), access to different types of underlying solvers (standard ODE v.s. geometric solvers), as well as general tools for transforming models of quantum systems for more efficient simulation (rotating frames and the rotating wave approximation). The package also contains advanced functionality for computing time-dependent perturbation theory expressions used in robust quantum control optimization `@perturb1`, `@perturb2`. 

As part of the Qiskit Ecosystem (https://qiskit.org/ecosystem), the package interfaces with other parts of Qiskit `@Qiskit`. Most notably, Qiskit Dynamics provides tools for simulating control sequences specified by Qiskit Pulse `@alexander_qiskit_2020`, which is used to specify hardware-level control of quantum computers. Higher level interfaces allow users to build and interact with simulation-based objects mimicking real IBM Quantum computers. (This last sentence is weird, having trouble summarizing the idea of ``DynamicsBackend`` without defining "backend".)

Lastly, to facilitate high-perfomance applications, Qiskit Dynamics is compatible with the JAX array library `@jax2018github`. As such, all core computations are just-in-time compilable, automatically differentiable, and executable on GPU.

# Statement of need

Numerical simulation of time-dependent quantum systems is a useful tool in quantum device characterization and design, as well as control optimization. As these applications often involve the expensive process of repeatedly simulating a system across different parameters (e.g. in exploratory parameter scans, or in optimizations), it is important for users to be able to easily select the numerical methods that are most performant for their specific problem. The ability to automatically differentiate and compile simulations is also critical for flexible control optimization research. 

Furthermore, having a simulation-based drop-in replacement for real quantum computing systems is useful for developers building software tools for describing low-level controls and experiments, such as Qiskit Pulse `@alexander_qiskit_2020` and Qiskit Experiments `@kanazawa_qiskit_2023`.

# Related packages

Due to its importance, many open source packages contain time-dependent quantum system simulation tools. In Python, these include QuTiP `@qutip`, TorchQuantum `@torchquantum`, and C3 `@C3`. C++ packages (also with Python interfaces) include lindbladmpo `@lindbladmpo` and Quandary `@quandary`. Packages also exist in other languages, such as the Hamiltonian open quantum system toolkit (HOQST) `@hoqst` in Julia, and Spinach `@spinach` in MATLAB.

# Documentation and community

Qiskit Dynamics documentation, including API docs and tutorials, is available at https://qiskit.org/ecosystem/dynamics/. A public slack channel for community discussion can be found here https://qiskit.slack.com/archives/C03E7UVCDEV.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References