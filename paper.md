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
 - name: IBM Quantum, IBM Canada, Markham, ON, Canada
   index: 2
 - name: IBM Quantum, IBM Research Tokyo, Tokyo, Japan
   index: 3
date: 31 July 2023
bibliography: paper.bib
---

# Summary

Qiskit Dynamics is an open source Python library for numerically simulating the time dynamics of finite-dimensional quantum systems. To facilitate the fine-tuning of performance to each model, the core of the package is designed to provide flexible control over how the differential equations are solved. The user can choose between different array representations and backends (numpy/JAX, sparse v.s. dense) and different types and implementations of solvers (standard ODE v.s. geometric, matrix exponentiation-based solvers). The user can also choose the rotating frame in which to solve the differential equation, and can perform the rotating wave approximation. FOOTNOTE on what these are?

The package is geared towards research applications like ... optimal control, device physics simulation (parameter scans).requiring heavy use of simulation, such as optimization and 


The core of the package is designed to enable flexible configuration of the numerical details of the simulations: choice of array representations (dense v.s. sparse, and different array libraries), access to different types of underlying solvers (standard ODE v.s. geometric solvers), and has general tools for performing transformations on models of quantum systems that are commonly done by hand before input into simulation software (such as entering rotating frames and performing the rotating wave approximation). The package also contains implementations of algorithms for computing time-dependent perturbation theory expressions given in `@perturb1` and `@perturb2`, used in robust quantum control optimization. Lastly, the package provides a higher level object DynamicsBackend, which has the same interface as a real IBM quantum computer backend, enabling… cite qiskit pulse and qiskit experiments

Qiskit Dynamics is compatible with the JAX array library, and as such all core computations are just-in-time compilable and automatically differentiable.

Write about other dependencies?

# Statement of need

The numerical simulation of time-dependent quantum systems is useful for both understanding, as well as optimizing, quantum systems and devices. Understanding systems requires generating simulated data to compare to experimental observations, and models are either updated or validated depending on these comparisons. Model-based optimization of device design and control can be automated through simulation (references to OCT stuff)? These tasks and workflows are ultimately limited by simulation speed; the faster the simulation, the larger the dimension of the system and/or parameters spaces that can be efficiently explored. Lastly, simulation interfaces mimicking real devices enable learning and testing of workflows before using real device time.

The algorithms used in the perturbation theory module in Qiskit Dynamics are published (cite), and the package has been cited in (https://arxiv.org/abs/2212.12911).

# Other packages

Should expand more here on what each package specifically includes.

Due to the topic’s importance, many open source packages contain time-dependent quantum system simulation tools. In Python, these include QuTiP `@qutip`, TorchQuantum `@torchquantum`, and C3 `@C3`. C++ packages (also with Python interfaces) include lindbladmpo `@lindbladmpo` and Quandary `@quandary`. Packages also exist in other languages, such as the Hamiltonian open quantum system toolkit (HOQST) `@hoqst` in Julia, and Spinach `@spinach` in MATLAB.

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