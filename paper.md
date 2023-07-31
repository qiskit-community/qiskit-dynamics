---
title: 'Qiskit Dynamics: A Python package for simulating quantum systems'
tags:
  - Python
  - quantum
  - quantum computer
  - pulse
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

Qiskit Dynamics is an open source Python library for numerically simulating the time dynamics of finite-dimensional quantum systems. The package provides flexible configuration over the numerical details of the simulations: choice of array representations (dense v.s. sparse, and different array libraries), access to different types of underlying solvers (standard ODE v.s. geometric solvers), and has general tools for performing transformations on models of quantum systems that are commonly done by hand before input into simulation software (such as the rotating wave approximation). The package also contains advanced functionality for computing time-dependent perturbation theory expressions used in robust quantum control optimization (cite). Lastly, the package provides a higher level object DynamicsBackend, which has the same interface as a real IBM quantum computer backend, enabling… cite qiskit pulse and qiskit experiments

Qiskit Dynamics is compatible with the JAX array library, and as such all core computations are just-in-time compilable and automatically differentiable.

Write about other dependencies?

# Statement of need

The numerical simulation of time-dependent quantum systems is useful for both understanding, as well as optimizing, quantum systems and devices. Understanding systems requires generating simulated data to compare to experimental observations, and models are either updated or validated depending on these comparisons. Model-based optimization of device design and control can be automated through simulation (references to OCT stuff)? These tasks and workflows are ultimately limited by simulation speed; the faster the simulation, the larger the dimension of the system and/or parameters spaces that can be efficiently explored. Lastly, simulation interfaces mimicking real devices enable learning and testing of workflows before using real device time.

The algorithms used in the perturbation theory module in Qiskit Dynamics are published (cite), and the package has been cited in (https://arxiv.org/abs/2212.12911).

# Other packages

Due to the topic’s importance, many open source packages contain quantum system simulation tools. 

QuTiP `@qutip`
TorchQuantum `@torchquantum`
C3 `@C3`
lindbladmpo `@lindbladmpo`


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