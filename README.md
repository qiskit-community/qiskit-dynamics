# Qiskit Dynamics

[![License](https://img.shields.io/github/license/Qiskit/qiskit-experiments.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)

**This repo is still in the early stages of development, there will be breaking API changes**

Qiskit Dynamics is an open-source project for building, transforming, and solving
time-dependent quantum systems in Qiskit.

The goal of Qiskit Dynamics is to provide access to different numerical
methods for solving differential equations, and to automate common processes typically performed by hand,
e.g. applying frame transformations or rotating wave approximations to system and control Hamiltonians.

Qiskit Dynamics can be configured to use either
[NumPy](https://github.com/numpy/numpy) or [JAX](https://github.com/google/jax)
as the backend for array operations. [NumPy](https://github.com/numpy/numpy) is the default,
and [JAX](https://github.com/google/jax) is an optional dependency.
[JAX](https://github.com/google/jax) provides just-in-time compilation, automatic differentiation,
and GPU execution, and therefore is well-suited to tasks involving repeated
evaluation of functions with different parameters; E.g. simulating a model of a quantum system
over a range of parameter values, or optimizing the parameters of control sequence.

Reference documentation may be found [here](https://qiskit.org/documentation/dynamics/),
including [tutorials](https://qiskit.org/documentation/dynamics/tutorials),
[user guide](https://qiskit.org/documentation/dynamics/userguide),
and [API reference](https://qiskit.org/documentation/dynamics/apidocs).

## Installation

Qiskit Dynamics may be installed using pip via:

```
pip install qiskit-dynamics
```

Additionally, Qiskit Dynamics may be installed simultaneously with the CPU version of
JAX via:

```
pip install "qiskit-dynamics[jax]"
```

Installing JAX with GPU support must be done manually, for instructions refer to the
[JAX installation guide](https://github.com/google/jax#installation).


## Contribution Guidelines

If you'd like to contribute to Qiskit Dynamics, please take a look at our
[contribution guidelines](CONTRIBUTING.md). This project adheres to Qiskit's
[code of conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-dynamics/issues) for
tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
and use our [Qiskit Slack channel](https://qiskit.slack.com) for discussion and
simple questions.
For questions that are more suited for a forum we use the Qiskit tag in the
[Stack Exchange](https://quantumcomputing.stackexchange.com/questions/tagged/qiskit).

## Authors and Citation

Qiskit Dynamics is the work of
[many people](https://github.com/Qiskit/qiskit-dynamics/graphs/contributors) who contribute
to the project at different levels. If you use Qiskit, please cite as per the included
[BibTeX file](https://github.com/Qiskit/qiskit-dynamics/blob/main/CITATION.bib).

## License

[Apache License 2.0](LICENSE.txt)
