# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
import setuptools

requirements = [
    "numpy==2.3.3",
    "scipy==1.16.2",
    "matplotlib==3.10.6",
    "qiskit==2.2.1",
    "multiset==3.2.0",
    "sympy==1.14.0",
    "arraylias==0.1.1"
]

jax_extras = ['jax==0.7.2', 'jaxlib==0.7.2']

PACKAGES = setuptools.find_packages(exclude=['test*'])

version_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'qiskit_dynamics',
                 'VERSION.txt'))

with open(version_path, 'r') as fd:
    version = fd.read().rstrip()

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setuptools.setup(
    name="qiskit-dynamics",
    version=version,
    packages=PACKAGES,
    description="Qiskit ODE solver",
    long_description=README,
    long_description_content_type='text/markdown',
    url="https://github.com/Qiskit-Community/qiskit-dynamics",
    author="Qiskit Development Team",
    author_email="qiskit@us.ibm.com",
    license="Apache 2.0",
    classifiers=[
        "Environment :: Console",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering",
    ],
    keywords="qiskit sdk quantum",
    project_urls={
        "Bug Tracker": "https://github.com/Qiskit-Community/qiskit-dynamics/issues",
        "Source Code": "https://github.com/Qiskit-Community/qiskit-dynamics",
        "Documentation": "https://qiskit-community.github.io/qiskit-dynamics/",
    },
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.10",
    extras_require={
        "jax": jax_extras
    },
    zip_safe=False
)
