# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module for higher level solving interfaces.
"""

from typing import Optional

from .DE_Problems import BMDE_Problem
from .DE_Options import DE_Options
from .DE_Solvers import BMDE_Solver

def solve(problem : BMDE_Problem, options: Optional[DE_Options] = None):
    """Solve a fully specified BMDE problem; i.e. a problem with a generator,
    an initial state, and a time interval set.

    Args:
        problem: Fully specified :class:`BMDE_Problem`
        options: Solver options.

    Returns:
        solution to the DE in format implied by the :class:`BMDE_Problem`
    """

    if problem.interval is None:
        raise Exception("solve requires problem specified over an interval.")

    # if options is None set to default
    if options is None:
        options = DE_Options()

    # construct solver, integrate to end of interval, return result
    solver = BMDE_Solver(problem, options)
    solver.integrate(problem.interval[-1])
    return solver.y
