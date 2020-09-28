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

import numpy as np
from typing import Callable, Union, List, Optional

from .DE_Methods import ODE_Method, method_from_string
from .DE_Problems import BMDE_Problem
from .DE_Options import DE_Options
from .type_utils import StateTypeConverter

class BMDE_Solver:
    """Solver class for differential equations specified in
    :class:`BMDE_Problem`. This class serves as an intermediary between the
    structures specified in :class:`BMDE_Problem`, and :class:`ODE_Method`
    instances, which wrap numerical DE solving methods (and are agnostic to
    the details of the computation of RHS functions).

    This class is meant to be interacted with in a similar way to
    :class:`ODE_Method`: the properties `t` and `y` are the time and state
    of the DE respectively, and the `integrate` and `integrate_over_interval`
    methods evolve the time and state.

    Details:
        - The state and time are fundamentally stored in the underlying
          :class:`ODE_Method`, and the properties in this class retrieve and
          set them. For the state this class provides a translation layer
          between the format and frame the user expects and the format
          and frame the DE is actually being solved in.
        - Continuing the above point, the translation consists of:
            - Deciding whether to translate states into/out of the frame
              depending on which frame the user is in.
            - Regardless of the previous point, the DE is always solved in a
              basis in which the frame is diagonal, so transforming into/out of
              this basis is always necessary when setting or getting the state.
            - Lastly, the state_type_converter in the bmde_problem may further
              transform the state, e.g. if the internally represented DE is
              the vectorized version of a DE the user is working with.
    """

    def __init__(self,
                 bmde_problem: BMDE_Problem,
                 method: Optional[ODE_Method] = None,
                 options: Optional[DE_Options] = None):
        """Construct a solver.

        Args:
            bmde_problem: Specification of the problem.
            method: Specification of the underlying solver method.
            options: Container class for options.
        """

        self._generator = bmde_problem._generator

        # Instantiate default options if none given
        if options is None:
            options = DE_Options()

        # if no method explicitly provided
        if method is None:
            method = options.method

        Method = None
        if isinstance(method, str):
            Method = method_from_string(method)
        elif issubclass(method, ODE_Method):
            Method = method

        # instantiate method object with minimal parameters, then populate
        t0 = bmde_problem.t0
        self._method = Method(t0, y0=None, rhs=None, options=options)

        # flag signifying whether the user is themselves working in the frame
        # or not
        self._user_in_frame = bmde_problem._user_in_frame

        # set the initial state - state_type_converter needs to be specified
        # first to enable automatic transformations
        self._state_type_converter = bmde_problem._state_type_converter
        if bmde_problem.y0 is not None:
            self.y = bmde_problem.y0

        # set RHS functions to evaluate in frame and frame basis
        rhs_dict = {'rhs': lambda t, y: self._generator.lmult(t, y,
                                                              in_frame_basis=True),
                    'generator': lambda t: self._generator.evaluate(t,
                                                                    in_frame_basis=True)}
        self._method.set_rhs(rhs_dict)

    @property
    def t(self) -> float:
        """Time stored in underlying method."""
        return self._method.t

    @t.setter
    def t(self, new_t: float):
        """Time stored in underlying method.

        Args:
            new_t: New time.
        """
        self._method.t = new_t

    @property
    def y(self):
        """Get y, returning in the user frame."""
        return self._get_y(return_in_frame=self._user_in_frame)

    @y.setter
    def y(self, new_y):
        """Set y, assuming it is specified in the user frame."""
        if new_y is not None:
            self._set_y(new_y, y_in_frame=self._user_in_frame)

    def _set_y(self, y: np.ndarray, y_in_frame: Optional[bool] = False):
        """Internal routine for setting state of the system.

        Steps:
            - convert y to inner type of state which the solver works with
            - apply frame transformation if necessary
            - apply basis transformation to frame basis

        Args:
            y: new state
            y_in_frame: whether or not y is specified in the rotating frame
        """

        # convert y into internal representation
        new_y = None
        if self._state_type_converter is None:
            new_y = y
        else:
            new_y = self._state_type_converter.outer_to_inner(y)

        if y_in_frame:
            # if y is already in the frame, just convert into frame basis
            new_y = self._generator.frame.state_into_frame_basis(new_y)
        else:
            # if y not in frame, convert it into frame and into frame basis
            new_y = self._generator.frame.state_into_frame(self.t,
                                                           new_y,
                                                           y_in_frame_basis=False,
                                                           return_in_frame_basis=True)

        # set the converted state into the internal method state
        self._method.y = new_y


    def _get_y(self, return_in_frame: Optional[bool] = False):
        """Intenral routine for returning the state.

        Steps:
            - take state out of frame basis
            - take state out of frame if necessary
            - convert to outer type

        Args:
            return_in_frame: whether or not to return in the solver frame
        """

        # method state is in frame and frame basis
        return_y = self._method.y

        if return_in_frame:
            # if state requested in frame, just take out of the frame basis
            return_y = self._generator.frame.state_out_of_frame_basis(return_y)
        else:
            # if state requested out of frame, map out of frame, requesting
            # result be returned out of the frame basis
            return_y = self._generator.frame.state_out_of_frame(self.t,
                                                                return_y,
                                                                y_in_frame_basis=True,
                                                                return_in_frame_basis=False)

        # convert to outer type
        if self._state_type_converter is None:
            return return_y
        else:
            return self._state_type_converter.inner_to_outer(return_y)

    def integrate(self, tf: float, **kwargs):
        """Evolve the DE forward up to a time tf, with additional keyword
        arguments for the underlying method.
        """
        self._method.integrate(tf, **kwargs)

    def integrate_over_interval(self, y0, interval, **kwargs):
        """Integrate over an interval=[t0,tf] with a given initial state,
        with additional keyword arguments for the underlying method.
        """
        self.t = interval[0]
        self.y = y0
        self.integrate(interval[1], **kwargs)
