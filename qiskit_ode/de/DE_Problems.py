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
from typing import Union, List, Optional
from warnings import warn

from qiskit.quantum_info.operators import Operator
from qiskit.providers.aer.pulse_new.models.frame import BaseFrame
from qiskit.providers.aer.pulse_new.models.operator_models import BaseOperatorModel
from qiskit.providers.aer.pulse_new.de.type_utils import StateTypeConverter

class BMDE_Problem:
    """Class for representing Bilinear Matrix Differential Equations (BMDEs),
    which are differential equations of the form:

    .. math::

        \dot{y}(t) = G(t)y(t)

    with initial condition :math:`y(0) = ` `y0`. This class is primarly a
    data container, with additional functionality for setting up the BMDE
    to be solved.

    This class specifies a BMDE in terms of:
        - The `generator` :math:`G(t)`, expected to be a concrete instance
          of :class:`BaseOperatorModel`.
        - The initial state `y0`.
        - Either the initial time `t0`, or an interval `[t0, tf]`.
        - A frame specifying the frame to solve in, expected to be a concrete
          instance of :class:`BaseFrame`, a valid argument to instantiate the
          subclass of :class:`BaseFrame` that the `generator` works with,
          the string `'auto'`, indicating an automatic choice, or `None`,
          indicating that the system should be solved in as is.
        - A `cutoff_freq`, indicating that the BMDE should be solved with
          a rotating wave approximation.
        - A :class:`StateTypeConverter` object, which is only of relevance if
          some transformation was required to get the BMDE in standard form.

    Some special behaviour:
        - If the `generator` already has a frame specified, the BMDE will be
          solved in that frame regardless of the frame specified when
          constructing this class. Additionally, in this case, when solved,
          the results will be returned in the rotating frame.
        - If a cutoff frequency is specified in both the `generator` and
          as an argument to this class, an `Exception` is raised.
    """

    def __init__(self,
                 generator: BaseOperatorModel,
                 y0: Optional[np.ndarray] = None,
                 t0: Optional[float] = None,
                 interval: Optional[List[float]] = None,
                 frame: Optional[Union[str, Operator, np.ndarray, BaseFrame]] = 'auto',
                 cutoff_freq: Optional[float] = None,
                 state_type_converter: Optional[StateTypeConverter] = None):
        """Specify a BMDE problem.

        Args:
            generator: The generator.
            y0: The initial state of the BMDE.
            t0: Initial time.
            interval: The interval of the BMDE
                      (specify only one of t0 or interval).
            frame: The frame to solve the system in. If the generator already
                   has a frame specified, the problem will default to using
                   that frame and returning results in that frame.
            cutoff_freq: Cutoff frequency to use when solving the BMDE.
            state_type_converter: Optional setting for translating between
                                  the state type the user interacts with
                                  to the state type the solver needs.
        """

        # set state and time parameters
        self.y0 = y0

        if (interval is not None) and (t0 is not None):
            raise Exception('Specify only one of t0 or interval.')

        self.interval = interval
        if interval is not None:
            self.t0 = self.interval[0]
        else:
            self.t0 = t0

        self._state_type_converter = state_type_converter

        # copy the generator to preserve state of user's generator
        self._generator = generator.copy()

        # set up frame
        if self._generator.frame.frame_operator is not None:
            # if the generator has a frame specified, leave it as,
            # and specify that the user is in the frame
            self._user_in_frame = True

            if frame != 'auto':
                warn("""A frame was specified in both the generator and in the
                        BMDE problem. Defaulting to use the generator frame
                        and return results in that frame.""")
        else:
            # if auto, go into the drift part of the generator, otherwise
            # set it to whatever is passed
            if frame == 'auto':
                self._generator.frame = anti_herm_part(generator.drift)
            else:
                self._generator.frame = frame

            self._user_in_frame = False

        # set up cutoff frequency
        if self._generator.cutoff_freq is not None and cutoff_freq is not None:
            raise Exception("""Cutoff frequency specified in generator and in
                                solver settings.""")

        if cutoff_freq is not None:
            self._generator.cutoff_freq = cutoff_freq


def anti_herm_part(A: Union[np.ndarray, Operator]):
    """Get the anti-hermitian part of an operator.
    """
    return 0.5 * (A - A.conj().transpose())
