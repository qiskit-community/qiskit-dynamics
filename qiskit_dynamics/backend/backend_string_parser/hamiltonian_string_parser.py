# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""
Functionality for importing qiskit.pulse model string representation.

This file is meant for internal use and may be changed at any point.
"""

from typing import Tuple, List, Optional
from collections import OrderedDict

# required for calls to exec
# pylint: disable=unused-import
import numpy as np

from qiskit import QiskitError

from .regex_parser import _regex_parser


# valid channel characters
CHANNEL_CHARS = ["U", "D", "M", "A", "u", "d", "m", "a"]


def parse_backend_hamiltonian_dict(
    hamiltonian_dict: dict, subsystem_list: Optional[List[int]] = None
) -> Tuple[np.ndarray, np.ndarray, List[str], dict]:
    r"""Convert Pulse backend Hamiltonian dictionary into concrete array format
    with an ordered list of corresponding channels.

    The Pulse backend Hamiltonian dictionary, ``hamiltonian_dict``, must have the
    following keys:

    * ``'h_str'``: List of Hamiltonian terms in string format (see below).
    * ``'qub'``: Dictionary giving subsystem dimensions. Keys are subsystem labels,
      values are their dimensions.
    * ``'vars'``: Dictionary whose keys are the variables appearing in the terms in
      the ``h_str`` list, and values being the numerical values of the variables.

    The optional argument ``subsystem_list`` specifies a subset of subsystems to keep when parsing.
    If ``None``, all subsystems are kept. If ``subsystem_list`` is specified, then terms
    including subsystems not in the list will be ignored.

    Entries in the list ``hamiltonian_dict['h_str']`` must be formatted as a product of
    constants (either numerical constants or variables in ``hamiltonian_dict['vars'].keys()``)
    with operators. Operators are indicated with a capital letters followed by an integer
    indicating the subsystem the operator acts on. Accepted operator strings are:

    * ``'X'``: If the target subsystem is two dimensional, the
      Pauli :math:`X` operator, and if greater than two dimensional, returns
      :math:`a + a^\dagger`, where :math:`a` and :math:`a^\dagger` are the
      annihiliation and creation operators, respectively.
    * ``'Y'``: If the target subsystem is two dimensional, the
      Pauli :math:`Y` operator, and if greater than two dimensional, returns
      :math:`-i(a - a^\dagger)`, where :math:`a` and :math:`a^\dagger` are the
      annihiliation and creation operators, respectively.
    * ``'Z'``: If the target subsystem is two dimensional, the
      Pauli :math:`Z` operator, and if greater than two dimensional, returns
      :math:`I - 2 * N`, where :math:`N` is the number operator.
    * ``'a'``, ``'A'``, or ``'Sm'``: If two dimensional, the sigma minus operator, and if greater,
      generalizes to the operator.
    * ``'C'``, or ``'Sp'``: If two dimensional, sigma plus operator, and if greater,
      generalizes to the creation operator.
    * ``'N'``, or ``'O'``: The number operator.
    * ``'I'``: The identity operator.

    In addition to the above, a term in ``hamiltonian_dict['h_str']`` can be associated with
    a channel by ending it with a string of the form ``'||Sxx'``, where ``S`` is a valid channel
    label, and ``'xx'`` is an integer. Accepted channel labels are:

    * ``'D'`` or ``'d'`` for drive channels.
    * ``'U'`` or ``'u'`` for control channels.
    * ``'M'`` or ``'m'`` for measurement channels.
    * ``'A'`` or ``'a'`` for acquire channels.

    Finally, summations of terms of the above form can be indicated in
    ``hamiltonian_dict['h_str']`` via strings with syntax ``'_SUM[i, lb, ub, aa||S{i}]'``,
    where:

    * ``i`` is the summation variable.
    * ``lb`` and ``ub`` are the summation endpoints (inclusive).
    * ``aa`` is a valid operator string, possibly including the string ``{i}`` to indicate
      operators acting on subsystem ``i``.
    * ``S{i}`` is the specification of a channel indexed by ``i``.


    For example, the following ``hamiltonian_dict`` specifies a single
    transmon with 4 levels:

    .. code-block:: python

        hamiltonian_dict = {
            "h_str": ["v*np.pi*O0", "alpha*np.pi*O0*O0", "r*np.pi*X0||D0"],
            "qub": {"0": 4},
            "vars": {"v": 2.1, "alpha": -0.33, "r": 0.02},
        }

    The following example specifies a two transmon system, with single system terms specified
    using the summation format:

    .. code-block:: python

        hamiltonian_dict = {
            "h_str": [
                "_SUM[i,0,1,wq{i}/2*(I{i}-Z{i})]",
                "_SUM[i,0,1,delta{i}/2*O{i}*O{i}]",
                "_SUM[i,0,1,-delta{i}/2*O{i}]",
                "_SUM[i,0,1,omegad{i}*X{i}||D{i}]",
                "jq0q1*Sp0*Sm1",
                "jq0q1*Sm0*Sp1",
                "omegad1*X0||U0",
                "omegad0*X1||U1"
            ],
            "qub": {"0": 4, "1": 4},
            "vars": {
                "delta0": -2.111793476400394,
                "delta1": -2.0894421352015744,
                "jq0q1": 0.010495754104003914,
                "omegad0": 0.9715458990879812,
                "omegad1": 0.9803812537440838,
                "wq0": 32.517894442809514,
                "wq1": 33.0948996120196,
            },
        }

    Args:
        hamiltonian_dict: Pulse backend Hamiltonian dictionary.
        subsystem_list: List of subsystems to include in the model. If ``None`` all are kept.

    Returns:
        Tuple: Model converted into concrete arrays - the static Hamiltonian,
        a list of Hamiltonians corresponding to different channels, a list of
        channel labels corresponding to the list of time-dependent Hamiltonians,
        and a dictionary with subsystem dimensions whose keys are the subsystem labels.
    """

    # raise errors for invalid hamiltonian_dict
    _hamiltonian_pre_parse_exceptions(hamiltonian_dict)

    # get variables
    variables = OrderedDict()
    if "vars" in hamiltonian_dict:
        variables = OrderedDict(hamiltonian_dict["vars"])

    # Get qubit subspace dimensions
    if subsystem_list is None:
        subsystem_list = [int(qubit) for qubit in hamiltonian_dict["qub"]]
    else:
        # if user supplied, make a copy and sort it
        subsystem_list = sorted(subsystem_list)

    # force keys in hamiltonian['qub'] to be ints
    qub_dict = {int(key): val for key, val in hamiltonian_dict["qub"].items()}

    subsystem_dims = {int(qubit): qub_dict[int(qubit)] for qubit in subsystem_list}

    # Parse the Hamiltonian
    system = _regex_parser(
        operator_str=hamiltonian_dict["h_str"],
        subsystem_dims=subsystem_dims,
        subsystem_list=subsystem_list,
    )

    # Extract which channels are associated with which Hamiltonian terms.
    # Assumes one channel appearing in each term appearing at the end.
    channels = []
    for _, ham_str in system:
        chan_idx = None

        for c in CHANNEL_CHARS:
            # if c in ham_str, and all characters after are digits, treat
            # as channel
            if c in ham_str:
                if all(a.isdigit() for a in ham_str[ham_str.index(c) + 1 :]):
                    chan_idx = ham_str.index(c)
                    break

        if chan_idx is None:
            channels.append(None)
        else:
            channels.append(ham_str[chan_idx:])

    # evaluate coefficients
    local_vars = {chan: 1.0 for chan in set(channels) if chan is not None}
    local_vars.update(variables)

    evaluated_ops = []
    for op, coeff in system:
        # pylint: disable=exec-used
        exec(f"evaluated_coeff = {coeff}", globals(), local_vars)
        evaluated_ops.append(local_vars["evaluated_coeff"] * op)

    # merge terms based on channel
    static_hamiltonian = None
    hamiltonian_operators = []
    reduced_channels = []

    for channel, op in zip(channels, evaluated_ops):
        # if None, add it to the static hamiltonian
        if channel is None:
            if static_hamiltonian is None:
                static_hamiltonian = op
            else:
                static_hamiltonian += op
        else:
            channel = channel.lower()
            if channel in reduced_channels:
                hamiltonian_operators[reduced_channels.index(channel)] += op
            else:
                hamiltonian_operators.append(op)
                reduced_channels.append(channel)

    # sort channels/operators according to channel ordering
    if len(reduced_channels) > 0:
        reduced_channels, hamiltonian_operators = zip(
            *sorted(zip(reduced_channels, hamiltonian_operators))
        )

    return static_hamiltonian, list(hamiltonian_operators), list(reduced_channels), subsystem_dims


def _hamiltonian_pre_parse_exceptions(hamiltonian_dict: dict):
    """Raises exceptions for improperly formatted or unsupported elements of
    hamiltonian dict specification.

    Parameters:
        hamiltonian_dict: Dictionary specification of hamiltonian.
    Returns:
    Raises:
        QiskitError: If some part of the Hamiltonian dictionary is unsupported or invalid.
    """

    ham_str = hamiltonian_dict.get("h_str", [])
    if ham_str in ([], [""]):
        raise QiskitError("Hamiltonian dict requires a non-empty 'h_str' entry.")

    if hamiltonian_dict.get("qub", {}) == {}:
        raise QiskitError(
            "Hamiltonian dict requires non-empty 'qub' entry with subsystem dimensions."
        )

    if hamiltonian_dict.get("osc", {}) != {}:
        raise QiskitError("Oscillator-type systems are not supported.")

    # verify that if terms in h_str have the divider ||, then the channels are in the valid format
    for term in hamiltonian_dict["h_str"]:
        malformed_text = f"""Term '{term}' does not conform to required string format.
                            Channels may only be specified in the format
                            'aa||Cxx', where 'aa' specifies an operator,
                            C is a valid channel character,
                            and 'xx' is a string of digits."""

        # if two vertical bars used together, check if channels in correct format
        if term.count("|") == 2 and term.count("||") == 1:
            # get the string reserved for channel
            channel_str = term[term.index("||") + 2 :]

            # if channel string is empty
            if len(channel_str) == 0:
                raise QiskitError(malformed_text)

            # if first entry in channel string isn't a valid channel character
            if channel_str[0] not in CHANNEL_CHARS:
                raise QiskitError(malformed_text)

            # Verify either that: all remaining characters are digits, or,
            # if term starts with _SUM[ and ends with ], all remaining characters
            # are either digits, or starts and ends with {}
            if term[-1] == "]" and len(term) > 5 and term[:5] == "_SUM[":
                # drop the closing ]
                channel_str = channel_str[:-1]

                # if channel string doesn't contain anything other than channel character
                if len(channel_str) == 1:
                    raise QiskitError(malformed_text)

                # if starts with opening bracket, verify that it ends with closing bracket
                if channel_str[1] == "{":
                    if not channel_str[-1] == "}":
                        raise QiskitError(malformed_text)
                # otherwise verify that the remainder of terms only contains digits
                elif any(not c.isdigit() for c in channel_str[1:]):
                    raise QiskitError(malformed_text)
            else:
                # if channel string doesn't contain anything other than channel character
                if len(channel_str) == 1:
                    raise QiskitError(malformed_text)

                if any(not c.isdigit() for c in channel_str[1:]):
                    raise QiskitError(malformed_text)

        # if bars present but not in correct format, raise error
        elif term.count("|") != 0:
            raise QiskitError(malformed_text)
