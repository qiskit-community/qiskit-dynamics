# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Functionality for importing qiskit.pulse model string representation.
"""

from typing import Tuple, List, Optional
from collections import OrderedDict

import numpy as np

from qiskit import QiskitError
from qiskit_dynamics.dispatch import Array

from .string_model_parser_old.string_model_parser import HamiltonianParser


def parse_hamiltonian_dict(
    hamiltonian_dict: dict, subsystem_list: Optional[List[int]] = None
) -> Tuple[Array, Array, List[str]]:
    """Convert Hamiltonian string representation into concrete operators
    and an ordered list of channels corresponding to the operators.

    Args:
        hamiltonian_dict: Dictionary representation of Hamiltonian.
                            ********************************************************************************
                            document this - what's required, what's unsupported?
        subsystem_list: List of qubits to include in the model. If ``None`` all are kept.

    Returns:
        Concrete Array representation of model: An Array for the static hamiltonian,
        an Array for the list of operators with time-dependent coefficients, and a list
        of channel names giving the time-dependent coefficients.
    """

    # raise errors for invalid hamiltonian_dict
    ######################################################################################################
    # Should we leave these as is or change them?
    #####################################################################################################
    hamiltonian_pre_parse_exceptions(hamiltonian_dict)

    ###########################################################
    # construct intermediate representation, where operators are
    # constructed but coefficients are still in string format
    ###########################################################

    # get variables
    variables = OrderedDict()
    if "vars" in hamiltonian_dict:
        variables = OrderedDict(hamiltonian_dict["vars"])

    # Get qubit subspace dimensions
    if "qub" in hamiltonian_dict:
        if subsystem_list is None:
            subsystem_list = [int(qubit) for qubit in hamiltonian_dict["qub"]]
        else:
            # if user supplied, make a copy and sort it
            subsystem_list = subsystem_list.copy()
            subsystem_list.sort()

        # force keys in hamiltonian['qub'] to be ints
        qub_dict = {int(key): val for key, val in hamiltonian_dict["qub"].items()}

        subsystem_dims = {int(qubit): qub_dict[int(qubit)] for qubit in subsystem_list}
    else:
        subsystem_dims = {}

    # Get oscillator subspace dimensions
    ##################################################################################################
    # We don't support this, should we drop it? Need to see to what extent
    # this is required by the HamiltonianParser object
    ##################################################################################################
    if "osc" in hamiltonian_dict:
        oscillator_dims = {int(key): val for key, val in hamiltonian_dict["osc"].items()}
    else:
        oscillator_dims = {}

    # Parse the Hamiltonian
    system = HamiltonianParser(
        h_str=hamiltonian_dict["h_str"], dim_osc=oscillator_dims, dim_qub=subsystem_dims
    )
    system.parse(subsystem_list)
    system = system.compiled

    ########################################################################################################
    # Next, extract the channels from the system
    # One issue is this does allow for channel expressions like D0 * D0
    # Not sure if we should allow or care about this?
    ########################################################################################################

    # extract which channels are associated with which Hamiltonian terms
    # This code assumes there is at most one channel appearing in each term, and that it
    # appears at the end.
    channels = []
    for _, ham_str in system:
        chan_idx = None
        if 'U' in ham_str:
            chan_idx = ham_str.index('U')
        elif 'D' in ham_str:
            chan_idx = ham_str.index('D')

        if chan_idx is None:
            channels.append(None)
        else:
            channels.append(ham_str[chan_idx:])

    ####################################################################################################
    # Try to evaluate the coefficients
    # should this be done in some separate function to protect variable names?
    # any way around this exec stuff? What if they set a variable to the name of a variable in
    # this function?
    ####################################################################################################

    # this seems like it works
    # set up variables to use in exec computation: set channels to 1 and
    # include variables
    local_vars = {chan: 1.0 for chan in set(channels) if chan is not None}
    local_vars.update(variables)

    evaluated_ops = []
    for op, coeff in system:
        loc = {}
        exec("evaluated_coeff = %s" % coeff, globals(), local_vars)
        evaluated_ops.append(local_vars["evaluated_coeff"] * op)

    ###################################################################################################
    # Merge terms based on channel
    ###################################################################################################

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
            if channel in reduced_channels:
                hamiltonian_operators[reduced_channels.index(channel)] += op
            else:
                hamiltonian_operators.append(op)
                reduced_channels.append(channel)

    # sort channels/operators according to channel ordering
    reduced_channels, hamiltonian_operators = zip(*sorted(zip(reduced_channels, hamiltonian_operators)))

    return static_hamiltonian, list(hamiltonian_operators), list(reduced_channels)


def hamiltonian_pre_parse_exceptions(hamiltonian_dict: dict):
    """Raises exceptions for improperly formatted or unsupported elements of
    hamiltonian dict specification.

    Parameters:
        hamiltonian: Dictionary specification of hamiltonian.
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

    # verify that terms in h_str contain at most a single channel label, and that it
    # is the only thing appearing after ||
    for term in hamiltonian_dict["h_str"]:
        channel_count = term.count("D") + term.count("U")
        # verify at most one channel appears
        if channel_count > 1:
            raise QiskitError(
                """Hamiltonian term '{}' contains more than one channel label.
                                 Hamiltonian terms can only contain one channel label.""".format(
                    term
                )
            )

        # if no channels ensure doesn't contain divider ||
        elif channel_count == 0 and term.count("|") > 0:
            raise QiskitError(
                """Operator-channel divider '||' appears in term '{}' with no channels specified.""".format(
                    term
                )
            )

        # ensure formatted as aa||Dxx or aa||Uxx where aa specifies an operator and
        # xx is a string of digits
        elif channel_count == 1:
            malformed_text = """Term '{}' does not conform to required string format.
                                Channels 'D' or 'U' may only be specified in the format
                                'aa||Dxx' or 'aa||Uxx', where 'aa' specifies an operator
                                and 'xx' is a string of digits.""".format(
                term
            )
            if term.count("||") != 1 or term.count("|") > 2:
                raise QiskitError(malformed_text)
            else:
                # make sure after || is either U or D followed by a string of digits
                divider_idx = term.index("||")
                if (
                    (term[divider_idx + 2] not in ["D", "U"])
                    or (len(term) == divider_idx + 3)
                    or any([not a.isdigit() for a in term[divider_idx + 3 :]])
                ):
                    raise QiskitError(malformed_text)
