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

#from .string_model_parser_old.string_model_parser import HamiltonianParser
from .string_model_parser_object import HamiltonianParser


# valid channel characters
CHANNEL_CHARS = ['U', 'D', 'M', 'u', 'd', 'm']


def parse_hamiltonian_dict(
    hamiltonian_dict: dict, subsystem_list: Optional[List[int]] = None
) -> Tuple[Array, Array, List[str]]:
    """Convert Hamiltonian string representation into concrete operators
    and an ordered list of channels corresponding to the operators.

    hamiltonian_dict needs to have keys:
        - 'h_str': List strings giving terms in the Hamiltonian.
        - 'qub': Dictionary giving subsystem dimensions. Keys are subsystem labels,
                 values are their dimensions.
        - 'vars': Dictionary with variables appearing in the terms in the h_str dict,
                  keys are strings giving the variables, values are the values of the variables.

    Operators in h_str are specified with capital letters, and their interpretation depends
    on the dimension of the subsystem.

    Pauli operators are represented with capital X, Y, Z. If the dimension of the subsystem is
    2, then these are the standard pauli operators. If dim > 2, then we have the associations:
        - X = a + adagger
        - Y = -1j * (a - adagger)
        - Z = (identity - 2 * N)

    ##################################################################################################
    # Do we want the above definition of Z? It orders the energy of the states in reverse order,
    # might be confusing.
    ##################################################################################################

    For oscillator type operators:
       - O is the number operator for some reason
       - Can we do adagger?


    Also Sp, Sm - check out operator_from_string.
    N is going to the identity for some reason? Need to look into this.


    Channels:
    I've modified the parser so that it will only accept/understand channel specs of the form:
    'aa||Dxx' or 'aa||Uxx', where 'aa' is a valid specification of an operator,
    and 'xx' is a string consisting only of numbers. This format must be obeyed, if not an
    error will be raised (or should be raised, if I've written it correctly). Note that this
    explicitly excludes the possibility of having multiple channels appear, which may have
    been possible in the Aer pulse simulator. For now though we will enforce this limitation,
    have it be actually documented, and then maybe expand on it later.

    Update to the above:
    - Accepted channel characters are now: ['D', 'U', 'M', 'd', 'u', 'm'].
      String rep stuff seems to assume upper case, but pulse itself gives channel names as lower
      case, so it makes sense to suppor this.


    The output merges all static terms, or terms with the same channel, into a single
    matrix. It returns these with the channel names, which have been sorted in lexicographic
    order (and hte matrices corresponding to those channels are set in the same order).

    Args:
        hamiltonian_dict: Dictionary representation of Hamiltonian.
                            ********************************************************************************
                            document this - what's required, what's unsupported?
        subsystem_list: List of qubits to include in the model. If ``None`` all are kept.

    Returns:
        Concrete Array representation of model: An Array for the static hamiltonian,
        an Array for the list of operators with time-dependent coefficients, and a list
        of channel names giving the time-dependent coefficients. Channel names are given
        in lower case in alignment with channel names in pulse.
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

    # Parse the Hamiltonian
    system = HamiltonianParser(
        h_str=hamiltonian_dict["h_str"], subsystem_dims=subsystem_dims
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

        for c in CHANNEL_CHARS:
            if c in ham_str:
                chan_idx = ham_str.index(c)
                break

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

    # All channels in the reduced list are set to lower case
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
        channel_count = 0
        for c in CHANNEL_CHARS:
            channel_count += term.count(c)
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

        # ensure formatted as aa||Cxx
        elif channel_count == 1:
            malformed_text = """Term '{}' does not conform to required string format.
                                Channels may only be specified in the format
                                'aa||Cxx', where 'aa' specifies an operator,
                                C is a valid channel character,
                                and 'xx' is a string of digits.""".format(
                term
            )
            if term.count("||") != 1 or term.count("|") > 2:
                raise QiskitError(malformed_text)
            else:
                # make sure after || is either U or D followed by a string of digits
                divider_idx = term.index("||")
                if (
                    (term[divider_idx + 2] not in CHANNEL_CHARS)
                    or (len(term) == divider_idx + 3)
                    or any([not a.isdigit() for a in term[divider_idx + 3 :]])
                ):
                    raise QiskitError(malformed_text)
