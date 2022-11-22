# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019, 2020, 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Legacy code for parsing operator strings.

This file is meant for internal use and may be changed at any point.
"""

import re
import copy
from typing import List, Dict, Tuple
from collections import namedtuple, OrderedDict

import numpy as np

from .operator_from_string import _operator_from_string


def _regex_parser(
    operator_str: List[str], subsystem_dims: Dict[int, int], subsystem_list: List[int]
) -> List[Tuple[np.array, str]]:
    """Function wrapper for regex parsing object.

    Args:
        operator_str: List of strings in accepted format as described in
                      string_model_parser.parse_hamiltonian_dict.
        subsystem_dims: Dictionary mapping subsystem labels to dimensions.
        subsystem_list: List of subsystems on which the operators are to be constructed.
    Returns:
        List of tuples containing pairs operators and their string coefficients.
    """

    return _HamiltonianParser(h_str=operator_str, subsystem_dims=subsystem_dims).parse(
        subsystem_list
    )


class _HamiltonianParser:
    """Legacy object for parsing string specifications of Hamiltonians."""

    Token = namedtuple("Token", ("type", "name"))

    str_elements = OrderedDict(
        QubOpr=re.compile(r"(?P<opr>O|Sp|Sm|X|Y|Z|I)(?P<idx>[0-9]+)"),
        PrjOpr=re.compile(r"P(?P<idx>[0-9]+),(?P<ket>[0-9]+),(?P<bra>[0-9]+)"),
        CavOpr=re.compile(r"(?P<opr>A|C|N)(?P<idx>[0-9]+)"),
        Func=re.compile(r"(?P<name>[a-z]+)\("),
        Ext=re.compile(r"\.(?P<name>dag)"),
        Var=re.compile(r"[a-z]+[0-9]*"),
        Num=re.compile(r"[0-9.]+"),
        MathOrd0=re.compile(r"[*/]"),
        MathOrd1=re.compile(r"[+-]"),
        BrkL=re.compile(r"\("),
        BrkR=re.compile(r"\)"),
    )

    def __init__(self, h_str, subsystem_dims):
        """Create new quantum operator generator

        Parameters:
            h_str (list): list of Hamiltonian string
            subsystem_dims (dict): dimension of subsystems
        """
        self.h_str = h_str
        self.subsystem_dims = {int(label): int(dim) for label, dim in subsystem_dims.items()}
        self.str2qopr = {}

    def parse(self, qubit_list=None):
        """Parse and generate Hamiltonian terms."""
        td_hams = []
        tc_hams = []

        # expand sum
        self._expand_sum()

        # convert to reverse Polish notation
        for ham in self.h_str:
            if len(re.findall(r"\|\|", ham)) > 1:
                raise Exception(f"Multiple time-dependent terms in {ham}")
            p_td = re.search(r"(?P<opr>[\S]+)\|\|(?P<ch>[\S]+)", ham)

            # find time-dependent term
            if p_td:
                coef, token = self._tokenizer(p_td.group("opr"), qubit_list)
                if token is None:
                    continue
                # combine coefficient to time-dependent term
                if coef:
                    td = "*".join([coef, p_td.group("ch")])
                else:
                    td = p_td.group("ch")
                token = self._shunting_yard(token)
                _td = self._token2qobj(token), td

                td_hams.append(_td)
            else:
                coef, token = self._tokenizer(ham, qubit_list)
                if token is None:
                    continue
                token = self._shunting_yard(token)

                if (coef == "") or (coef is None):
                    coef = "1."

                _tc = self._token2qobj(token), coef

                tc_hams.append(_tc)

        return tc_hams + td_hams

    def _expand_sum(self):
        """Takes a string-based Hamiltonian list and expands the _SUM action items out."""
        sum_str = re.compile(r"_SUM\[(?P<itr>[a-z]),(?P<l>[a-z\d{}+-]+),(?P<u>[a-z\d{}+-]+),")
        brk_str = re.compile(r"]")

        ham_list = copy.copy(self.h_str)
        ham_out = []

        while any(ham_list):
            ham = ham_list.pop(0)
            p_sums = list(sum_str.finditer(ham))
            p_brks = list(brk_str.finditer(ham))
            if len(p_sums) != len(p_brks):
                raise Exception(f"Missing correct number of brackets in {ham}")

            # find correct sum-bracket correspondence
            if any(p_sums) == 0:
                ham_out.append(ham)
            else:
                itr = p_sums[0].group("itr")
                _l = int(p_sums[0].group("l"))
                _u = int(p_sums[0].group("u"))
                for ii in range(len(p_sums) - 1):
                    if p_sums[ii + 1].end() > p_brks[ii].start():
                        break
                else:
                    ii = len(p_sums) - 1

                # substitute iterator value
                _temp = []
                for kk in range(_l, _u + 1):
                    trg_s = ham[p_sums[0].end() : p_brks[ii].start()]
                    # generate replacement pattern
                    pattern = {}
                    for p in re.finditer(r"\{(?P<op_str>[a-z0-9*/+-]+)\}", trg_s):
                        if p.group() not in pattern:
                            sub = parse_binop(p.group("op_str"), operands={itr: str(kk)})
                            if sub.isdecimal():
                                pattern[p.group()] = sub
                            else:
                                pattern[p.group()] = f"{{{sub}}}"
                    for key, val in pattern.items():
                        trg_s = trg_s.replace(key, val)
                    _temp.append(
                        "".join([ham[: p_sums[0].start()], trg_s, ham[p_brks[ii].end() :]])
                    )
                ham_list.extend(_temp)

        self.h_str = ham_out

        return ham_out

    def _tokenizer(self, op_str, qubit_list=None):
        """Convert string to token and coefficient
        Check if the index is in qubit_list
        """

        # generate token
        _op_str = copy.copy(op_str)
        token_list = []
        prev = "none"
        while any(_op_str):
            for key, parser in _HamiltonianParser.str_elements.items():
                p = parser.match(_op_str)
                if p:
                    # find quantum operators
                    if key in ["QubOpr", "CavOpr"]:
                        _key = key
                        _name = p.group()
                        if p.group() not in self.str2qopr:
                            idx = int(p.group("idx"))
                            if qubit_list is not None and idx not in qubit_list:
                                return 0, None
                            name = p.group("opr")
                            opr = _operator_from_string(name, idx, self.subsystem_dims)
                            self.str2qopr[p.group()] = opr
                    elif key == "PrjOpr":
                        _key = key
                        _name = p.group()
                        if p.group() not in self.str2qopr:
                            idx = int(p.group("idx"))
                            name = "P"
                            opr = _operator_from_string(name, idx, self.subsystem_dims)
                            self.str2qopr[p.group()] = opr
                    elif key in ["Func", "Ext"]:
                        _name = p.group("name")
                        _key = key
                    elif key == "MathOrd1":
                        _name = p.group()
                        if prev not in ["QubOpr", "PrjOpr", "CavOpr", "Var", "Num"]:
                            _key = "MathUnitary"
                        else:
                            _key = key
                    else:
                        _name = p.group()
                        _key = key
                    token_list.append(_HamiltonianParser.Token(_key, _name))
                    _op_str = _op_str[p.end() :]
                    prev = _key
                    break
            else:
                raise Exception(f"Invalid input string {op_str} is found")

        # split coefficient
        coef = ""
        if any(k.type == "Var" for k in token_list):
            for ii, _ in enumerate(token_list):
                if token_list[ii].name == "*":
                    if all(k.type != "Var" for k in token_list[ii + 1 :]):
                        coef = "".join([k.name for k in token_list[:ii]])
                        token_list = token_list[ii + 1 :]
                        break
            else:
                raise Exception(f"Invalid order of operators and coefficients in {op_str}")

        return coef, token_list

    def _shunting_yard(self, token_list):
        """Reformat token to reverse Polish notation"""
        stack = []
        queue = []
        while any(token_list):
            token = token_list.pop(0)
            if token.type in ["QubOpr", "PrjOpr", "CavOpr", "Num"]:
                queue.append(token)
            elif token.type in ["Func", "Ext"]:
                stack.append(token)
            elif token.type in ["MathUnitary", "MathOrd0", "MathOrd1"]:
                while stack and math_priority(token, stack[-1]):
                    queue.append(stack.pop(-1))
                stack.append(token)
            elif token.type in ["BrkL"]:
                stack.append(token)
            elif token.type in ["BrkR"]:
                while stack[-1].type not in ["BrkL", "Func"]:
                    queue.append(stack.pop(-1))
                    if not any(stack):
                        raise Exception("Missing correct number of brackets")
                pop = stack.pop(-1)
                if pop.type == "Func":
                    queue.append(pop)
            else:
                raise Exception(f"Invalid token {token.name} is found")

        while any(stack):
            queue.append(stack.pop(-1))

        return queue

    def _token2qobj(self, tokens):
        """Generate Hamiltonian term from tokens."""
        stack = []
        for token in tokens:
            if token.type in ["QubOpr", "PrjOpr", "CavOpr"]:
                stack.append(self.str2qopr[token.name])
            elif token.type == "Num":
                stack.append(float(token.name))
            elif token.type in ["MathUnitary"]:
                if token.name == "-":
                    stack.append(-stack.pop(-1))
            elif token.type in ["MathOrd0", "MathOrd1"]:
                op2 = stack.pop(-1)
                op1 = stack.pop(-1)
                if token.name == "+":
                    stack.append(op1 + op2)
                elif token.name == "-":
                    stack.append(op1 - op2)
                elif token.name == "*":
                    stack.append(op1 @ op2)
                elif token.name == "/":
                    stack.append(op1 / op2)
            elif token.type in ["Func", "Ext"]:
                if token.name == "dag":
                    stack.append(np.conjugate(np.transpose(stack.pop(-1))))
                else:
                    raise Exception(f"Invalid token {token.name} of type Func, Ext.")
            else:
                raise Exception(f"Invalid token {token.name} is found.")

        if len(stack) > 1:
            raise Exception("Invalid mathematical operation in string.")

        return stack[0]


def math_priority(o1, o2):
    """Check priority of given math operation"""
    rank = {"MathUnitary": 2, "MathOrd0": 1, "MathOrd1": 0}
    diff_ops = rank.get(o1.type, -1) - rank.get(o2.type, -1)

    if diff_ops > 0:
        return False
    else:
        return True


# pylint: disable=dangerous-default-value
def parse_binop(op_str, operands={}, cast_str=True):
    """Calculate binary operation in string format"""
    oprs = OrderedDict(
        sum=r"(?P<v0>[a-zA-Z0-9]+)\+(?P<v1>[a-zA-Z0-9]+)",
        sub=r"(?P<v0>[a-zA-Z0-9]+)\-(?P<v1>[a-zA-Z0-9]+)",
        mul=r"(?P<v0>[a-zA-Z0-9]+)\*(?P<v1>[a-zA-Z0-9]+)",
        div=r"(?P<v0>[a-zA-Z0-9]+)\/(?P<v1>[a-zA-Z0-9]+)",
        non=r"(?P<v0>[a-zA-Z0-9]+)",
    )

    for key, regr in oprs.items():
        p = re.match(regr, op_str)
        if p:
            val0 = operands.get(p.group("v0"), p.group("v0"))
            if key == "non":
                # substitution
                retv = val0
            else:
                val1 = operands.get(p.group("v1"), p.group("v1"))
                # binary operation
                if key == "sum":
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) + int(val1)
                    else:
                        retv = "+".join([str(val0), str(val1)])
                elif key == "sub":
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) - int(val1)
                    else:
                        retv = "-".join([str(val0), str(val1)])
                elif key == "mul":
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) * int(val1)
                    else:
                        retv = "*".join([str(val0), str(val1)])
                elif key == "div":
                    if val0.isdecimal() and val1.isdecimal():
                        retv = int(val0) / int(val1)
                    else:
                        retv = "/".join([str(val0), str(val1)])
                else:
                    retv = 0
            break
    else:
        raise Exception(f"Invalid string {op_str}")

    if cast_str:
        return str(retv)
    else:
        return retv
