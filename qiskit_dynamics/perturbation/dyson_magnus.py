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

r"""
Core functionality for computing Dyson series [1] and Magnus expansion [2, 3] terms.
Specifically, Dyson-like terms are computed via the algorithm in [4], and
Dyson series and Magnus expansion terms are computed via the
method in [5].

References:
    1. F. Dyson, *The radiation theories of Tomonaga, Schwinger, and Feynman*,
       Phys. Rev. 75, 486-502
    2. W. Magnus, *On the exponential solution of differential equations*
       *for a linear operator*, Commun. Pure Appl. Math. 7, 649-73
    3. S. Blanes, F. Casas, J. Oteo, J. Ros, *The Magnus expansion and some*
       *of its applications*, Phys. Rep. 470, 151-238
    4. H. Haas, D. Puzzuoli, F. Zhang, D. Cory, *Engineering Effective Hamiltonians*,
       New J. Phys. 21, 103011 (2019).
    5. Forthcoming
"""

from typing import Optional, List, Callable, Tuple, Union

import numpy as np
from scipy.special import factorial

from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import

from multiset import Multiset

from qiskit_dynamics.array import Array
from qiskit_dynamics.solvers.solver_functions import solve_ode

from qiskit_dynamics.perturbation.custom_binary_op import (
    _compile_custom_operation_rule,
    _CustomMatmul,
)

from qiskit_dynamics.perturbation.multiset_utils import (
    _get_all_submultisets,
    _submultiset_filter,
    _submultisets_and_complements,
)

from .perturbation_data import PowerSeriesData, DysonLikeData

try:
    import jax.numpy as jnp
    from jax.lax import scan, switch
    from jax import vmap
except ImportError:
    pass


def _solve_lmde_dyson(
    perturbations: List[Callable],
    t_span: Array,
    dyson_terms: Union[List[List[int]], List[Multiset]],
    perturbation_labels: Optional[Union[List[List], List[Multiset]]] = None,
    generator: Optional[Callable] = None,
    y0: Optional[Array] = None,
    dyson_in_frame: Optional[bool] = True,
    dyson_like: Optional[bool] = False,
    integration_method: Optional[str] = "DOP853",
    t_eval: Optional[Array] = None,
    **kwargs,
) -> OdeResult:
    """Helper function for computing Dyson terms using methods in References [4, 5].
    See documentation for :meth:`solve_lmde_perturbation`.

    Args:
        perturbations: List of callable matrix functions to appear in Dyson terms.
        t_span: Integration limits.
        dyson_terms: Terms to compute.
        perturbation_labels: Ordering/specification of the elements of perturbations. Only used
                             for dyson_like==False.
        generator: Optional frame generator.
        y0: Optional initial state for frame generator LMDE.
        dyson_in_frame: Whether to return the Dyson terms in the frame of the
                        the frame generator.
        dyson_like: Compute either Dyson or Dyson-like terms.
        integration_method: Integration method.
        t_eval: Optional additional time points at which to return the solution.
        **kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult
    """
    mat_dim = perturbations[0](t_span[0]).shape[0]

    if generator is None:
        # pylint: disable=unused-argument
        def default_generator(t):
            return np.zeros((mat_dim, mat_dim), dtype=complex)

        generator = default_generator

    if y0 is None:
        y0 = np.eye(mat_dim, dtype=complex)

    # construct term list depending on whether computation is dyson or dyson_like
    complete_term_list = None
    if dyson_like:
        complete_term_list = _get_complete_dyson_like_indices(dyson_terms)
    else:
        complete_term_list = _get_all_submultisets(dyson_terms)

    dyson_rhs = _setup_dyson_rhs(
        generator,
        perturbations,
        complete_term_list,
        mat_dim,
        dyson_like=dyson_like,
        perturbation_labels=perturbation_labels,
    )

    # initial state
    y0 = np.append(
        np.expand_dims(y0, 0),
        np.zeros((len(complete_term_list), y0.shape[-2], y0.shape[-1]), dtype=complex),
        axis=0,
    )

    results = solve_ode(
        rhs=dyson_rhs, t_span=t_span, y0=y0, method=integration_method, t_eval=t_eval, **kwargs
    )

    # extract Dyson terms and the solution to the base LMDE
    results.y = results.y.transpose((1, 0, 2, 3))
    dyson_terms = results.y[1:]
    results.y = results.y[0]

    if dyson_in_frame:
        for idx, dyson_term in enumerate(dyson_terms):
            dyson_terms[idx] = np.linalg.solve(results.y, dyson_term)

    expansion_type = "dyson"
    DataClass = PowerSeriesData
    if dyson_like:
        expansion_type = "dyson_like"
        DataClass = DysonLikeData

    results.perturbation_data = DataClass(
        data=Array(dyson_terms),
        labels=complete_term_list,
        metadata={"expansion_type": expansion_type},
    )

    return results


def _solve_lmde_magnus(
    perturbations: List[Callable],
    t_span: Array,
    magnus_terms: List[Multiset],
    perturbation_labels: Optional[List[Multiset]] = None,
    generator: Optional[Callable] = None,
    y0: Optional[Array] = None,
    integration_method: Optional[str] = "DOP853",
    t_eval: Optional[Array] = None,
    **kwargs,
) -> OdeResult:
    """Helper function for computing Magnus terms using method in Reference [5].
    See documentaiton for :meth:`solve_lmde_perturbation`.

    Args:
        perturbations: List of callable matrix functions to appear in Dyson terms.
        t_span: Integration limits.
        magnus_terms: Terms to compute.
        perturbation_labels: Ordering/specification of the elements of perturbations.
        generator: Optional frame generator.
        y0: Optional initial state for frame generator LMDE.
        integration_method: Integration method.
        t_eval: Optional additional time points at which to return the solution.
        **kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult
    """

    # first compute Dyson terms
    results = _solve_lmde_dyson(
        perturbations,
        t_span,
        dyson_terms=magnus_terms,
        perturbation_labels=perturbation_labels,
        generator=generator,
        y0=y0,
        dyson_in_frame=True,
        dyson_like=False,
        integration_method=integration_method,
        t_eval=t_eval,
        **kwargs,
    )

    # compute Magnus terms from Dyson and update the results
    magnus_terms = _magnus_from_dyson(
        results.perturbation_data.labels, results.perturbation_data.data
    )
    results.perturbation_data.metadata = {"expansion_type": "magnus"}
    results.perturbation_data.data = Array(magnus_terms)

    return results


def _solve_lmde_dyson_jax(
    perturbations: List[Callable],
    t_span: Array,
    dyson_terms: Union[List[List[int]], List[Multiset]],
    perturbation_labels: Optional[Union[List[List], List[Multiset]]] = None,
    generator: Optional[Callable] = None,
    y0: Optional[Array] = None,
    dyson_in_frame: Optional[bool] = True,
    dyson_like: Optional[bool] = True,
    integration_method: Optional[str] = "jax_odeint",
    t_eval: Optional[Array] = None,
    **kwargs,
) -> OdeResult:
    """JAX version of ``_solve_lmde_dyson``.
    See documentation for :meth:`solve_lmde_perturbation`.

    Args:
        perturbations: List of callable matrix functions to appear in Dyson terms.
        t_span: Integration limits.
        dyson_terms: Terms to compute.
        perturbation_labels: Ordering/specification of the elements of perturbations. Only used if
                             dyson_like==False.
        generator: Optional frame generator.
        y0: Optional initial state for frame generator LMDE.
        dyson_in_frame: Whether to return the Dyson terms in the frame of the
                        the frame generator.
        dyson_like: Compute either Dyson or Dyson-like terms.
        integration_method: Integration method.
        t_eval: Optional additional time points at which to return the solution.
        **kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult
    """

    mat_dim = perturbations[0](t_span[0]).shape[0]

    if generator is None:
        # pylint: disable=unused-argument
        def default_generator(t):
            return jnp.zeros((mat_dim, mat_dim), dtype=complex)

        generator = default_generator

    if y0 is None:
        y0 = jnp.eye(mat_dim, dtype=complex)
    else:
        y0 = Array(y0).data

    # ensure perturbations and generator to return raw jax arrays
    def func_transform(f):
        def new_func(t):
            return Array(f(t), backend="jax").data

        return new_func

    generator = func_transform(generator)
    perturbations = [func_transform(a_func) for a_func in perturbations]

    # construct term list an RHS based on whether dyson or dyson-like
    complete_term_list = None
    if dyson_like:
        complete_term_list = _get_complete_dyson_like_indices(dyson_terms)
    else:
        complete_term_list = _get_all_submultisets(dyson_terms)

    dyson_rhs = _setup_dyson_rhs_jax(
        generator,
        perturbations,
        complete_term_list,
        dyson_like=dyson_like,
        perturbation_labels=perturbation_labels,
    )

    # initial state
    y0 = jnp.append(
        jnp.expand_dims(y0, 0),
        jnp.zeros((len(complete_term_list), y0.shape[-2], y0.shape[-1]), dtype=complex),
        axis=0,
    )

    results = solve_ode(
        rhs=dyson_rhs, t_span=t_span, y0=y0, method=integration_method, t_eval=t_eval, **kwargs
    )

    # extract Dyson terms and the solution to the base LMDE
    results.y = results.y.transpose((1, 0, 2, 3))
    dyson_terms = jnp.array(results.y[1:])
    results.y = jnp.array(results.y[0])

    if dyson_in_frame:
        dyson_terms = vmap(lambda x: jnp.linalg.solve(results.y, x))(dyson_terms)

    expansion_type = "dyson"
    DataClass = PowerSeriesData
    if dyson_like:
        expansion_type = "dyson_like"
        DataClass = DysonLikeData

    results.perturbation_data = DataClass(
        data=Array(dyson_terms, backend="jax"),
        labels=complete_term_list,
        metadata={"expansion_type": expansion_type},
    )

    return results


def _solve_lmde_magnus_jax(
    perturbations: List[Callable],
    t_span: Array,
    magnus_terms: List[Multiset],
    perturbation_labels: Optional[List[Multiset]] = None,
    generator: Optional[Callable] = None,
    y0: Optional[Array] = None,
    integration_method: Optional[str] = "DOP853",
    t_eval: Optional[Array] = None,
    **kwargs,
) -> OdeResult:
    """JAX version of ``_solve_lmde_magnus``.
    See documentation for :meth:`solve_lmde_perturbation`.

    Args:
        perturbations: List of callable matrix functions to appear in Dyson terms.
        t_span: Integration limits.
        magnus_terms: Terms to compute.
        perturbation_labels: Ordering/specification of the elements of perturbations.
        generator: Optional frame generator.
        y0: Optional initial state for frame generator LMDE.
        integration_method: Integration method.
        t_eval: Optional additional time points at which to return the solution.
        **kwargs: Additional arguments to pass to the solver.

    Returns:
        OdeResult
    """

    # first compute Dyson terms
    results = _solve_lmde_dyson_jax(
        perturbations,
        t_span,
        dyson_terms=magnus_terms,
        perturbation_labels=perturbation_labels,
        generator=generator,
        y0=y0,
        dyson_in_frame=True,
        dyson_like=False,
        integration_method=integration_method,
        t_eval=t_eval,
        **kwargs,
    )

    # compute Magnus terms from Dyson and update the results
    magnus_terms = _magnus_from_dyson_jax(
        results.perturbation_data.labels, Array(results.perturbation_data.data).data
    )
    results.perturbation_data.metadata = {"expansion_type": "magnus"}
    results.perturbation_data.data = Array(magnus_terms, backend="jax")

    return results


def _setup_dyson_rhs(
    generator: Callable,
    perturbations: List[Callable],
    oc_dyson_indices: List[Multiset],
    mat_dim: int,
    dyson_like: Optional[bool] = True,
    perturbation_labels: Optional[List[Multiset]] = None,
) -> Callable:
    """Construct the RHS function for propagating Dyson terms.

    Args:
        generator: The frame generator G.
        perturbations: List of matrix functions appearing in Dyson terms.
        oc_dyson_indices: Ordered complete list of Dyson terms to compute.
        mat_dim: Dimension of outputs of generator and functions in perturbations.
        dyson_like: Whether the computation is for Dyson or Dyson-like terms.
        perturbation_labels: List of lists specifying index information for perturbations.
                             Only used when dyson_like==False.

    Returns:
        Callable
    """
    lmult_rule = None
    perturbations_evaluation_order = None
    if dyson_like:
        generator_eval_indices = _required_dyson_generator_indices(oc_dyson_indices)
        perturbations_evaluation_order = [0] + [idx + 1 for idx in generator_eval_indices]
        lmult_rule = _get_dyson_like_lmult_rule(oc_dyson_indices, generator_eval_indices)
    else:
        # filter members of perturbations required for given list of dyson terms
        if perturbation_labels is None:
            perturbation_labels = [[idx] for idx in range(len(perturbations))]
        reduced_perturbation_labels = _submultiset_filter(perturbation_labels, oc_dyson_indices)
        perturbations_evaluation_order = [0] + [
            perturbation_labels.index(multiset) + 1 for multiset in reduced_perturbation_labels
        ]
        lmult_rule = _get_dyson_lmult_rule(oc_dyson_indices, reduced_perturbation_labels)

    custom_matmul = _CustomMatmul(lmult_rule, index_offset=1)

    perturbations_evaluate_len = len(perturbations_evaluation_order)
    new_list = [generator] + perturbations

    def gen_evaluator(t):
        mat = np.empty((perturbations_evaluate_len, mat_dim, mat_dim), dtype=complex)

        for idx, a_idx in enumerate(perturbations_evaluation_order):
            mat[idx] = new_list[a_idx](t)

        return mat

    def dyson_rhs(t, y):
        return custom_matmul(gen_evaluator(t), y)

    return dyson_rhs


def _setup_dyson_rhs_jax(
    generator: Callable,
    perturbations: List[Callable],
    oc_dyson_indices: List[Multiset],
    dyson_like: Optional[bool] = True,
    perturbation_labels: Optional[List[Multiset]] = None,
) -> Callable:
    """JAX version of _setup_dyson_rhs.

    Args:
        generator: The frame generator G.
        perturbations: List of matrix functions appearing in Dyson terms.
        oc_dyson_indices: Ordered complete list of Dyson terms to compute.
        dyson_like: Whether the computation is for Dyson or Dyson-like terms.
        perturbation_labels: List of lists specifying index information for perturbations.
                             Only used when dyson_like==False.

    Returns:
        Callable
    """

    lmult_rule = None
    perturbations_evaluation_order = None
    if dyson_like:
        generator_eval_indices = _required_dyson_generator_indices(oc_dyson_indices)
        perturbations_evaluation_order = [0] + [idx + 1 for idx in generator_eval_indices]
        lmult_rule = _get_dyson_like_lmult_rule(oc_dyson_indices, generator_eval_indices)
    else:
        # filter members of perturbations required for given list of dyson terms
        if perturbation_labels is None:
            perturbation_labels = [[idx] for idx in range(len(perturbations))]
        reduced_perturbation_labels = _submultiset_filter(perturbation_labels, oc_dyson_indices)
        perturbations_evaluation_order = [0] + [
            perturbation_labels.index(multiset) + 1 for multiset in reduced_perturbation_labels
        ]
        lmult_rule = _get_dyson_lmult_rule(oc_dyson_indices, reduced_perturbation_labels)

    custom_matmul = _CustomMatmul(lmult_rule, index_offset=1, backend="jax")

    perturbations_evaluation_order = jnp.array(perturbations_evaluation_order, dtype=int)

    new_list = [generator] + perturbations

    def single_eval(idx, t):
        return switch(idx, new_list, t)

    multiple_eval = vmap(single_eval, in_axes=(0, None))

    def dyson_rhs(t, y):
        return custom_matmul(multiple_eval(perturbations_evaluation_order, t), y)

    return dyson_rhs


def _required_dyson_generator_indices(complete_dyson_indices: List) -> List:
    """Given a complete list of dyson indices, determine which generator terms
    are actually required.
    """
    generator_indices = []
    for term in complete_dyson_indices:
        if term[0] not in generator_indices:
            generator_indices.append(term[0])

    generator_indices.sort()
    return generator_indices


def _get_dyson_like_lmult_rule(
    complete_dyson_indices: List[List[int]], generator_indices: List[List[int]]
) -> List:
    """Construct custom product rules, in the format required by ``custom_product``,
    for a given set of Dyson terms.

    Assumption: the supplied list is complete, i.e. if a term depends on other
    terms, then the terms it depends on are also in the list.

    Convention: G(t) is given the index -1 to preserve the indexing of perturbations.

    Args:
        complete_dyson_indices: Complete list of Dyson terms.
        generator_indices: List of required generator terms.

    Returns:
        List: lmult rule.
    """

    # construct multiplication rules
    lmult_rule = [(np.array([1.0]), np.array([[-1, -1]]))]

    for term_idx, term in enumerate(complete_dyson_indices):

        if len(term) == 1:
            l_idx = generator_indices.index(term[0])
            lmult_rule.append((np.array([1.0, 1.0]), np.array([[-1, term_idx], [l_idx, -1]])))
        else:
            # self multiplied by generator
            lmult_indices = [[-1, term_idx]]
            # the left index is the first entry in term
            # check if it is required before adding

            l_idx = generator_indices.index(term[0])
            r_idx = complete_dyson_indices.index(term[1:])
            lmult_indices.append([l_idx, r_idx])

            lmult_rule.append(
                (np.ones(len(lmult_indices), dtype=float), np.array(lmult_indices, dtype=int))
            )

    return lmult_rule


def _get_complete_dyson_like_indices(dyson_terms: List[List[int]]) -> List[List[int]]:
    """Given a list of Dyson terms to compute specified as lists of indices,
    recursively construct all other Dyson terms that need to be computed,
    returned as a list, ordered by increasing Dyson order, and
    in lexicographic order within an order.

    Args:
        dyson_terms: Terms to compute.

    Returns:
        list: List of all terms that need to be computed.
    """

    max_order = max(map(len, dyson_terms))
    term_dict = {k: [] for k in range(1, max_order + 1)}

    # first populate with requested terms
    for term in dyson_terms:
        order = len(term)
        if term not in term_dict[order]:
            term_dict[order].append(list(term))

    # loop through orders in reverse order
    for order in range(max_order, 1, -1):
        for term in term_dict[order]:
            term = list(term)
            if term[1:] not in term_dict[order - 1]:
                term_dict[order - 1].append(term[1:])

    ordered_term_list = []

    for order in range(1, max(term_dict.keys()) + 1):
        ordered_term_list += term_dict[order]

    # sort in terms of increasing length and lexicographic order
    ordered_term_list.sort(key=str)
    ordered_term_list.sort(key=len)

    return ordered_term_list


def _magnus_from_dyson(complete_index_multisets: List[Multiset], dyson_terms: np.array) -> np.array:
    """Compute magnus terms from dyson terms using the recursion
    relation presented in [5]. The term "Q Matrices" in helper functions refers to
    the matrices used in the recursion relation in [5].

    Args:
        complete_index_multisets: A complete and canonically ordered list of index Multisets.
        dyson_terms: Array of Dyson terms.

    Returns:
        np.array: The Magnus terms.
    """

    ordered_q_terms = _get_q_term_list(complete_index_multisets)
    start_idx, magnus_indices, stacked_q_update_rules = _q_recursive_compiled_rules(ordered_q_terms)

    # if all terms are first order, nothing needs to be done
    if start_idx == len(dyson_terms):
        return dyson_terms

    # initialize array of q matrices with dyson terms
    q_mat_shape = (len(ordered_q_terms) + 1,) + dyson_terms.shape[1:]
    q_mat = np.zeros(q_mat_shape, dtype=complex)
    # add an identity at the end
    q_mat[-1] = np.broadcast_to(np.eye(q_mat.shape[-1], dtype=complex), q_mat.shape[1:])
    q_mat[magnus_indices] = dyson_terms

    index_list = start_idx + np.arange(len(stacked_q_update_rules[0]))

    for rule_idx, mat_idx in enumerate(index_list):
        compiled_rule = (
            stacked_q_update_rules[0][rule_idx],
            (stacked_q_update_rules[1][0][rule_idx], stacked_q_update_rules[1][1][rule_idx]),
        )
        custom_matmul = _CustomMatmul(compiled_rule, operation_rule_compiled=True)
        q_mat[mat_idx] = custom_matmul(q_mat, q_mat)[0]

    return q_mat[magnus_indices]


def _magnus_from_dyson_jax(
    complete_index_multisets: List[Multiset], dyson_terms: np.array
) -> np.array:
    """JAX version of _magnus_from_dyson."""

    ordered_q_terms = _get_q_term_list(complete_index_multisets)
    start_idx, magnus_indices, stacked_q_update_rules = _q_recursive_compiled_rules(ordered_q_terms)

    # if all terms are first order, nothing needs to be done
    if start_idx == len(dyson_terms):
        return dyson_terms

    # initialize array of q matrices with dyson terms
    q_mat_shape = (len(ordered_q_terms) + 1,) + dyson_terms.shape[1:]
    q_init = jnp.zeros(q_mat_shape, dtype=complex)
    q_init = q_init.at[magnus_indices].set(dyson_terms)
    q_init = q_init.at[-1].set(
        jnp.broadcast_to(jnp.eye(q_init.shape[-1], dtype=complex), q_init.shape[1:])
    )

    index_list = start_idx + jnp.arange(len(stacked_q_update_rules[0]))

    def scan_fun(B, x):
        idx, compiled_rule = x
        custom_matmul = _CustomMatmul(compiled_rule, operation_rule_compiled=True)
        update = custom_matmul(B, B)[0]
        new_B = B.at[idx].set(update)
        return new_B, None

    q_mats = scan(scan_fun, init=q_init, xs=(index_list, stacked_q_update_rules))[0]

    return q_mats[magnus_indices]


def _q_recursive_compiled_rules(ordered_q_terms: List) -> Tuple[int, np.array, Tuple]:
    """Construct compiled custom product rules for recursive computation
    of Q matrices.

    Note: this function "stacks" the rules into a single tuple whose formatting
    is chosen to be usable with jax loop constructs.

    Args:
        ordered_q_terms: Ordered list of Q matrix specifications.

    Returns:
        start_idx: the index at q_terms need to start being updated
        magnus_indices: The locations in the q matrix list corresponding to Magnus terms.
        stacked_compiled_rules: stacked rules as per the above note
    """

    # create list of locations in q_term_list corresponding to Magnus terms
    # and find start index
    start_idx = 0
    magnus_indices = []
    for idx, q_term in enumerate(ordered_q_terms):
        if q_term[1] == 1:
            magnus_indices.append(idx)

        if len(q_term[0]) == 1:
            start_idx += 1

    magnus_indices = np.array(magnus_indices)

    # first, construct rules, and determine a maximum length
    max_unique_mults = 0
    max_linear_rule = 0
    rules = []
    for q_term in ordered_q_terms[start_idx:]:
        rule = _q_product_rule(q_term, ordered_q_terms)
        rules.append(rule)

        unique_mults, linear_rule = _compile_custom_operation_rule(rule)

        max_unique_mults = max(max_unique_mults, len(unique_mults))
        max_linear_rule = max(max_linear_rule, linear_rule[0].shape[1])

    stacked_unique_mults = []
    stacked_linear_rules = ([], [])
    for rule in rules:
        unique_mults, linear_rule = _compile_custom_operation_rule(
            rule, unique_evaluation_len=max_unique_mults, linear_combo_len=max_linear_rule
        )
        stacked_unique_mults.append(unique_mults)
        stacked_linear_rules[0].append(linear_rule[0])
        stacked_linear_rules[1].append(linear_rule[1])

    # convert to arrays and put into standard format
    stacked_unique_mults = np.array(stacked_unique_mults)
    stacked_linear_combo_rule = (
        np.array(stacked_linear_rules[0]),
        np.array(np.array(stacked_linear_rules[1])),
    )
    stacked_compiled_rules = (stacked_unique_mults, stacked_linear_combo_rule)

    return start_idx, magnus_indices, stacked_compiled_rules


def _q_product_rule(q_term: Tuple, oc_q_term_list: List[Tuple]) -> List:
    """Given a specification of a Q matrix and an ordered complete
    list of Q matrix specifications, constructs the recursion relation required to
    compute q_term, specified as a custom product rule for instantiating
    a CustomProduct.

    Note:
        - This assumes len(sym_index) > 1, as the purpose of this
          function is to apply the recursion rules, and no rule is required
          when len(sym_index) == 1.
        - This function also assumes that q_term, and oc_q_term_list are
          correctly formatted in terms of internal sorting.

    Args:
        q_term: Tuple with a Multiset and a product order (int)
        oc_q_term_list: Ordered complete list of a q terms.

    Returns:
        List
    """

    sym_index, q_term_order = q_term
    q_term_idx = oc_q_term_list.index(q_term)
    q_term_len = len(sym_index)

    if q_term_order == 1:
        # if the order is 1, it is just a linear combination of lower terms
        coeffs = np.append(1.0, -1 / factorial(range(2, q_term_len + 1), exact=True))

        products = [[len(oc_q_term_list), q_term_idx]]
        for prod_order in range(2, q_term_len + 1):
            products.append([len(oc_q_term_list), oc_q_term_list.index((sym_index, prod_order))])

        return [(coeffs, np.array(products))]
    else:

        # construct a list of products
        # need to consider all possible sub-multisets of the Multiset index in q_term
        products = []
        submultisets, complements = _submultisets_and_complements(
            sym_index, len(sym_index) - (q_term_order - 1) + 1
        )

        for subset, complement in zip(submultisets, complements):
            product = [
                oc_q_term_list.index((subset, 1)),
                oc_q_term_list.index((complement, q_term_order - 1)),
            ]
            if product not in products:
                products.append(product)

        coeffs = np.ones(len(products), dtype=float)
        return [(coeffs, np.array(products))]


def _get_q_term_list(complete_index_multisets: List[Multiset]) -> List:
    """Construct a specification of the recursive Q matrices
    required to compute all Magnus terms specified by
    ``complete_index_multisets``. Each Q matrix is specified as
    a 2-tuple with first entry a Multiset indexing the term,
    and second entry the product order of the Q matrix.

    Note: This function assumes ``complete_index_multisets`` are
    canonically ordered and correctly formatted. The output is then
    a canonical ordering of the Q matrices.

    Args:
        complete_index_multisets: Canonically ordered complete list of index Multisets.

    Returns:
        List: Q matrix specification
    """

    q_terms = []
    for term in complete_index_multisets:
        for order in range(len(term), 0, -1):
            q_terms.append((term, order))

    return q_terms


def _get_dyson_lmult_rule(
    complete_index_multisets: List[Multiset], perturbation_labels: Optional[List[Multiset]] = None
) -> List:
    """Given a complete list of index multisets, return
    the lmult rule in the format required for ``CustomProduct``.
    Note, the generator :math:`G(t)` is encoded as index ``-1``, as
    it will be prepended to the list of A matrices.

    While not required within the logic of this function, the input
    should be canonically ordered according to ``_get_all_submultisets``.

    Args:
        complete_index_multisets: List of complete multisets.
        perturbation_labels: List of index multisets describing perturbations.

    Returns:
        List: Left multiplication rule description.
    """

    # If perturbation_labels is not specified, use the elements of complete_index_multisets
    # of length 1
    if perturbation_labels is None:
        perturbation_labels = []
        for entry in complete_index_multisets:
            if len(entry) == 1:
                perturbation_labels.append(entry)
            else:
                break

    # construct multiplication rules
    lmult_rule = [(np.array([1.0]), np.array([[-1, -1]]))]

    for term_idx, term in enumerate(complete_index_multisets):

        if len(term) == 1:
            lmult_rule.append((np.array([1.0, 1.0]), np.array([[-1, term_idx], [term_idx, -1]])))
        else:
            # self multiplied by base generator
            lmult_indices = [[-1, term_idx]]

            for l_idx, l_term in enumerate(perturbation_labels):
                if l_term <= term:
                    if len(l_term) == len(term):
                        lmult_indices.append([l_idx, -1])
                    else:
                        r_term = term - l_term
                        r_idx = complete_index_multisets.index(r_term)
                        lmult_indices.append([l_idx, r_idx])

            lmult_rule.append(
                (np.ones(len(lmult_indices), dtype=float), np.array(lmult_indices, dtype=int))
            )

    return lmult_rule
