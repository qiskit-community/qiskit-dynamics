# -*- coding: utf-8 -*-

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
Compute perturbation theory terms for an LMDE.
"""

from typing import List, Optional, Callable
from itertools import combinations_with_replacement, product

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit_dynamics import solve_ode
from qiskit_dynamics.array import Array
from qiskit_dynamics.perturbation.power_series_utils import clean_index_multisets
from qiskit_dynamics.perturbation.custom_dot import (
    compile_custom_dot_rule,
    custom_dot,
    custom_dot_jax,
)

from qiskit_dynamics.perturbation.dyson_magnus import (
    solve_lmde_dyson,
    solve_lmde_symmetric_magnus,
    solve_lmde_dyson_jax,
    solve_lmde_symmetric_magnus_jax,
)

try:
    import jax.numpy as jnp
    from jax import vmap
    from jax.lax import switch
except ImportError:
    pass


def solve_lmde_perturbation(
    A_list: List[Callable],
    t_span: Array,
    perturbation_method: str,
    perturbation_order: Optional[int] = None,
    perturbation_terms: Optional[List] = None,
    A_list_indices: Optional[List[List[int]]] = None,
    generator: Optional[Callable] = None,
    y0: Optional[Array] = None,
    dyson_in_frame: Optional[bool] = True,
    method: Optional[str] = "DOP853",
    t_eval: Optional[Array] = None,
    **kwargs,
) -> OdeResult:
    r"""Given a list of matrix functions :math:`A_0(t), \dots, A_{r-1}(t)`, compute
    either Dyson series [1], symmetric Dyson series [5], or symmetric Magnus expansion [2,3,5]
    terms. Which expansion is used is controlled by the ``perturbation_method`` argument.

    If ``perturbation_method == 'dyson'``, for a list of indices :math:`i_1, \dots, i_k`
    this function computes nested integrals of the form

    .. math::
        \int_{T_0}^{T_F} dt_1 \int_{T_0}^{t_1} dt_2 \dots \int_{T_0}^{t_{k-1}}dt_k
                A_{i_1}(t_1) \dots A_{i_k}(t_k).

    A list of specific integrals to compute can be specified as lists of ``int`` s via the
    ``perturbation_terms`` argument. Alternatively, the ``perturbation_order`` argument
    may be used to compute all terms up to a given order. Both arguments can be used
    simultaneously to specify computation of all terms up to a given order, along with
    some other higher order terms.

    If ``perturbation_method == 'symmetric_dyson'``, for a list of indices,
    :math:`I = (i_1, \dots, i_k)`, this function computes the expression

    .. math::
        \sum_{\sigma \in S(I)} \int_{T_0}^{T_F} dt_1 \int_{T_0}^{t_1} dt_2 \dots
                \int_{T_0}^{t_{k-1}}dt_k A_{\sigma_1}(t_1) \dots A_{\sigma_k}(t_k)

    where :math:`S(I)` is the set of all permutations of :math:`I`. The ``perturbation_order``
    and ``perturbation_terms`` arguments behave similarly to the
    ``perturbation_method == 'dyson'`` case, however due to the symmetrization in the above
    definition, the index lists :math:`I` are treated as multisets rather than ordered lists.
    E.g. the index lists ``[0, 1]`` and ``[1, 0]`` define the same integral above.
    As such we choose a canonical ordering of these lists to be non-decreasing.

    If ``perturbation_method == 'symmetric_magnus'``, this functions computes the
    implicitly defined symmetric Magnus terms in [5]. The handling of index lists is the
    same as that of the ``'symmetric_dyson'`` case.

    For both ``perturbation_method == 'symmetric_dyson'`` and
    ``perturbation_method == 'symmetric_magnus'`` cases, the optional argument ``A_list_indices``
    can be used to augment the meaning of the terms in ``A_list``. If ``A_list_indices`` is
    used, it must satisfy ``len(A_list_indices) == len(A_list)``, and each entry of
    ``A_list_indices`` must be a list of ``int``s representing multisets of indices, formatted
    in the same manner as required by ``perturbation_terms`` in the symmetric case. If this
    is used, then in the Dyson case, the results object,
    for a list of indices :math:`I = (i_1, \dots, i_k)`,
    will contain the integral

    .. math::
        \sum_{m=1}^{|I|}\sum_{(I_1, \dots, I_m) \in P_m(I)}
        \int_{T_0}^{T_F} dt_1 \int_{T_0}^{t_1} dt_2 \dots
                \int_{T_0}^{t_{m-1}}dt_m A_{I_1}(t_1) \dots A_{I_m}(t_m)

    where :math:`P_m(I)` is the set of ordered partitions of the multiset :math:`I`. As described
    in [5], this computation corresponds to computing a power series decomposition of
    the truncated Dyson series, given a polynomial decomposition of a generator. The
    symmetric Magnus case is analagous.

    Notes on computational methods:

        - The computational methods of [4, 5] are based on constructing a single large
          differential equation to compute all requested terms simultaneously. In this
          construction, a given term requires computation of a collection of lower order
          terms. For example, computing the 3rd order Dyson term for index list
          ``[0, 0, 0]`` requires also computing the 2nd order term ``[0, 0]`` and the
          1st order term ``[0]``. As such, even if only the single term ``[0, 0, 0]``
          is requested, the output of this function will contain the results of the
          computations as if the user had specified ``[[0], [0, 0], [0, 0, 0]]``.
        - As the core computation is performed via solving a differential equation,
          this function sets up the RHS function, then passes this to
          :meth:`~qiskit_dynamics.solve.solve_ode`. As such, all optional arguments and
          methods of :meth:`~qiskit_dynamics.solve.solve_ode` can be passed
          to this function.

    More generally, this function accepts the optional specification of an LMDE
    :math:`\dot{y}(t) = G(t)y(t)`, where :math:`G(t)` is passed via the ``generator``
    argument, and the initial condition via ``y0`` (which must be square). If these
    arguments are passed, this LMDE will be simultaneously solved with the
    perturbation terms, and will modify the perturbation calculations to compute
    the terms for :math:`\tilde{A}_i(t) = U^{-1}(t) A_i(t) U(t)`. That is, the
    perturbation theory terms will be computed in the time-dependent frame of :math:`G(t)`.
    Further notes on ``generator`` and ``y0``:

        - ``generator`` defaults to the constant 0 matrix, and ``y0`` defaults the identity.
        - For ``'dyson'`` and ``'symmetric_dyson'`` computations, the computational
          method actually returns the above quoted matrices with a prefactor of :math:`U(t)`,
          which needs to be removed. Setting the optional argument ``dyson_in_frame=False``
          skips this removal step.

    Results are returned in an ``OdeResult`` in the same manner as
    :meth:`~qiskit_dynamics.solve.solve_ode`. The result object stores the results
    of the LMDE for ``generator`` and ``y0`` in the ``y`` attribute as in
    :meth:`~qiskit_dynamics.solve.solve_ode` before, and the perturbation
    results are in the ``perturbation_results`` attribute storing a
    ``PerturbationResults`` object, which is a data container with attributes:

        - ``perturbation_method``: Method as specified by the user.
        - ``term_labels``: Index labels for all computed perturbation terms.
        - ``perturbation_terms``: A 4d array storing all computed terms. The first axis indexes
                                  the perturbation terms in the same ordering as ``term_labels``,
                                  and the second axis indexes the perturbation terms evaluated
                                  at the times in ``results.t`` in the same manner as
                                  ``results.y``.

    Additionally, to retrieve the term with a given label, the ``PerturbationResults`` object
    can be subscripted, e.g. the results for the computation for the term with indices
    ``[0, 1]`` is retrievable via ``results.perturbation_terms[[0, 1]]``.

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

    Args:
        A_list: List of matrix-valued callables.
        t_span: Integration bounds.
        perturbation_method: Either ``'dyson'``, ``'symmetric_dyson'``, or ``'symmetric_magnus'``.
        perturbation_order: Order of perturbation terms to compute up to. Specifying this
                            argument results in computation of all terms up to the given order.
                            Can be used in conjunction with ``perturbation_terms``.
        perturbation_terms: Specific perturbation terms to compute. If both ``perturbation_order``
                            and ``perturbation_terms`` are specified, then all terms up to
                            ``perturbation_order`` are computed, along with the additional terms
                            specified in ``perturbation_terms``.
        A_list_indices: Optional description of power series terms specified by A_list. To only
                        be used with ``'symmetric_dyson'`` and ``'symmetric_magnus'`` methods.
        generator: Optional frame generator.
        y0: Optional initial state for frame generator LMDE.
        dyson_in_frame: For ``perturbation_method`` ``'dyson'`` or ``'symmetric_dyson'``,
                        whether or not to remove the frame transformation pre-factor from the
                        Dyson terms.
        method: Integration method to use.
        t_eval: Points at which to evaluate the system.
        kwargs: Additional arguments to pass to solver method used to compute terms.

    Returns:
        OdeResult: Results object.

    Raises:
        QiskitError: If problem with inputs, either ``perturbation_method`` is unsupported,
                     or both of ``perturbation_order`` and ``perturbation_terms`` unspecified.
    """

    if (perturbation_order is None) and (perturbation_terms is None):
        raise QiskitError(
            """Must specify one of perturbation_order or
                          perturbation_terms when calling solve_lmde_perturbation."""
        )

    if y0 is not None:
        if len(y0.shape) != 2 or y0.shape[0] != y0.shape[1]:
            raise QiskitError("""If used, optional arg y0 must be a square 2d array.""")

    # determine whether to use jax looping logic
    use_jax = True
    for A_func in A_list:
        if Array(A_func(t_span[0])).backend != "jax":
            use_jax = False
            break

    if use_jax and generator is not None:
        if Array(generator(t_span[0])).backend != "jax":
            use_jax = False

    # clean and validate A_list_indices
    if A_list_indices is not None:
        if perturbation_method == "dyson":
            raise QiskitError(
                "A_list_indices argument not usable with perturbation_method='dyson'."
            )

        # validate A_list_indices
        A_list_len = len(A_list_indices)
        A_list_indices = clean_index_multisets(A_list_indices)
        if len(A_list_indices) != A_list_len:
            raise QiskitError("A_list_indices argument contains duplicates as multisets.")
    else:
        A_list_indices = [[idx] for idx in range(len(A_list))]

    # merge perturbation_order and perturbation_terms args
    perturbation_terms = merge_perturbation_order_terms(
        perturbation_order, perturbation_terms, A_list_indices, "symmetric" in perturbation_method
    )

    if perturbation_method in ["dyson", "symmetric_dyson"]:
        symmetric = perturbation_method == "symmetric_dyson"
        if not use_jax:
            return solve_lmde_dyson(
                A_list=A_list,
                t_span=t_span,
                dyson_terms=perturbation_terms,
                A_list_indices=A_list_indices,
                generator=generator,
                y0=y0,
                dyson_in_frame=dyson_in_frame,
                symmetric=symmetric,
                method=method,
                t_eval=t_eval,
                **kwargs,
            )
        else:
            return solve_lmde_dyson_jax(
                A_list=A_list,
                t_span=t_span,
                dyson_terms=perturbation_terms,
                A_list_indices=A_list_indices,
                generator=generator,
                y0=y0,
                dyson_in_frame=dyson_in_frame,
                symmetric=symmetric,
                method=method,
                t_eval=t_eval,
                **kwargs,
            )
    elif perturbation_method == "symmetric_magnus":
        if not use_jax:
            return solve_lmde_symmetric_magnus(
                A_list=A_list,
                t_span=t_span,
                magnus_terms=perturbation_terms,
                A_list_indices=A_list_indices,
                generator=generator,
                y0=y0,
                method=method,
                t_eval=t_eval,
                **kwargs,
            )
        else:
            return solve_lmde_symmetric_magnus_jax(
                A_list=A_list,
                t_span=t_span,
                magnus_terms=perturbation_terms,
                A_list_indices=A_list_indices,
                generator=generator,
                y0=y0,
                method=method,
                t_eval=t_eval,
                **kwargs,
            )

    # raise error if none apply
    raise QiskitError("perturbation_method " + str(perturbation_method) + " not supported.")


def merge_perturbation_order_terms(
    perturbation_order: int, perturbation_terms: List, A_list_indices: List[int], symmetric: bool
) -> List:
    """Combine ``perturbation_order`` and ``perturbation_terms`` into a single
    explicit list of perturbation terms to compute. It is assumed that at least
    one of the two arguments is in correct format.

    Note that this function generates a minimal list of term labels sufficient to
    generate all required terms when the list is 'completed'. E.g. for order=3,
    it is sufficient here to only return all third order terms, as ``solve_lmde_perturbation``
    will 'complete' this list and add all lower order terms required for computing
    the specified third order terms.

    Args:
        perturbation_order: Order of expansion to compute all terms up to.
        perturbation_terms: Specific individual terms requested to compute.
        A_list_indices: Labels for perturbations.
        symmetric: Whether or not the perturbation terms represent symmetric or non-symmetric
                   expansions.
    Returns:
        List of perturbation terms to compute based on merging of perturbation_order and
        perturbation_terms.
    """

    # determine unique indices in A_list_indices
    unique_indices = []
    for multiset in A_list_indices:
        for idx in multiset:
            if idx not in unique_indices:
                unique_indices.append(idx)

    perturbation_terms = perturbation_terms or []
    if perturbation_order is not None:
        up_to_order_terms = None
        if symmetric:
            up_to_order_terms = list(
                map(list, combinations_with_replacement(unique_indices, perturbation_order))
            )
        else:
            up_to_order_terms = list(map(list, product(unique_indices, repeat=perturbation_order)))
        perturbation_terms = perturbation_terms + up_to_order_terms

    return perturbation_terms
