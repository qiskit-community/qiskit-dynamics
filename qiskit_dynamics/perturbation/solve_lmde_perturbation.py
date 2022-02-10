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

# pylint: disable=unused-import
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit_dynamics import solve_ode
from qiskit_dynamics.array import Array
from qiskit_dynamics.perturbation.multiset import Multiset, clean_multisets
from qiskit_dynamics.perturbation.perturbation_utils import (
    merge_multiset_expansion_order_labels,
    merge_list_expansion_order_labels,
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
    perturbations: List[Callable],
    t_span: Array,
    expansion_method: str,
    expansion_order: Optional[int] = None,
    expansion_labels: Optional[List[Multiset]] = None,
    perturbation_labels: Optional[List[Multiset]] = None,
    generator: Optional[Callable] = None,
    y0: Optional[Array] = None,
    dyson_in_frame: Optional[bool] = True,
    integration_method: Optional[str] = "DOP853",
    t_eval: Optional[Array] = None,
    **kwargs,
) -> OdeResult:
    r"""Compute time-dependent perturbation theory terms for an LMDE.

    This function computes symmetric Dyson or Magnus expansion terms
    via the algorithm in [forthcoming], or Dyson-like terms via the algorithm in
    [:footcite:`haas_engineering_2019`]. Which expansion is used is controlled by
    the ``expansion_method`` argument, which impacts the interpretation of
    several of the function arguments (described below).

    Regardless of ``expansion_method``, the main computation is performed by
    solving a differential equation, utilizing :func:`~qiskit_dynamics.solvers.solve_ode`,
    and as such several of the function arguments are direct inputs into this function:

        - ``integration_method`` is the ODE method used (passed as ``method``
          to :func:`~qiskit_dynamics.solvers.solve_ode`),
        - ``t_span`` is the integration interval, and
        - ``t_eval`` is an optional set of points to evaluate the perturbation terms at.
        - ``kwargs`` are passed directly to :func:`~qiskit_dynamics.solvers.solve_ode`, enabling
          passing through of tolerance or step size arguments.


    If ``expansion_method == 'symmetric_dyson'`` or ``expansion_method == 'symmetric_magnus'``,
    the computation is performed as described in the
    :ref:`time-dependent perturbation theory section <td perturbation theory>`
    of the perturbation API doc. In this case:

        - ``perturbations`` gives a list of the :math:`A_I` functions as callables.
        - ``perturbation_labels`` is an optional list specifying the labels for the terms in
          ``perturbations`` in the form of
          :class:`~qiskit_dynamics.perturbation.multiset.Multiset`\s.
          If not specified, the labels are assumed to be
          ``[Multiset({0: 1}), ..., Multiset({len(perturbations) - 1: 1})]``.
        - ``expansion_order`` specifies that all symmetric expansion terms up to a given
          order are to be computed.
        - ``expansion_labels`` allows specification of specific terms to be computed, by
          specifying :class:`~qiskit_dynamics.perturbation.Multiset`\s.
          Both of ``expansion_order``
          and ``expansion_labels`` are optional, however at least one must be specified.
          If both are specified, then all terms up to ``expansion_order`` will be computed,
          along with any additional specific terms given by ``expansion_labels``.
        - ``generator`` is the unperturbed generator, and the computation is performed
          in the toggling frame of this generator.
        - ``y0`` is the initial state for the LMDE given by the unperturbed generator.


    If ``expansion_method == 'dyson'``, the setup is different. In this case,
    for a list of matrix-valued functions :math:`A_0(t), \dots, A_{r-1}(t)`,
    this function computes integrals of the form

    .. math::
        \int_{t_0}^{t_F} dt_1 \int_{t_0}^{t_1} dt_2 \dots \int_{t_0}^{t_{k-1}}dt_k
                \tilde{A}_{i_1}(t_1) \dots \tilde{A}_{i_k}(t_k),

    for lists of integers :math:`[i_1, \dots, i_k]`, and similar to the symmetric case,
    :math:`\tilde{A}_j(t) = V(t_0, t)^\dagger A_j(t)V(t_0, t)`, i.e. the computation
    is performed in the toggling frame specified by ``generator``.

        - ``perturbations`` gives the list of matrix functions as callables
          :math:`A_0(t), \dots, A_{r-1}(t)`.
        - ``perturbation_labels`` is not used in this mode.
        - ``expansion_order`` specifies that all possible integrals of the above form
          should be computed up to a given order
          (i.e. integrals up to a given order with all possible orderings of the
          :math:`A_0(t), \dots, A_{r-1}(t)`).
        - ``expansion_labels`` allows for specification of specific terms to be computed.
          In this case, a term is specified by a list of ``int``\s, where the length
          of the list is the order of the integral, and the :math:`A_0(t), \dots, A_{r-1}(t)`
          appear in the integral in the order given by the list.
        - ``generator`` serves the same function as in the symmetric case - the computation
          is performed in the toggling frame of ``generator``.
        - ``y0`` again is the initial state of the LMDE given by ``generator``.

    .. note::

        The ``dyson_in_frame`` argument is used for both the ``'dyson'``
        and ``'symmetric_dyson'`` expansion methods. If ``True``, the results are
        returned as above. If ``False``, the returned expansion terms will include
        a pre-factor of :math:`V(t_0, t)`, e.g. :math:`V(t_0, t)\mathcal{D}_I(t_0, t)`
        in the case of a symmetric Dyson term. If the ``'symmetric_magnus'`` method is
        used, ``dyson_in_frame`` has no effect on the output.


    Regardless of the value of ``expansion_method``, results are returned in an
    ``OdeResult`` instance in the same manner as :func:`~qiskit_dynamics.solvers.solve_ode`.
    The result object stores the results of the LMDE for ``generator`` and ``y0``
    in the ``y`` attribute as in :func:`~qiskit_dynamics.solvers.solve_ode` before,
    and the perturbation results are in the ``perturbation_results`` attribute storing a
    :class:`~qiskit_dynamics.perturbation.PerturbationResults` object, which is a
    data container with attributes:

        - ``expansion_method``: Method as specified by the user.
        - ``expansion_labels``: Index labels for all computed perturbation terms. In the case of
          a symmetric expansion, the labels are
          :class:`~qiskit_dynamics.perturbation.multiset.Multiset` instances, and in the
          non-symmetric case are lists of ``int``\s.
        - ``expansion_terms``: A 4d array storing all computed terms. The first axis indexes
          the expansion terms in the same ordering as ``expansion_labels``,
          and the second axis indexes the perturbation terms evaluated
          at the times in ``results.t`` in the same manner as ``results.y``.

    Additionally, to retrieve the term with a given label, a
    :class:`~qiskit_dynamics.perturbation.PerturbationResults` instance
    can be subscripted, e.g. the results for the computation for the term with indices
    ``[0, 1]`` is retrievable via ``results.perturbation_results[[0, 1]]``.

    Args:
        perturbations: List of matrix-valued callables.
        t_span: Integration bounds.
        expansion_method: Either ``'symmetric_dyson'``, ``'symmetric_magnus'``, or ``'dyson'``.
        expansion_order: Order of perturbation terms to compute up to. Specifying this
                         argument results in computation of all terms up to the given order.
                         Can be used in conjunction with ``expansion_labels``.
        expansion_labels: Specific perturbation terms to compute. If both ``expansion_order``
                          and ``expansion_labels`` are specified, then all terms up to
                          ``expansion_order`` are computed, along with the additional terms
                          specified in ``expansion_labels``.
        perturbation_labels: Optional description of power series terms specified by
                             ``perturbations``. To only be used with ``'symmetric_dyson'``
                             and ``'symmetric_magnus'`` methods.
        generator: Optional frame generator. Defaults to 0.
        y0: Optional initial state for frame generator LMDE. Defaults to the identity matrix.
        dyson_in_frame: For ``expansion_method`` ``'dyson'`` or ``'symmetric_dyson'``,
                        whether or not to remove the frame transformation pre-factor from the
                        Dyson terms.
        integration_method: Integration method to use.
        t_eval: Points at which to evaluate the system.
        **kwargs: Additional arguments to pass to ode integration method used to compute terms.

    Returns:
        OdeResult: Results object containing standard ODE results for LMDE given by ``generator``
        and ``y0``, with additional ``perturbation_results`` attribute containing the
        requested perturbation theory terms.

    Raises:
        QiskitError: If problem with inputs, either ``expansion_method`` is unsupported,
                     or both of ``expansion_order`` and ``expansion_labels`` unspecified.

    .. footbibliography::
    """

    # validation checks
    if y0 is not None:
        if len(y0.shape) != 2 or y0.shape[0] != y0.shape[1]:
            raise QiskitError("""If used, optional arg y0 must be a square 2d array.""")

    if perturbation_labels is not None and expansion_method == "dyson":
        raise QiskitError("perturbation_labels argument not usable with expansion_method='dyson'.")

    # clean and validate perturbation_labels, and setup expansion terms to compute
    if expansion_method in ["symmetric_dyson", "symmetric_magnus"]:

        if perturbation_labels is None:
            perturbation_labels = [Multiset({idx: 1}) for idx in range(len(perturbations))]
        else:
            # validate perturbation_labels
            perturbations_len = len(perturbation_labels)
            perturbation_labels = clean_multisets(perturbation_labels)
            if len(perturbation_labels) != perturbations_len:
                raise QiskitError("perturbation_labels argument contains duplicates as multisets.")

        expansion_labels = merge_multiset_expansion_order_labels(
            perturbation_labels=perturbation_labels,
            expansion_order=expansion_order,
            expansion_labels=expansion_labels,
        )
    elif expansion_method in ["dyson"]:
        expansion_labels = merge_list_expansion_order_labels(
            perturbation_num=len(perturbations),
            expansion_order=expansion_order,
            expansion_labels=expansion_labels,
        )

    if expansion_method in ["dyson", "symmetric_dyson"]:
        symmetric = expansion_method == "symmetric_dyson"
        if not Array.default_backend() == "jax":
            return solve_lmde_dyson(
                perturbations=perturbations,
                t_span=t_span,
                dyson_terms=expansion_labels,
                perturbation_labels=perturbation_labels,
                generator=generator,
                y0=y0,
                dyson_in_frame=dyson_in_frame,
                symmetric=symmetric,
                integration_method=integration_method,
                t_eval=t_eval,
                **kwargs,
            )
        else:
            return solve_lmde_dyson_jax(
                perturbations=perturbations,
                t_span=t_span,
                dyson_terms=expansion_labels,
                perturbation_labels=perturbation_labels,
                generator=generator,
                y0=y0,
                dyson_in_frame=dyson_in_frame,
                symmetric=symmetric,
                integration_method=integration_method,
                t_eval=t_eval,
                **kwargs,
            )
    elif expansion_method == "symmetric_magnus":
        if not Array.default_backend() == "jax":
            return solve_lmde_symmetric_magnus(
                perturbations=perturbations,
                t_span=t_span,
                magnus_terms=expansion_labels,
                perturbation_labels=perturbation_labels,
                generator=generator,
                y0=y0,
                integration_method=integration_method,
                t_eval=t_eval,
                **kwargs,
            )
        else:
            return solve_lmde_symmetric_magnus_jax(
                perturbations=perturbations,
                t_span=t_span,
                magnus_terms=expansion_labels,
                perturbation_labels=perturbation_labels,
                generator=generator,
                y0=y0,
                integration_method=integration_method,
                t_eval=t_eval,
                **kwargs,
            )

    # raise error if none apply
    raise QiskitError("expansion_method " + str(expansion_method) + " not supported.")
