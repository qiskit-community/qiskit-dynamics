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

from scipy.integrate._ivp.ivp import OdeResult  # pylint: disable=unused-import

from multiset import Multiset

from qiskit import QiskitError

from qiskit_dynamics.array import Array
from qiskit_dynamics.perturbation.multiset_utils import _clean_multisets
from qiskit_dynamics.perturbation.perturbation_utils import (
    _merge_multiset_expansion_order_labels,
    _merge_list_expansion_order_labels,
)

from qiskit_dynamics.perturbation.dyson_magnus import (
    _solve_lmde_dyson,
    _solve_lmde_magnus,
    _solve_lmde_dyson_jax,
    _solve_lmde_magnus_jax,
)


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

    This function computes multi-variable Dyson or Magnus expansion terms
    via the algorithm in :footcite:`puzzuoli_sensitivity_2022`, or Dyson-like terms
    via the algorithm in :footcite:`haas_engineering_2019`. See the
    :ref:`review on time-dependent perturbation theory <perturbation review>`
    to understand the details and notation used in this documentation.

    Which expansion is used is specified by the ``expansion_method`` argument, which impacts
    the interpretation of several of the function arguments (described below).
    Regardless of ``expansion_method``, the main computation is performed by
    solving a differential equation, utilizing :func:`.solve_ode`,
    and as such several of the function arguments are direct inputs into this function:

        - ``integration_method`` is the ODE method used (passed as ``method``
          to :func:`.solve_ode`), ``t_span`` is the integration interval,
          and ``t_eval`` is an optional set of points to evaluate the perturbation terms at.
        - ``kwargs`` are passed directly to :func:`.solve_ode`, enabling
          passing through of tolerance or step size arguments.

    Other arguments which are treated the same regardless off ``expansion_method`` are:

        - ``generator`` is the unperturbed generator, and the computation is performed
          in the toggling frame of this generator.

    If ``expansion_method in ['dyson', 'magnus']``, this function computes either
    multivariable Dyson series or Magnus expansion terms.
    That is, given a (finitely truncated) power series for the generator:

    .. math::

        G(t, c_0, \dots, c_{r-1}) = G_\emptyset(t)
            + \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I G_I(t),

    this function computes, in the toggling frame of :math:`G_\emptyset(t)` given by ``generator``,
    either a collection of multivariable Dyson terms :math:`\mathcal{D}_I(t)` or
    multivariable Magnus terms :math:`\mathcal{O}_I(t)`, whose definitions are
    given in the :ref:`perturbation theory review <perturbation review>`.
    In this case, the arguments to the function are interpreted as follows:

        - ``perturbations`` and ``perturbation_labels`` specify the truncated generator power
          series. ``perturbations`` provides a list of python callable functions for the non-zero
          :math:`G_I(t)`, and the ``perturbation_labels`` is a list of the corresponding
          multiset labels :math:`I`, in the form of ``Multiset`` instances. If not specified,
          the labels are assumed to be
          ``[Multiset({0: 1}), ..., Multiset({len(perturbations) - 1: 1})]``.
        - ``expansion_order`` and ``expansion_labels`` specify which terms in the chosen
          expansion are to be computed. ``expansion_order`` specifies that all expansion terms up
          to a given order are to be computed, and ``expansion_labels`` specifies individual terms
          to be computed, specified as ``Multiset`` instances.
          At least one of ``expansion_order`` and ``expansion_labels`` must be specified.
          If both are specified, then all terms up to ``expansion_order`` will be computed,
          along with any additional specific terms given by ``expansion_labels``.

    Note that in the above, this function requires that the Multisets consist of
    non-negative integers. Arguments requiring lists of ``Multiset`` instances also accept
    lists of any valid format acceptable to the ``Multiset`` constructor (modulo the
    non-negative integer constraint).

    If ``expansion_method == 'dyson_like'``, the setup is different. In this case,
    for a list of matrix-valued functions :math:`G_0(t), \dots, G_{r-1}(t)`,
    this function computes integrals of the form

    .. math::
        \int_{t_0}^{t_F} dt_1 \int_{t_0}^{t_1} dt_2 \dots \int_{t_0}^{t_{k-1}}dt_k
                \tilde{G}_{i_1}(t_1) \dots \tilde{G}_{i_k}(t_k),

    for lists of integers :math:`[i_1, \dots, i_k]`, and similar to the other
    cases, :math:`\tilde{G}_j(t) = V(t)^\dagger G_j(t)V(t)`, i.e. the computation
    is performed in the toggling frame specified by ``generator``.

        - ``perturbations`` gives the list of matrix functions as callables
          :math:`G_0(t), \dots, G_{r-1}(t)`.
        - ``perturbation_labels`` is not used in this mode.
        - ``expansion_order`` specifies that all possible integrals of the above form
          should be computed up to a given order
          (i.e. integrals up to a given order with all possible orderings of the
          :math:`G_0(t), \dots, G_{r-1}(t)`).
        - ``expansion_labels`` allows for specification of specific terms to be computed.
          In this case, a term is specified by a list of ``int``\s, where the length
          of the list is the order of the integral, and the :math:`G_0(t), \dots, G_{r-1}(t)`
          appear in the integral in the order given by the list.

    Finally, additional optional arguments which can be used in the
    ``'dyson'`` and ``'dyson_like'`` cases are:

        - ``dyson_in_frame`` controls which frame the results are returned in. The default
          is ``True``, in which case the results are returned as described above. If
          ``False``, the returned results include a pre-factor of :math:`V(t)`, the solution
          of the unperturbed generator, e.g.
          :math:`V(t)\mathcal{D}_I(t)`. If ``expansion_method=='magnus'``, this argument
          has no effect on the computation.
        - ``y0`` is the initial state for the LMDE given by the unperturbed generator.
          The effect of this argument on the output is to multiply all outputs by
          ``y0`` on the right.
          If ``y0`` is supplied, the argument ``dyson_in_frame`` must be ``False``,
          as the operator :math:`V(t)` is never explicitly computed, and therefore it
          cannot be removed from the results. If ``y0`` is 1d, it will first be
          transformed into a 2d column vector to ensure consistent usage of
          matrix multiplication.

    Regardless of the value of ``expansion_method``, results are returned in an
    ``OdeResult`` instance in the same manner as :func:`.solve_ode`. The result object
    stores the results of the LMDE for ``generator`` and ``y0`` in the ``y`` attribute as in
    :func:`.solve_ode` before, and the perturbation results are in the ``perturbation_results``
    attribute storing a :class:`.PerturbationResults` object, which is a
    data container with attributes:

        - ``expansion_method``: Method as specified by the user.
        - ``expansion_labels``: Index labels for all computed perturbation terms. In the case of
          the Dyson or Magnus expansion, the labels are ``Multiset`` instances and in the
          `'dyson_like'` case are lists of ``int``\s.
        - ``expansion_terms``: A 4d array storing all computed terms. The first axis indexes
          the expansion terms in the same ordering as ``expansion_labels``,
          and the second axis indexes the perturbation terms evaluated
          at the times in ``results.t`` in the same manner as ``results.y``.

    Additionally, to retrieve the term with a given label, the :meth:`.PerturbationResults.get_term`
    method of a :class:`.PerturbationResults` instance can be called
    e.g. the results for the computation for the term with label
    ``[0, 1]`` is retrievable via ``results.perturbation_results.get_term([0, 1])``.

    Args:
        perturbations: List of matrix-valued callables.
        t_span: Integration bounds.
        expansion_method: Either ``'dyson'``, ``'magnus'``, or ``'dyson_like'``.
        expansion_order: Order of perturbation terms to compute up to. Specifying this
                         argument results in computation of all terms up to the given order.
                         Can be used in conjunction with ``expansion_labels``.
        expansion_labels: Specific perturbation terms to compute. If both ``expansion_order``
                          and ``expansion_labels`` are specified, then all terms up to
                          ``expansion_order`` are computed, along with the additional terms
                          specified in ``expansion_labels``.
        perturbation_labels: Optional description of power series terms specified by
                             ``perturbations``. To only be used with ``'dyson'``
                             and ``'magnus'`` methods.
        generator: Optional frame generator. Defaults to 0.
        y0: Optional initial state for frame generator LMDE. Defaults to the identity matrix.
        dyson_in_frame: For ``expansion_method`` ``'dyson'`` or ``'dyson_like'``,
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
        if "magnus" in expansion_method:
            raise QiskitError("""Argument y0 cannot be used for expansion_method=='magnus'.""")

        if dyson_in_frame:
            raise QiskitError(
                """If expansion_method in ['dyson', 'dyson_like']
                   and y0 passed, dyson_in_frame must be False."""
            )

        # if 1d in a dyson case, turn into a column vector
        if y0.ndim == 1:
            y0 = Array([y0]).transpose()

    if perturbation_labels is not None and expansion_method == "dyson_like":
        raise QiskitError(
            """perturbation_labels argument not usable with
            expansion_method='dyson_like'."""
        )

    # clean and validate perturbation_labels, and setup expansion terms to compute
    if expansion_method in ["dyson", "magnus"]:

        if perturbation_labels is None:
            perturbation_labels = [Multiset({idx: 1}) for idx in range(len(perturbations))]
        else:
            # validate perturbation_labels
            perturbations_len = len(perturbation_labels)
            perturbation_labels = _clean_multisets(perturbation_labels)
            if len(perturbation_labels) != perturbations_len:
                raise QiskitError("perturbation_labels argument contains duplicates as multisets.")

        expansion_labels = _merge_multiset_expansion_order_labels(
            perturbation_labels=perturbation_labels,
            expansion_order=expansion_order,
            expansion_labels=expansion_labels,
        )
    elif expansion_method in ["dyson_like"]:
        expansion_labels = _merge_list_expansion_order_labels(
            perturbation_num=len(perturbations),
            expansion_order=expansion_order,
            expansion_labels=expansion_labels,
        )

    if expansion_method in ["dyson", "dyson_like"]:
        dyson_like = expansion_method == "dyson_like"
        if not Array.default_backend() == "jax":
            return _solve_lmde_dyson(
                perturbations=perturbations,
                t_span=t_span,
                dyson_terms=expansion_labels,
                perturbation_labels=perturbation_labels,
                generator=generator,
                y0=y0,
                dyson_in_frame=dyson_in_frame,
                dyson_like=dyson_like,
                integration_method=integration_method,
                t_eval=t_eval,
                **kwargs,
            )
        else:
            return _solve_lmde_dyson_jax(
                perturbations=perturbations,
                t_span=t_span,
                dyson_terms=expansion_labels,
                perturbation_labels=perturbation_labels,
                generator=generator,
                y0=y0,
                dyson_in_frame=dyson_in_frame,
                dyson_like=dyson_like,
                integration_method=integration_method,
                t_eval=t_eval,
                **kwargs,
            )
    elif expansion_method == "magnus":
        if not Array.default_backend() == "jax":
            return _solve_lmde_magnus(
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
            return _solve_lmde_magnus_jax(
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
    raise QiskitError(f"expansion_method {expansion_method} not supported.")
