.. _perturbation review:

Time-dependent perturbation theory and multi-variable series expansions review
==============================================================================

The :mod:`.perturbation` module contains functionality for
numerically computing perturbation theory expansions used in the study of the
dynamics of quantum systems. Following :footcite:`puzzuoli_sensitivity_2022`,
this discussion reviews key concepts required
to understand and utilize the module, including the Dyson series and Magnus expansion,
their generalization to a multi-variable setting, and the notation used to represent
multi-variable power series in terms of multisets.


The Dyson series and Magnus expansion
-------------------------------------

The Dyson series :footcite:`dyson_radiation_1949` and Magnus expansion
:footcite:`magnus_exponential_1954,blanes_magnus_2009`
are time-dependent perturbation theory expansions for solutions of linear matrix differential
equations (LMDEs). For an LMDE

.. math::
    
    \dot{U}(t) = G(t)U(t)

with :math:`U(0) = I` the identity matrix, the Dyson series directly expands the solution:

.. math::

    U(t) = I + \sum_{k=1}^\infty D_k(t)

for :math:`D_k(t) = \int_0^t dt_1 \dots \int_0^{t_{k-1}} dt_k G(t_1) \dots G(t_k)`.

The Magnus expansion alternatively seeks to construct a time-averaged generator, i.e. an operator
:math:`\Omega(t)` for which 

.. math::
    U(t) = \exp(\Omega(t))`.
    
The Magnus expansion provides
a series expansion, which, under certain conditions :footcite:`blanes_magnus_2009`,
converges to :math:`\Omega(t)`:

.. math::

    \Omega(t) = \sum_{k=1}^\infty \Omega_k(t),

where explicit expressions for the :math:`\Omega_k(t)` are given in the literature
:footcite:`blanes_magnus_2009`.


Generalizing to the multi-variable case
---------------------------------------

In applications, these expansions are often used in a *multi-variable* setting, in which
the generator :math:`G(t)` depends on several variables :math:`c_0, \dots, c_{r-1}`,
and the Dyson series or Magnus expansion are used to understand how the evolution changes
under perturbations to several of these parameters simultaneously. For working with
these expansions algorithmically it is necessary to formalize
the expression of these expansions in the multi-variable setting.

Mathematically, we explicitly write the generator as a function of these variables
:math:`G(t, c_0, \dots, c_{r-1})`, and expand :math:`G` in a
multi-variable power series in the variables :math:`c_i`:

.. math::

    G(t, c_0, \dots, c_{r-1}) =
    G_\emptyset(t) +
    \sum_{k=1}^\infty \sum_{0 \leq i_1 \leq \dots \leq i_k \leq r-1}
    c_{i_1} \dots c_{i_k} G_{i_1, \dots, i_k}(t).

For physical applications we take the existence of such a power series for granted:
up to constant factors, the coefficients :math:`G_{i_1, \dots, i_k}(t)` are the partial
derivatives of :math:`G` with respect to the variables :math:`c_i`. Commonly, :math:`G`
depends linearly on the variables, e.g. when representing couplings between quantum systems.

Before defining the multi-variable Dyson series and Magnus expansions, we transform
the generator into the *toggling frame* of :math:`G_\emptyset(t)`
:footcite:`evans_timedependent_1967,haeberlen_1968`. Denoting
:math:`V(t) = \mathcal{T}\exp(\int_{t_0}^t ds G_\emptyset(s))`,
the generator :math:`G` in the toggling frame of :math:`G_\emptyset(t)`,
the unperturbed generator, is given by:

.. math::

    \tilde{G}(t, c_0, \dots, c_{r-1}) =
    \sum_{k=1}^\infty \sum_{0 \leq i_1 \leq \dots \leq i_k \leq r-1}
    c_{i_1} \dots c_{i_k} \tilde{G}_{i_1, \dots, i_k}(t),

with :math:`\tilde{G}_{i_1, \dots, i_k}(t) = V^{-1}(t) G_{i_1, \dots, i_k}(t)V(t)`.
Denoting :math:`U(t, c_0, \dots, c_{r-1})` as the solution of the LMDE with
generator :math:`\tilde{G}`, note that

.. math::

    U(t, c_0, \dots, c_{r-1}) =
    V(t)\mathcal{T}\exp\left(\int_{t_0}^t ds \tilde{G}(s, c_0, \dots, c_{r-1})\right),

and hence solution for :math:`G` and :math:`\tilde{G}` are simply related by :math:`V(t)`.

Using this, :footcite:`puzzuoli_sensitivity_2022` defines the multi-variable Dyson series
for the generator :math:`\tilde{G}(t, c_0, \dots, c_{r-1})` as:

.. math::

    U(t, c_0, \dots, c_{r-1}) = I +
    \sum_{k=1}^\infty \sum_{0 \leq i_1 \leq \dots \leq i_k \leq r-1}
    c_{i_1} \dots c_{i_k} \mathcal{D}_{i_1, \dots, i_k}(t),

where the :math:`\mathcal{D}_{i_1, \dots, i_k}(t)` are defined implicitly by the above
equation, and are called the *multi-variable Dyson series terms*. Similarly the
multi-variable Magnus expansion for :math:`\tilde{G}` is given as:

.. math::

    \Omega(t, c_0, \dots, c_{r-1}) =
    \sum_{k=1}^\infty \sum_{0 \leq i_1 \leq \dots \leq i_k \leq r-1}
    c_{i_1} \dots c_{i_k} \mathcal{O}_{i_1, \dots, i_k}(t),

with the :math:`\mathcal{O}_{i_1, \dots, i_k}(t)` again defined implicitly, and called the
*multi-variable Magnus expansion terms*.


Computing multi-variable Dyson series and Magnus expansion terms
----------------------------------------------------------------

Given a power series decomposition of the generator as above,
the function :func:`.solve_lmde_perturbation` computes,
in the toggling frame of the unperturbed generator, either multi-variable
Dyson series or Magnus expansion terms via the algorithms in
:footcite:`puzzuoli_sensitivity_2022`. It can also be used to compute Dyson-like terms via
the algorithm in :footcite:`haas_engineering_2019`. In the presentation here and elsewhere,
the expansions are phrased as infinite series, but of course in practice truncated
versions must be specified and computed.

Utilizing this function, and working with the other objects in the module, requires
understanding the notation and data structures used to represent power series.

.. _multiset power series:

Multiset power series notation
------------------------------

Following :footcite:`puzzuoli_sensitivity_2022`, the :mod:`.perturbation`
module utilizes a *multiset* notation to more compactly represent and work with power series.

Consider the power series expansion above for the generator :math:`G(t, c_0, \dots, c_{r-1})`.
Structurally, each term in the power series is labelled by the number of times each
variable :math:`c_0, \dots, c_{r-1}` appears in the product :math:`c_{i_1} \dots c_{i_k}`.
Equivalently, each term may be indexed by the number of times each variable label
:math:`0, \dots, r-1` appears. The data structure used to represent these labels in this
module is that of a *multiset*, i.e. a "set with repeated entries". Denoting multisets
with round brackets, e.g. :math:`I = (i_1, \dots, i_k)`, we define

.. math::

    c_I = c_{i_1} \times \dots \times c_{i_k}.

and similarly denote :math:`G_I = G_{i_1, \dots, i_k}`. This notation is chosen due to
the simple relationship between algebraic operations and multiset operations. E.g.,
for two multisets :math:`I, J`, it holds that:

.. math::

    c_{I + J} = c_I \times c_J,

where :math:`I + J` denotes the multiset whose object counts are the sum of both :math:`I` and
:math:`J`.

Some example usages of this notation are:

    - :math:`c_{(0, 1)} = c_0 c_1`,
    - :math:`c_{(1, 1)} = c_1^2`, and
    - :math:`c_{(1, 2, 2, 3)} = c_1 c_2^2 c_3`.

Finally, we denote the set of multisets of size $k$ with elements in :math:`\{0, \dots, r-1\}`
as :math:`\mathcal{I}_k(r)`. Combining everything, the power series for :math:`G` may be
rewritten as:

.. math::

    G(t, c_0, \dots, c_{r-1}) = G_\emptyset(t)
    + \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I G_I(t).

Similarly, the multi-variable Dyson series is written as:

.. math::

    U(t, c_0, \dots, c_{r-1}) =
            I + \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \mathcal{D}_I(t),

and the multi-variable Magnus expansion as:

.. math::

    \Omega(t, c_0, \dots, c_{r-1}) =
            \sum_{k=1}^\infty \sum_{I \in \mathcal{I}_k(r)} c_I \mathcal{O}_I(t).

In the module, multisets are represented using the ``Multiset`` object in the
`multiset package <https://pypi.org/project/multiset/>`_. Arguments to functions
which must specify a multiset or a list of multisets accept either ``Multiset`` instances
directly, or a valid argument to the constructor to ``Multiset``, with the restriction that
the multiset entries must be non-negative integers.




.. footbibliography::
