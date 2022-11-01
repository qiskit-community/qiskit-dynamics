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
# pylint: disable=invalid-name

"""
Utility functions for solvers.
"""

from typing import Callable, Iterable, List, Optional, Tuple, Union

from functools import lru_cache, partial, wraps
import inspect
import numpy as np
from scipy.integrate._ivp.ivp import OdeResult

from qiskit import QiskitError

from qiskit_dynamics.array import Array
from qiskit_dynamics.models import LindbladModel

try:
    from jax import jit
    from jax.lax import cond
    import jax.numpy as jnp
except ImportError:
    pass


def is_lindblad_model_vectorized(obj: any) -> bool:
    """Return True if obj is a vectorized LindbladModel."""
    return isinstance(obj, LindbladModel) and ("vectorized" in obj.evaluation_mode)


def is_lindblad_model_not_vectorized(obj: any) -> bool:
    """Return True if obj is a non-vectorized LindbladModel."""
    return isinstance(obj, LindbladModel) and ("vectorized" not in obj.evaluation_mode)


def merge_t_args(
    t_span: Union[List, Tuple, Array], t_eval: Optional[Union[List, Tuple, Array]] = None
) -> Union[List, Tuple, Array]:
    """Merge ``t_span`` and ``t_eval`` into a single array.

    Validition is similar to scipy ``solve_ivp``:
    ``t_eval`` must be contained in ``t_span``, and be increasing if
    ``t_span[1] > t_span[0]`` or decreasing if ``t_span[1] < t_span[0]``.

    Note: this is done explicitly with ``numpy``, and hence this is
    not differentiable or compilable using jax.

    If ``t_eval is None`` returns ``t_span`` with no modification.

    Args:
        t_span: Interval to solve over.
        t_eval: Time points to include in returned results.

    Returns:
        Union[List, Tuple, Array]: Combined list of times.

    Raises:
        ValueError: If one of several validation checks fail.
    """

    if t_eval is None:
        return t_span

    t_span = Array(t_span, backend="numpy")

    t_min = np.min(t_span)
    t_max = np.max(t_span)
    t_direction = np.sign(t_span[1] - t_span[0])

    t_eval = Array(t_eval, backend="numpy")

    if t_eval.ndim > 1:
        raise ValueError("t_eval must be 1 dimensional.")

    if np.min(t_eval) < t_min or np.max(t_eval) > t_max:
        raise ValueError("t_eval entries must lie in t_span.")

    diff = np.diff(t_eval)

    if np.any(t_direction * diff < 0.0):
        raise ValueError("t_eval must be ordered according to the direction of integration.")

    # add endpoints
    t_eval = np.append(np.append(t_span[0], t_eval), t_span[1])

    return Array(t_eval, backend="numpy")


def trim_t_results(
    results: OdeResult,
    t_eval: Optional[Union[List, Tuple, Array]] = None,
) -> OdeResult:
    """Trim ``OdeResult`` object if ``t_eval is not None``.

    Args:
        results: Result object, assumed to contain solution at time points
                 from the output of ``validate_and_merge_t_span_t_eval(t_span, t_eval)``.
        t_eval: Time points to include in returned results.

    Returns:
        OdeResult: Results with only times/solutions in ``t_eval``. If ``t_eval``
                   is ``None``, does nothing, returning solver default output.
    """

    if t_eval is None:
        return results

    # remove endpoints
    results.t = results.t[1:-1]
    results.y = Array(results.y[1:-1])

    return results


def merge_t_args_jax(t_span, t_eval=None):
    """JAX-compilable version of merge_t_args. Rather than raise errors, sets return values to
    jnp.nan to signal errors.
    """

    if t_eval is None:
        return Array(t_span, backend="jax")

    t_span = Array(t_span, backend="jax").data
    t_eval = Array(t_eval, backend="jax").data

    out = jnp.append(jnp.append(t_span[0], t_eval), t_span[1])

    # raise error if not one dimensional
    if t_eval.ndim > 1:
        raise ValueError("t_eval must be 1 dimensional.")

    # output nan if t_eval point lies outside t_span
    t_min = jnp.min(t_span)
    t_max = jnp.max(t_span)
    out = cond(
        (jnp.min(t_eval) < t_min) | (jnp.max(t_eval) > t_max),
        lambda s: jnp.nan * s,
        lambda s: s,
        out,
    )

    # output nan if t_eval and t_span have incompatible orderings
    diff = jnp.diff(t_eval)
    t_direction = jnp.sign(t_span[1] - t_span[0])
    out = cond(jnp.any(t_direction * diff < 0.0), lambda s: jnp.nan * s, lambda s: s, out)

    return Array(out)


def trim_t_results_jax(results, t_eval):
    """JAX-compilable version of trim_t_results. The cond call is necessary due to
    a bug in odeint that only occurs if the first two time values are duplicates.
    """

    if t_eval is None:
        return results

    results.y = Array(
        cond(
            results.t[0] == results.t[1],
            lambda y: jnp.append(jnp.array([y[0]]), y[2:-1], axis=0),
            lambda y: y[1:-1],
            Array(results.y).data,
        )
    )
    results.t = Array(results.t[1:-1])
    return results


def setup_args_lists(
    args_list: List[any], args_names: List[str], args_to_list: List[Callable]
) -> Tuple[List[List[any]], bool]:
    """Transform each entry of ``args_list`` into lists of the same length.

    All elements of args_list must be either given as lists of valid specifications,
    or as a singleton of a valid specification. ``args_to_list`` contains a list of functions
    that maps its corresponding entry in args_list to a valid list of singletons, along
    with a bool indicating whether the corresponding argument was already a list of singletons.
    All args are expanded to be the same length as the longest argument. For input in args_list:
        - If len(arg) == 1, it will be repeated max_len times
        - if len(arg) == max_len, nothing is done
        - If len(arg) not in (1, max_len), an error is raised

    Args:
        args_list: List of 'inputs'.
        args_names: Names of inputs, used for error raising.
        args_to_list: As described in the main function doc. Specialized error handling for
                      the entries in args_list should be part of these functions in the
                      case that the arg is neither a valid singleton, nor a valid list
                      of singletons.

    Returns:
        Tuple: First entry is the transformed version of args_list so that all entries are the
               lists of the same length. Second entry is a boolean stating whether the inputs
               were given all as single instances (False), or one was a list (True).

    Raises:
        QiskitError: If inputs have incompatible lengths.
    """

    args_as_lists = []
    args_were_lists = False
    for arg, to_list in zip(args_list, args_to_list):
        arg_as_list, arg_was_list = to_list(arg)
        args_as_lists.append(arg_as_list)
        args_were_lists = args_were_lists or arg_was_list

    arg_lens = [len(x) for x in args_as_lists]
    max_len = max(arg_lens)
    for idx, arg_len in enumerate(arg_lens):
        if arg_len not in (1, max_len):
            max_name = args_names[arg_lens.index(max_len)]

            arg_name_sequence = ""
            for name in args_names[:-1]:
                arg_name_sequence += f"{name}, "
            arg_name_sequence += f"and {args_names[-1]}"

            raise QiskitError(
                f"""If one of {arg_name_sequence} is given as a list of valid inputs, then the
                others must specify only a single input, or a list of the same length.
                {max_name} specifies {max_len} inputs, but {args_names[idx]} is of length {arg_len},
                which is incompatible."""
            )

    args_as_lists = [
        x * max_len if arg_len == 1 else x for x, arg_len in zip(args_as_lists, arg_lens)
    ]
    return args_as_lists, args_were_lists


class HashWrapper:
    """Thin hashable wrapper over any object."""

    def __init__(self, obj):
        self.obj = obj
        try:
            self._hash = hash(obj)
        except TypeError:
            # force hash collisions on unhashable objects; whenever this HashWrapper
            # is used in a hash table, it will have to fall back to __eq__
            self._hash = 0

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self.obj == other

    def __repr__(self):
        return f"HashWrapper({repr(self.obj)})"


def jit_with_static_mutables(
    fn: Callable,
    maxsize: int = 128,
    static_mutable_argnames: Union[str, Iterable[str], None] = None,
    **jit_kwargs,
) -> Callable:
    r"""A drop-in replacement for ``jax.jit`` that additionally allows mutable static arguments.

    .. code-block::

        from functools import partial
        from qiskit_dynamics.solvers.solver_utils import jit_with_static_mutables
        import numpy as np

        # with jax.jit alone, the following would not be possible because dicts are mutable
        @partial(jit_with_static_mutables, static_argnames="cls", static_mutable_argnames="kwargs")
        def my_jit(arr, cls, kwargs: dict):
            return arr

        my_jit(np.array([1,2,3]), int, dict(foo=True))
        my_jit(np.array([1,2,3]), int, dict(foo=True))
        my_jit(np.array([1,2,3]), int, dict(foo=True))

        # for mutable argnames, we get access to the same cache mechanisms as functools.lru_cache
        print(my_jit.cache_info()))

    Args:
        fn: The input function to JIT.
        maxsize: The maximum size of the internal LRU cache for static mutable arguments.
        static_mutable_argnames: One or more argument name of ``fn`` to treat as a static mutable.
        **jit_kwargs: Arguments to be forwarded to ``jax.jit``.

    Returns:
        A JITed callable with the same signature as ``fn``.

    Raises:
        ValueError: If one of the ``static_mutable_argnames`` is not in ``fn``s signature.
    """
    sig = inspect.signature(fn)
    static_mutable_argnames = set(
        [static_mutable_argnames]
        if isinstance(static_mutable_argnames, str)
        else static_mutable_argnames
    )
    if any(name not in sig.parameters for name in static_mutable_argnames):
        raise ValueError(f"A name in {static_mutable_argnames} is not present in fn's signature.")

    # we can lru_cache this function because its inputs are HashWrappers
    @lru_cache(maxsize=maxsize)
    def get_jit_fn(**mutable_args):
        # unwrap the HashWrappers and pass them to the underlying function
        mutable_args = {name: val.obj for name, val in mutable_args.items()}
        fn_static = partial(fn, **mutable_args)
        return jit(fn_static, **jit_kwargs)

    @wraps(fn)
    def wrapped(*args, **kwargs):
        # divide the arguments into static_mutable_argname ones, and all others
        all_args = sig.bind(*args, **kwargs)
        all_args.apply_defaults()
        mutable_args = {}
        static_args = {}
        for name, arg in all_args.arguments.items():
            if name in static_mutable_argnames:
                mutable_args[name] = HashWrapper(arg)
            else:
                static_args[name] = arg
        return get_jit_fn(**mutable_args)(**static_args)

    # attach cache information to the output function in case it is helpful
    wrapped.cache_info = get_jit_fn.cache_info
    wrapped.cache_parameters = get_jit_fn.cache_parameters
    wrapped.cache_clear = get_jit_fn.cache_clear

    return wrapped
