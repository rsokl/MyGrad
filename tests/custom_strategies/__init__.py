""" Custom hypothesis search strategies """
import math
from functools import lru_cache, partial
from itertools import groupby
from numbers import Integral
from operator import itemgetter
from typing import Any, Iterable, List, Optional, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis.extra.numpy import broadcastable_shapes
from numpy import ndarray

__all__ = [
    "adv_integer_index",
    "basic_indices",
    "broadcastable_shapes",
    "choices",
    "everything_except",
    "valid_axes",
]

Shape = Tuple[int, ...]

basic_indices = partial(hnp.basic_indices, allow_newaxis=True, allow_ellipsis=True)


def everything_except(
    excluded_types: Union[type, Tuple[type, ...]]
) -> st.SearchStrategy[Any]:
    """Returns hypothesis strategy that generates values of any type other than
    those specified in ``excluded_types``."""
    return (
        st.from_type(type)
        .flatmap(st.from_type)
        .filter(lambda x: not isinstance(x, excluded_types))
    )


def _check_min_max(min_val, min_dim, max_dim, param_name, max_val=None):
    """Ensures that:
        min_val <= min_dim
        min_val <= max_dim
        min_val <= max_val
    If `max_val` is specified, ensures that `max_dim <= max_val`

    Raises
    ------
    ValueError"""
    if not isinstance(min_dim, Integral) or min_dim < min_val:
        raise ValueError(
            f"`min_{param_name}` must be larger than {min_val}. " f"Got {min_dim}"
        )

    if not isinstance(max_dim, Integral) or max_dim < min_dim:
        raise ValueError(
            f"`max_{param_name}` must be an integer that is "
            f"not smaller than `min_{param_name}`. Got {max_dim}"
        )
    if max_val is not None and max_dim > max_val:
        raise ValueError(
            f"`min_{param_name}` cannot be larger than {max_val}. " f"Got {max_dim}"
        )

    if max_dim < min_dim:
        raise ValueError(
            f"`min_{param_name}={min_dim}` cannot be larger than max_{param_name}={max_dim}."
        )


def choices(seq, size, replace=True):
    """Randomly choose elements from `seq`, producing a tuple of length `size`.

    Examples from this strategy shrink towards `tuple(seq[:size])` when `replace=False.
    Examples from this strategy shrink towards `(seq[0], ) * size` when `replace=True.

    Parameters
    ----------
    seq : Sequence[Any]
    size : int
    replace : bool

    Returns
    -------
    hypothesis.strategiesSearchStrategy[Tuple[Any, ...]]
        A tuple of length `size` containing elements of `seq`"""
    if not isinstance(size, Integral) or size < 0:
        raise ValueError(f"`size` must be a non-negative integer. Got {size}")
    if size > len(seq) and not replace:
        raise ValueError(
            "`size` must not exceed the length of `seq` when `replace` is `False`"
        )
    if not seq:
        if size:
            raise ValueError("`size` must be 0, given an empty `seq`")
        return st.just(())
    return st.lists(
        st.sampled_from(range(len(seq))),
        min_size=size,
        max_size=size,
        unique=not replace,
    ).map(lambda x: tuple(seq[i] for i in x))


def _to_positive(x: Union[int, Iterable], ndim: int) -> Union[int, Tuple[int, ...]]:
    if hasattr(x, "__iter__"):
        return tuple(_to_positive(i, ndim) for i in x)
    return x if -1 < x else ndim + x


def valid_axes(
    ndim: int,
    pos_only: bool = False,
    single_axis_only: bool = False,
    permit_none: bool = True,
    permit_int: bool = True,
    min_dim: int = 0,
    max_dim: Optional[int] = None,
) -> st.SearchStrategy[Union[None, int, Tuple[int, ...]]]:
    """ Hypothesis search strategy: Given array dimensionality, generate valid
    `axis` arguments (including `None`) for numpy's sequential functions.

    Examples from this strategy shrink towards an empty tuple of axes.
    If `single_axis_only=True`, then it shrinks towards 0.

    Parameters
    ----------
    ndim : int
        The dimensionality of the array.

    pos_only : bool, optional (default=False)
        If True, the returned value(s) will be positive.

    single_axis_only : bool, optional (default=False)
        If True, a single integer axis or `None` (assuming `permit_none=True`)
        will be returned.

    permit_none : bool, optional (default=True)
        If True, `None` may be returned instead of a tuple of all of the
        available axes.

    permit_int: bool, optional (default=True)
        If True, the returned value may be an integer

    min_dim : int, optional (default=0)
        The smallest number of entries permitted in the returned tuple of axes

    max_dim : Optional[int]
        The largest number of entries permitted in the returned tuple of axes.
        The defaults is ``ndim``.

    Returns
    -------
    st.SearchStrategy[Union[None, int, Tuple[int, ...]]]

    Examples
    --------
    >>> valid_axes(4).example()
    (0, 1)
    """
    if isinstance(ndim, (tuple, list)):
        ndim = len(ndim)

    single_axis_strat = st.integers(-ndim, ndim - 1) if ndim else st.just(0)

    strats = []

    if permit_none:
        strats.append(st.none())

    if permit_int and min_dim <= 1 and (max_dim is None or 1 <= max_dim):
        strats.append(single_axis_strat)

    if not single_axis_only:
        strats.append(hnp.valid_tuple_axes(ndim, min_size=min_dim, max_size=max_dim))

    strat = st.one_of(*strats)
    if pos_only:
        strat = strat.map(lambda x: x if x is None else _to_positive(x, ndim))
    return strat


def integer_index(size):
    """ Generate a valid integer-index for an axis of a given size,
    either a positive or negative value: [-size, size).

    Examples from this strategy shrink towards 0.

    Parameters
    ----------
    size : int
        Size of the axis for which the index is drawn

    Returns
    -------
    hypothesis.searchstrategy.SearchStrategy[int]
    """
    return st.integers(-size, size - 1)


@st.composite
def slice_index(
    draw,
    size,
    min_start=None,
    max_start=None,
    min_stop=None,
    max_stop=None,
    min_step=1,
    max_step=2,
    negative_step=True,
):
    """ Hypothesis search strategy: Generate a valid slice-index
    for an axis of a given size. Slices are chosen such that
    most slices will not be empty.

    Examples from this strategy shrink towards `slice(0, 0, 1)`. In the
    case that a negative step size is drawn, start and stop will be flipped
    so that it is less likely to have an empty slice

    Parameters
    ----------
    size : int
        Size of the axis for which the index is drawn
    min_start : int
    max_start : int
    min_stop : int
    max_stop : int
    min_step : int, optional (default=1)
    max_step : int
    negative_step : bool

    Notes
    -----
    `draw` is a parameter reserved by hypothesis, and should not be specified
    by the user.

    Returns
    -------
    hypothesis.searchstrategy.SearchStrategy[slice]
    """
    if not size:
        return slice(None)

    min_start = -size if min_start is None else min_start
    max_start = size - 1 if max_start is None else max_start
    _check_min_max(-math.inf, min_start, max_start, "start")

    min_stop = -size if min_stop is None else min_stop
    max_stop = -size if max_stop is None else max_stop
    _check_min_max(min_start, min_stop, max_stop, "stop")

    _check_min_max(0, min_step, max_step, "step")

    start = draw(st.one_of(st.integers(min_start, max_start - 1), st.none()))
    stop = draw(
        st.one_of(st.integers(start if start is not None else 0, size), st.none())
    )

    step = draw(st.integers(min_step, max_step))

    if negative_step:
        neg_step = draw(st.booleans())

        if neg_step:
            step *= -1
    return slice(start, stop, step) if step > 0 else slice(stop, start, step)


def adv_integer_index(
    shape: Shape,
    min_dims: int = 1,
    max_dims: int = 3,
    min_side: int = 1,
    max_side: int = 3,
) -> st.SearchStrategy[Tuple[ndarray, ...]]:
    """ Hypothesis search strategy: given an array shape, generate a
    a valid index for specifying an element/subarray of that array,
    using advanced indexing with integer-valued arrays.

    Examples from this strategy shrink towards the index
    `len(shape) * (np.array([0]), )`.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape of the array whose indices are being generated

    min_dims : int, optional (default=1)
        The minimum dimensionality permitted for the index-arrays.

    max_dims : int, optional (default=3)
        The maximum dimensionality permitted for the index-arrays.

    min_side : int, optional (default=1)
        The minimum side permitted for the index-arrays.

    max_side : int, optional (default=3)
        The maximum side permitted for the index-arrays.

    Returns
    -------
    hypothesis.searchstrategy.SearchStrategy[Tuple[numpy.ndarray, ...]]
    """

    return hnp.integer_array_indices(
        shape=shape,
        result_shape=hnp.array_shapes(
            min_dims=min_dims, max_dims=max_dims, min_side=min_side, max_side=max_side
        ),
    )


@lru_cache(maxsize=1000)
def _factors(n: int) -> List[int]:
    """ Returns the divisors of n

        >>> _factors(4)
        {1, 2, 4}"""
    if not isinstance(n, int) and 0 <= n:
        raise ValueError(f"n={n} must be a non-negative integer")
    gen = ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)
    return sorted(set(sum(gen, [])))


@st.composite
def valid_shapes(
    draw, size: int, min_len: int = 1, max_len: int = 6
) -> st.SearchStrategy[Union[int, Tuple[int, ...]]]:
    """ Given an array's size, generate a compatible random shape

    Parameters
    ----------
    size : int
    min_len : int
    max_len : int

    Returns
    -------
    st.SearchStrategy[Tuple[int, ...]]
    """

    if not isinstance(size, int) or size < 0:
        raise ValueError(f"size={size} must be a non-negative integer")

    shape_length = draw(st.integers(min_len, max_len))  # type: int
    shape = []  # type: List[int]
    rem = int(size / np.prod(shape))
    while len(shape) < shape_length:
        if len(shape) == shape_length - 1:
            shape.append(-1 if draw(st.booleans()) else rem)
            break

        shape.append(draw(st.sampled_from(_factors(rem))))
        rem = int(size / np.prod(shape))

    return (
        shape[0]
        if len(shape) == 1 and draw(st.booleans())
        else tuple(int(i) for i in shape)
    )


@st.composite
def arbitrary_indices(draw, shape: Tuple[int]):
    """
    Hypothesis search strategy: Generate a valid index
    for an array of a given shape. The index can contain
    any type of valid object used for indexing, including
    integers, slices, Ellipsis's, newaxis's, boolean arrays,
    and integer arrays.

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the array to be indexed into

    Notes
    -----
    `draw` is a parameter reserved by hypothesis, and should not be specified
    by the user.

    Returns
    -------
    hypothesis.searchstrategy.SearchStrategy[Tuple[Union[int, slice, Ellipsis, NoneType, numpy.ndarray], ...]]
    """

    def group_continuous_integers(ls):
        """
        Given a list of integers, find and group continuous sequences

        Parameters
        ----------
        ls: List[int]

        Returns
        -------
        List[Tuple[int]]

        Examples
        --------
        >>> group_continuous_integers([1, 3, 4, 5, 7, 8])
        [(1,), (3, 4, 5), (7, 8)]
        """
        return [
            tuple(map(itemgetter(1), g))
            for k, g in groupby(enumerate(ls), lambda x: x[0] - x[1])
        ]

    if not shape or 0 in shape:
        return draw(hnp.basic_indices(shape=shape, allow_newaxis=True))

    shape_inds = list(range(len(shape)))
    index = []  # stores tuples of (dim, indexing object)

    # add integers, slices
    basic_inds = sorted(draw(st.lists(st.sampled_from(shape_inds), unique=True)))

    if len(basic_inds) > 0:
        basic_dims = tuple(shape[i] for i in basic_inds)

        # only draw ints and slices
        # will handle possible ellipsis and newaxis objects later
        # as these can make array indices difficult to handle
        basics = draw(hnp.basic_indices(shape=basic_dims, allow_ellipsis=False))
        if not isinstance(basics, tuple):
            basics = (basics,)

        index += [tup for tup in zip(basic_inds, basics)]

        # will not necessarily index all dims from basic_inds as
        # `basic_indices` can return indices with omitted trailing slices
        # so only remove dimensions directly indexed into
        for i in basic_inds[: len(basics)]:
            shape_inds.pop(shape_inds.index(i))

    if len(shape_inds) > 0:
        # add integer arrays to index
        int_arr_inds = sorted(draw(st.lists(st.sampled_from(shape_inds), unique=True)))

        if len(int_arr_inds) > 0:
            int_arr_dims = tuple(shape[i] for i in int_arr_inds)
            int_arrs = draw(hnp.integer_array_indices(shape=int_arr_dims))
            index += [tup for tup in zip(int_arr_inds, int_arrs)]

            for i in int_arr_inds:
                shape_inds.pop(shape_inds.index(i))

    if len(shape_inds) > 0:
        # add boolean arrays to index
        bool_inds = sorted(draw(st.lists(st.sampled_from(shape_inds), unique=True)))

        if len(bool_inds) > 0:
            # boolean arrays can be multi-dimensional, so by grouping all
            # adjacent indices to make a single boolean array, this can be tested for
            grouped_bool_inds = group_continuous_integers(bool_inds)
            bool_dims = [tuple(shape[i] for i in ind) for ind in grouped_bool_inds]

            # if multiple boolean array indices, the number of trues must be such that
            # the output of ind.nonzero() for each index are broadcast compatible
            # this must also be the same as the trailing index of each integer array, if any used
            if len(int_arr_inds):
                max_trues = max(i.shape[-1] for i in int_arrs)
            else:
                max_trues = st.integers(
                    min_value=0, max_value=min(bool_dims, key=lambda x: np.prod(x))
                )

            index += [
                (
                    i[0],
                    draw(
                        hnp.arrays(shape=sh, dtype=bool).filter(
                            lambda x: x.sum() in (1, max_trues)
                        )
                    ),
                )
                for i, sh in zip(grouped_bool_inds, bool_dims)
            ]

            for i in bool_inds:
                shape_inds.pop(shape_inds.index(i))

    grouped_shape_inds = group_continuous_integers(sorted(shape_inds))
    if len(grouped_shape_inds) == 1:
        # unused indices form a continuous stretch of dimensions
        # so can replace with an ellipsis

        # to test ellipsis vs omitted slices, randomly
        # add ellipsis when the unused indices are trailing
        if max(shape_inds) + 1 == len(shape):
            if draw(st.booleans()):
                index += [(min(shape_inds), Ellipsis)]
        else:
            index += [(min(shape_inds), Ellipsis)]
    elif len(grouped_shape_inds) == 0 and draw(st.booleans()):
        # all indices filled already
        # can randomly add ellipsis that expands to 0-d tuple
        # this can have counter-intuitive behavior
        # (particularly in conjunction with array indices)
        i = draw(st.integers(min_value=0, max_value=len(index)))
        index.insert(i, (i, Ellipsis))
    else:
        # so that current chosen index's work,
        # fill in remaining any gaps with empty slices
        index += [(i, slice(None)) for i in shape_inds]

    index = sorted(index, key=lambda x: x[0])

    # can now randomly add in newaxis objects
    newaxis_pos = sorted(
        draw(st.lists(st.integers(min_value=0, max_value=len(index)), unique=True)),
        reverse=True,
    )
    for i in newaxis_pos:
        index.insert(i, (-1, np.newaxis))

    out_ind = tuple(i[1] for i in index)
    return out_ind
