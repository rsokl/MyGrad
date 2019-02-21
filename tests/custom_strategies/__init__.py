""" Custom hypothesis search strategies """
from numbers import Integral

import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from hypothesis import note

import numpy as np

__all__ = ["adv_integer_index",
           "broadcastable_shape",
           "choices",
           "valid_axes",
           "basic_index"]

def _check_min_max(min_val, min_dim, max_dim, param_name):
    if not isinstance(min_dim, Integral) or min_dim < min_val:
        raise ValueError("`min_{name}` must be larger than {min_val}. "
                         "Got {val}".format(min_val=min_val, name=param_name,
                                            val=min_dim))

    if not isinstance(max_dim, Integral) or max_dim < min_dim:
        raise ValueError("`max_{name}` must be an integer that is "
                         "not smaller than `min_{name}`. Got {val}".format(name=param_name,
                                                                           val=max_dim))

def choices(seq, size, replace=True):
    """Randomly choose elements from `seq`, producing a tuple of length `size`."""
    if not isinstance(size, Integral) or size < 0:
        raise ValueError("`size` must be a non-negative integer. Got {}".format(size))
    if size > len(seq) and not replace:
        raise ValueError("`size` must not exceed the length of `seq` when `replace` is `False`")
    if size > len(seq) and not seq:
        raise ValueError("`size` must be 0, given an empty `seq`")
    inds = list(range(len(seq)))
    if replace or size == 0:
        strat = st.tuples(*[st.sampled_from(inds)]*size)
    else:
        strat = st.permutations(inds)
    return strat.map(lambda x: tuple(seq[i] for i in x[:size]))


@st.composite
def _rand_neg_axis(draw, axes, ndim):
    """Randomly replace axis values with negative counterpart: axis - ndim"""
    size = draw(st.integers(0, len(axes)))
    neg_inds = draw(choices(range(len(axes)), size, replace=False))
    axes = st.just(axes)
    return draw(axes.map(lambda x: tuple(i - ndim if n in neg_inds else i
                                         for n, i in enumerate(x))))


@st.composite
def valid_axes(draw, ndim, pos_only=False, single_axis_only=False, permit_none=True):
    """ Hypothesis search strategy: Given array dimensionality, generate valid
        `axis` arguments (including `None`).


        Parameters
        ----------
        ndim : int
            The dimensionality of the array.

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
         -> [Union[NoneType, Tuple[int...]]

        Examples
        --------
        >>> valid_axes(4).example()
        (0, 1)
        """
    if isinstance(ndim, (tuple, list)):
        ndim = len(ndim)

    if 0 > ndim:
        raise ValueError("`ndim` must be an integer 0 or greater.")
    if single_axis_only:
        num_axes = draw(st.integers(min_value=0, max_value=1))
    else:
        num_axes = draw(st.integers(min_value=0, max_value=ndim))

    axes = draw(choices(range(ndim), num_axes, replace=False))

    if not pos_only:
        axes = draw(_rand_neg_axis(axes, ndim))

    if single_axis_only and axes:
        axes = axes[0]

    if permit_none:
        return draw(st.none()) if axes == () else axes
    else:
        return axes if axes else tuple(range(ndim))


@st.composite
def broadcastable_shape(draw, shape, min_dim=0, max_dim=5,
                        min_side=1, max_side=5, allow_singleton=True):
    """ Hypothesis search strategy: given an array shape, generate a
        broadcast-compatible shape, specifying the minimum/maximum permissable
        number of dimensions in the resulting shape (both values are inclusive).

        `draw` is a parameter reserved by hypothesis, and should not be specified
        by the user.

        Parameters
        ----------
        shape : Tuple[int, ...]
            The shape with which

        min_dim : int, optional (default=0)
            The smallest number of dimensions that the broadcast-compatible
            shape can possess.

        max_dim : int, optional (default=5)
            The largest number of dimensions that the broadcast-compatible
            shape can possess.

        min_side : int, optional (default=1)
            The smallest size that a new, leading dimensions can
            possess

        max_side : int, optional (default=5)
            The largest size that a new, leading dimension can
            possess.

        allow_singleton : bool, optional (default=True)
            If `False` the aligned dimensions of the broadcastable
            shape cannot contain singleton dimensions (i.e. size-1
            dimensions aligned with larger dimensions)

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy[Tuple[int, ...]]
            -> Tuple[int, ...]

        Examples
        --------
        >>> for i in range(5):
        ...    print(broadcastable_shape(shape=(2, 3)).example())
        (1, 3)
        ()
        (2, 3)
        (5, 2, 3)
        (8, 5, 1, 3)
        (3, )
        """
    _check_min_max(0, min_dim, max_dim, "dim")
    _check_min_max(1, min_side, max_side, "side")

    ndim = draw(st.integers(min_dim, max_dim))
    n_aligned = min(len(shape), ndim)
    n_leading = ndim - n_aligned
    if n_aligned > 0:
        if allow_singleton:
            aligned_dims = draw(st.tuples(*(st.sampled_from([1, size])
                                            for size in shape[-n_aligned:])))
        else:
            aligned_dims = shape[-n_aligned:]
    else:
        aligned_dims = tuple()

    leading_dims = draw(st.tuples(*(st.integers(min_side, max_side)
                                    for i in range(n_leading))))
    return leading_dims + aligned_dims


def integer_index(size):
    """ Generate a valid integer-index for an axis of a given size,
        either a positive or negative value.

        Parameters
        ----------
        size : int
            Size of the axis for which the index is drawn

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> int"""
    return st.integers(-size, size - 1)


@st.composite
def slice_index(draw, size):
    """ Hypothesis search strategy: Generate a valid slice-index
        for an axis of a given size. Slices are chosen such that
        most slices will not be empty.

        `draw` is a parameter reserved by hypothesis, and should not be specified
        by the user.

        Parameters
        ----------
        size : int
            Size of the axis for which the index is drawn

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> slice"""
    if not size:
        return slice(None)

    pos_start, pos_stop, pos_step = draw(st.tuples(*[st.booleans()]*3))

    step = draw(st.integers(1, max(2, size // 2)))
    if not pos_start:
        step *= -1
    start = draw(st.integers(0, size - 1) if pos_start else st.integers(-size, -1))
    _abs_start = start if start >= 0 else start + size
    stop = draw(st.integers(_abs_start, size) if pos_stop else st.integers(_abs_start - size, -1))
    return slice(start, stop, step) if pos_step else slice(stop, start, step)


@st.composite
def basic_index(draw, shape, min_dim=0, max_dim=5):
    """ Hypothesis search strategy: given an array shape, generate a
        a valid index for specifying an element/subarray of that array,
        using basic indexing.

        `draw` is a parameter reserved by hypothesis, and should not be specified
        by the user.

        Parameters
        ----------
        shape : Tuple[int, ...]
            The shape of the array whose indices are being generated

        max_dim : int
            The max dimensionality permitted, given the addition of new-axes.

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> Tuple[int, ...]
        """
    _check_min_max(0, min_dim, max_dim, "dim")

    ndim = len(shape)
    ndim_out = draw(st.integers(min_dim, max_dim))

    if not ndim_out:
        return draw(st.tuples(*(integer_index(size) for size in shape)))

    num_slice_axes = draw(st.sampled_from(range(0, min(ndim, ndim_out))))
    num_newaxis = max(0, ndim_out - num_slice_axes)
    num_int_axes = max(0, ndim - num_slice_axes)
    int_axes = draw(choices(range(ndim), size=num_int_axes, replace=False))
    slice_axes = draw(choices(sorted(set(range(ndim)) - set(int_axes)),
                              size=num_slice_axes, replace=False))
    note(f"shape: {shape}"
         f"\nndim_out: {ndim_out}"
         f"\nint_axes: {int_axes}"
         f"\nslice_axes: {slice_axes}"
         f"\nnum_newaxis: {num_newaxis}")

    index = [np.newaxis]*ndim
    for i in int_axes:
        index[i] = draw(integer_index(shape[i]))

    for i in slice_axes:
        index[i] = draw(slice_index(shape[i]))

    for i in draw(choices(range(len(index)+1), size=num_newaxis)):
        index.insert(i, np.newaxis)
    return tuple(index)


@st.composite
def adv_integer_index(draw, shape, max_dim=3):
    """ Hypothesis search strategy: given an array shape, generate a
        a valid index for specifying an element/subarray of that array,
        using advanced indexing with integer-valued arrays.

        `draw` is a parameter reserved by hypothesis, and should not be specified
        by the user.

        Parameters
        ----------
        shape : Tuple[int, ...]
            The shape of the array whose indices are being generated

        max_dim : int
            The max dimensionality permitted for the index-arrays.

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> Tuple[numpy.ndarray, ...]
        """
    index_shape = draw(hnp.array_shapes(max_dims=max_dim))
    index = draw(st.tuples(*(hnp.arrays(dtype=int,
                                        shape=index_shape, elements=integer_index(size))
                             for size in shape)))
    return index
