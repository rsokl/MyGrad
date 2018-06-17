""" Custom hypothesis search strategies """
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np

__all__ = ["adv_integer_index",
           "broadcastable_shape",
           "choices",
           "valid_axes",
           "basic_index"]


def choices(seq, size, replace=True):
    """Randomly choose elements from `seq`, producing a tuple of length `size`."""
    if size > len(seq) and not replace:
        raise ValueError("`size` must not exceed the length of `seq` when `replace` is `False`")
    if size > len(seq) and not seq:
        raise ValueError("`size` must be 0, given an empty `seq`")
    inds = list(range(len(seq)))
    if replace:
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
    return draw(axes.map(lambda x: tuple(i - ndim if n in neg_inds else i for n, i in enumerate(x))))


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
def broadcastable_shape(draw, shape, min_dim=0, max_dim=5):
    """ Hypothesis search strategy: given an array shape, generate a
        broadcast-compatible shape, specifying the minimum/maximum permissable
        number of dimensions in the resulting shape (both values are inclusive).

        `draw` is a parameter reserved by hypothesis, and should not be specified
        by the user.

        Parameters
        ----------
        shape : Tuple[int, ...]
        min_dim : int
        max_dim : int

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> Tuple[int, ...]
        """
    ndim = draw(st.integers(min_dim, max_dim))
    n_aligned = min(len(shape), ndim)
    n_leading = ndim - n_aligned
    aligned_dims = draw(st.tuples(*(st.sampled_from([1, size]) for size in shape[::-1][:n_aligned])))[::-1]
    leading_dims = draw(st.tuples(*(st.integers(1, 5) for i in range(n_leading))))
    return leading_dims + aligned_dims


def integer_index(size):
    """ Generate a valid integer-index for an axis of a given size

        Parameters
        ----------
        size : int
            Size of the axis for which the index is drawn

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> int"""
    return st.sampled_from(list(range(size)) + list(range(-1, -size, -1)))


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
    step = draw(st.sampled_from(list(range(1, max(2, size // 2))) + [-1]))
    start = draw(st.sampled_from(range(size + 1)))
    if step > 0:
        stop = draw(st.sampled_from(range(start, size)))
    else:
        stop = draw(st.sampled_from(range(0, start)))

    return slice(start, stop, step)


@st.composite
def basic_index(draw, shape, max_dim=5):
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
    min_dim = len(shape)
    max_dim = max(min_dim, max_dim)

    index = list(draw(st.tuples(*(st.one_of(integer_index(size), slice_index(size)) for size in shape))))
    if min_dim != max_dim:
        num_newaxis = draw(st.integers(min_dim, max_dim)) - len(index)
        if num_newaxis:
            for ind in draw(choices(range(min_dim + 1), num_newaxis)):
                index.insert(ind, np.newaxis)
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
                                        shape=index_shape, elements=st.integers(0, size - 1))
                             for size in shape)))

    return index