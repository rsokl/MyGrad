""" Custom hypothesis search strategies """
import hypothesis.strategies as st

__all__ = ["broadcastable_shape",
           "choices",
           "valid_axes"]

def choices(seq, size, replace=True):
    """Randomly choose elements from `seq`, producing a tuple of length `size`."""
    if size > len(seq) and not replace:
        raise ValueError("`size` must not exceed the length of `seq` when `replace` is `False`")
    if size > len(seq) and not seq:
        raise ValueError("`size` must be 0, given an empty `seq`")
    inds = range(len(seq))
    strat = st.tuples(*[st.sampled_from(inds)]*size)
    if not replace:
        strat = strat.filter(lambda x: len(set(x)) == size)
    return strat.map(lambda x: tuple(seq[i] for i in x))


@st.composite
def _rand_neg_axis(draw, axes, ndim):
    """Randomly replace axis values with negative counterpart: axis - ndim"""
    size = draw(st.integers(0, len(axes)))
    neg_inds = draw(choices(range(len(axes)), size, replace=False))
    axes = st.just(axes)
    return draw(axes.map(lambda x: tuple(i - ndim if n in neg_inds else i for n, i in enumerate(x))))


@st.composite
def valid_axes(draw, ndim, pos_only=False, single_axis_only=False):
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

    return draw(st.none()) if axes == () else axes


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
