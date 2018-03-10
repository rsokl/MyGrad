""" Custom hypothesis search strategies """
import hypothesis.strategies as st


def choices(seq, size, replace=True):
    """Randomly choose elements from `seq`, producing a tuple of length `size`."""
    if size > len(seq) and not replace:
        raise ValueError("`size` must not exceed the length of `seq`")
    inds = range(len(seq))
    strat = st.tuples(*[st.sampled_from(inds)]*size)
    if not replace:
        strat = strat.filter(lambda x: len(set(x)) == size)
    return strat.map(lambda x: tuple(seq[i] for i in x))


@st.composite
def rand_neg_axis(draw, axes, ndim):
    """Randomly replace axis values with negative counterpart: axis - ndim"""
    size = draw(st.integers(0, len(axes)))
    neg_inds = draw(choices(range(len(axes)), size, replace=False))
    axes = st.just(axes)
    return draw(axes.map(lambda x: tuple(i - ndim if n in neg_inds else i for n, i in enumerate(x))))


@st.composite
def valid_axes(draw, ndim, pos_only=False):
    """ Hypothesis search strategy: Given array dimensionality, generate valid
        `axis` arguments (including `None`).


        Parameters
        ----------
        ndim : int
            The dimensionality of the array.

        Returns
        -------
        hypothesis.searchstrategy.lazy.LazyStrategy

        Examples
        --------
        >>> valid_axes(4).example()
        (0, 1)
        """
    if 0 > ndim:
        raise ValueError("`ndim` must be an integer 0 or greater.")
    num_axes = draw(st.integers(min_value=0, max_value=ndim))
    axes = draw(choices(range(ndim), num_axes, replace=False))
    if not pos_only:
        return draw(rand_neg_axis(axes, ndim))
    return draw(st.none()) if not axes else axes
