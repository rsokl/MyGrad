""" Custom hypothesis search strategies """
import hypothesis.strategies as st
from hypothesis import assume

from decimal import Decimal, getcontext

getcontext().prec = 14

@st.composite
def numerical_derivative(draw, f, xbnds=(-100, 100), no_go=(), h=1e-8):
    """ Hypothesis search strategy: Sample x from specified bounds,
        and compute::

                  dfdx = (f(x + h) - f(x - h)) / (2h)

        Returning a search-strategy for: (x, dfdx)

        Makes use of `decimal.Decimal` for high-precision arithmetic.

        Parameters
        ----------
        f : Callable[[Real], Real]
            A differentiable unary function: f(x)

        xbnds : Tuple[Real, Real], optional (default=(-100, 100))
            Defines the domain bounds (inclusive) from which `x` is drawn.

        no_go : Iterable[Real, ...], optional (default=())
            An iterable of values from which `x` will not be drawn.

        h : Real, optional (default=1e-8)
            Approximating infinitesimal.

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> Tuple[decimals.Decimal, decimals.Decimal]
            (x, df/dx) """
    h = Decimal(h)
    x = draw(st.decimals(min_value=xbnds[0], max_value=xbnds[1]))
    for x_val in no_go:
        assume(x != x_val)

    dx = (Decimal(f(x + h)) - Decimal(f(x - h))) / (Decimal(2) * h)
    return x, dx


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
        hypothesis.searchstrategy.SearchStrategy
         -> [Union[NoneType, Tuple[int...]]

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
        axes = draw(rand_neg_axis(axes, ndim))
    return draw(st.none()) if not axes else axes