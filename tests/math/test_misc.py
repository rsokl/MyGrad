from functools import partial
from typing import Callable

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

from mygrad import clip, maximum, minimum
from mygrad.tensor_base import Tensor
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def is_not_close(arr0: Tensor, arr1: Tensor) -> bool:
    return not np.any(np.isclose(arr0.data, arr1.data))


@backprop_test_factory(
    mygrad_func=maximum, true_func=np.maximum, num_arrays=2, assumptions=is_not_close
)
def test_maximum_bkwd():
    pass


def test_maximum_bkwd_equal():
    """regression test for documented behavior of maximum/minimum where
    x == y"""

    x = Tensor([1.0, 0.0, 2.0])
    y = Tensor([2.0, 0.0, 1.0])

    o = maximum(x, y)
    o.backward()

    assert_allclose(x.grad, [0.0, 0.0, 1])
    assert_allclose(y.grad, [1.0, 0.0, 0])

    # ensure branch covered for equal scalars
    x = Tensor(1.0)
    y = Tensor(1.0)

    o = maximum(x, y)
    o.backward()

    assert_allclose(x.grad, 0.0)
    assert_allclose(y.grad, 0.0)


@backprop_test_factory(
    mygrad_func=minimum, true_func=np.minimum, num_arrays=2, assumptions=is_not_close
)
def test_minimum_bkwd():
    pass


def test_minimum_bkwd_equal():
    """regression test for documented behavior of minimum/minimum where
    x == y"""

    x = Tensor([1.0, 0.0, 2.0])
    y = Tensor([2.0, 0.0, 1.0])

    o = minimum(x, y)
    o.backward()

    assert_allclose(x.grad, [1.0, 0.0, 0.0])
    assert_allclose(y.grad, [0.0, 0.0, 1.0])

    # ensure branch covered for equal scalars
    x = Tensor(1.0)
    y = Tensor(1.0)

    o = minimum(x, y)
    o.backward()

    assert_allclose(x.grad, 0.0)
    assert_allclose(y.grad, 0.0)


def to_min_max(arr: np.ndarray) -> st.SearchStrategy:
    bnd_shape = hnp.broadcastable_shapes(
        shape=arr.shape, max_dims=arr.ndim, max_side=min(arr.shape) if arr.ndim else 1
    )
    bnd_strat = hnp.arrays(
        shape=bnd_shape, elements=st.floats(-1e6, 1e6), dtype=np.float64
    )
    return st.fixed_dictionaries(dict(a_min=bnd_strat, a_max=bnd_strat))


def amin_clip_only(clip_func, a, b, constant=False):
    return (
        clip_func(a, a_min=b, a_max=None, constant=constant)
        if constant is not None
        else clip_func(a, a_min=b, a_max=None)
    )


def amax_clip_only(clip_func, a, b, constant=False):
    return (
        clip_func(a, a_max=b, a_min=None, constant=constant)
        if constant is not None
        else clip_func(a, a_max=b, a_min=None)
    )


skip_if_lower_than_numpy_1p17 = pytest.mark.skipif(
    np.__version__ < "1.17",
    reason="numpy.clip behavior was made consistent in numpy-1.17; "
    "test must by run on numpy 1.17 or later",
)


@pytest.mark.parametrize(
    ("mygrad_clip", "numpy_clip", "num_arrays"),
    [
        (
            partial(amin_clip_only, clip),
            partial(amin_clip_only, np.clip, constant=None),
            2,
        ),
        (
            partial(amax_clip_only, clip),
            partial(amax_clip_only, np.clip, constant=None),
            2,
        ),
        (clip, np.clip, 3),
    ],
)
@skip_if_lower_than_numpy_1p17
def test_clip_fwd(mygrad_clip: Callable, numpy_clip: Callable, num_arrays: int):
    @fwdprop_test_factory(
        num_arrays=num_arrays, mygrad_func=mygrad_clip, true_func=numpy_clip
    )
    def wrapped_test():
        pass

    wrapped_test()


def is_not_close_clip(a: Tensor, a_min=None, a_max=None) -> bool:
    min_close = np.any(np.isclose(a.data, a_min.data)) if a_min is not None else False
    max_close = np.any(np.isclose(a.data, a_max.data)) if a_max is not None else False
    return not (min_close or max_close)


@pytest.mark.parametrize(
    ("mygrad_clip", "numpy_clip", "num_arrays"),
    [
        (
            partial(amin_clip_only, np.clip),  # exercises __array_function__
            partial(amin_clip_only, np.clip, constant=None),
            2,
        ),
        (
            partial(amax_clip_only, np.clip),  # exercises __array_function__
            partial(amax_clip_only, np.clip, constant=None),
            2,
        ),
        (clip, np.clip, 3),
    ],
)
@skip_if_lower_than_numpy_1p17
def test_clip_bkwd(mygrad_clip: Callable, numpy_clip: Callable, num_arrays: int):
    @backprop_test_factory(
        num_arrays=num_arrays,
        mygrad_func=mygrad_clip,
        true_func=numpy_clip,
        vary_each_element=True,
        assumptions=is_not_close_clip,  # derivative is not defined where bounds and array are equal
    )
    def wrapped_test():
        pass

    wrapped_test()


@settings(max_examples=500)
@given(
    a=hnp.arrays(shape=hnp.array_shapes(min_dims=0), elements=st.floats(), dtype=float),
    a_min=st.none()
    | hnp.arrays(
        shape=hnp.array_shapes(min_dims=0),
        elements=st.floats(allow_nan=False),
        dtype=float,
    ),
    a_max=st.none()
    | hnp.arrays(
        shape=hnp.array_shapes(min_dims=0),
        elements=st.floats(allow_nan=False),
        dtype=float,
    ),
)
@skip_if_lower_than_numpy_1p17
@pytest.mark.filterwarnings("ignore: invalid value")
def test_clip_input_validation(a, a_min, a_max):
    try:
        numpy_out = np.clip(a, a_min, a_max)
    except Exception as e:
        with pytest.raises(type(e)):
            clip(a, a_min, a_max)
        return
    mygrad_out = clip(a, a_min, a_max)

    np.testing.assert_array_equal(numpy_out, mygrad_out.data)
