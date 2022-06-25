import warnings
from functools import partial

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, settings
from numpy.testing import assert_allclose

import mygrad as mg
from tests.custom_strategies import valid_axes
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def axis_strat(*ts, permit_none=False):
    (t,) = ts
    if t.ndim == 0:
        return st.none()
    v = valid_axes(
        t.ndim, single_axis_only=True, permit_int=True, permit_none=permit_none
    )
    return v | st.tuples(v)


def ord_strat(*ts):
    (t,) = ts
    if t.ndim == 0:
        return st.none()

    return st.sampled_from([1.0, 2.0, -1.0, 0.5, 1.25, 1.5, 2.5, -np.inf, np.inf])


def keepdims_strat(*ts):
    return st.booleans()


def manual_norm(x, ord=None, axis=None, keepdims=False):
    if ord is None:
        ord = 2
    if np.isinf(ord):
        if ord > 0:
            return mg.max(mg.abs(x), axis=axis, keepdims=keepdims)
        else:
            return mg.min(mg.abs(x), axis=axis, keepdims=keepdims)
    return (mg.sum(mg.abs(x) ** ord, axis=axis, keepdims=keepdims)) ** (1 / ord)


def manual_norm_2(x, ord=None, axis=None, keepdims=False):
    return np.sqrt(np.sum(x ** 2, axis=axis, keepdims=keepdims))


@given(
    st.integers(-2, 2).filter(lambda x: x != 0)
    | st.sampled_from([-np.inf, np.inf, "fro", "nuc"])
)
def test_matrix_norm_raises(ord):
    with pytest.raises(NotImplementedError):
        t = mg.arange(4.0).reshape((2, 2))
        mg.linalg.norm(t, ord=ord)

    with pytest.raises(NotImplementedError):
        t = mg.arange(12.0).reshape((3, 2, 2))
        mg.linalg.norm(t, ord=ord, axis=(0, 1))


def test_ord_0_raises():
    with pytest.raises(NotImplementedError):
        t = mg.arange(4.0)
        mg.linalg.norm(t, ord=0)


def test_manual_norm_with_tuple_axis():
    pass


@settings(max_examples=1000)
@fwdprop_test_factory(
    mygrad_func=mg.linalg.norm,
    true_func=np.linalg.norm,
    num_arrays=1,
    kwargs=dict(axis=axis_strat, ord=ord_strat, keepdims=keepdims_strat),
    assumptions=lambda x, axis, ord, keepdims: np.all(np.abs(x) > 1e-20),
)
def test_norm_fwd():
    pass


@backprop_test_factory(
    mygrad_func=mg.linalg.norm,
    true_func=manual_norm_2,
    assumptions=lambda x, axis, ord, keepdims: x.ndim > 0 and np.all(np.abs(x) > 1e-20),
    num_arrays=1,
    kwargs=dict(
        axis=partial(axis_strat, permit_none=False), ord=2, keepdims=keepdims_strat
    ),
    vary_each_element=True,
)
def test_norm_bkwd_ord_2():
    pass


@pytest.mark.parametrize("ord", [-1.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, -np.inf, np.inf])
@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=1, min_side=0),
        dtype=float,
        elements=st.floats(-1e9, 1e9).filter(lambda x: np.abs(x) > 1e-6),
    ),
    data=st.data(),
)
def test_norm_backward(x, data, ord):
    if np.isinf(ord) and x.size == 0:
        # raises for numpy.linalg.norm too
        assume(False)
    p = ord
    keepdims = data.draw(keepdims_strat(x), label="keepdims")
    axis = data.draw(axis_strat(x, permit_none=False), label="axis")
    t1 = mg.tensor(x)
    t2 = mg.tensor(x.copy())

    o1 = mg.linalg.norm(t1, axis=axis, keepdims=keepdims, ord=p)
    o1.backward()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o2 = manual_norm(t2, axis=axis, keepdims=keepdims, ord=p)
        o2.backward()

    assert_allclose(o1, o2)
    assert_allclose(t1.grad, t2.grad, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("ord", [-1.0, 0.5, 1.0, 1.25, 1.5, 2.0, 2.5, -np.inf, np.inf])
@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=0),
        dtype=float,
        elements=st.floats(-1e9, 1e9).filter(lambda x: np.abs(x) > 1e-6),
    ),
    data=st.data(),
)
def test_norm_backward_1d(x, data, ord):
    if np.isinf(ord) and x.size == 0:
        # raises for numpy.linalg.norm too
        assume(False)
    p = ord
    keepdims = data.draw(keepdims_strat(x), label="keepdims")
    axis = data.draw(axis_strat(x, permit_none=True), label="axis")
    if axis == (None,):
        axis = None
    t1 = mg.tensor(x)
    t2 = mg.tensor(x.copy())

    o1 = mg.linalg.norm(t1, axis=axis, keepdims=keepdims, ord=p)
    o1.backward()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o2 = manual_norm(t2, axis=axis, keepdims=keepdims, ord=p)
        o2.backward()

    assert_allclose(o1, o2)
    assert_allclose(t1.grad, t2.grad, atol=1e-7, rtol=1e-7)


def test_nan_to_num_behavior():
    x = mg.tensor([[1.0, 2.0, 3.0], [1.0, 0.0, 0.0]])
    y = x.copy()
    z = x.copy()

    mg.linalg.norm(x, axis=1, nan_to_num=False).backward()
    mg.linalg.norm(y, axis=1, nan_to_num=True).backward()
    mg.linalg.norm(z, axis=1).backward()  # default behavior should be `nan_to_num=True`

    assert np.isnan(x.grad).sum() == 2
    assert_allclose(np.nan_to_num(x.grad), y.grad)
    assert_allclose(z.grad, y.grad)
