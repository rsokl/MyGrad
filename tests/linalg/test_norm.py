from functools import partial
import pytest
from tests.wrappers.uber import fwdprop_test_factory, backprop_test_factory
from tests.custom_strategies import valid_axes
import mygrad as mg
import numpy as np
from hypothesis import given
import hypothesis.strategies as st


def axis_strat(*ts, permit_none=True):
    (t,) = ts
    if t.ndim == 0:
        return st.none()

    return valid_axes(
        t.ndim, single_axis_only=True, permit_int=True, permit_none=permit_none
    )


def ord_strat(*ts):
    (t,) = ts
    if t.ndim == 0 or t.ndim == 2:
        return st.none()

    return st.none() | st.integers(-2, 2).filter(lambda x: x != 0) | st.floats(0.1, 3)


def keepdims_strat(*ts):
    return st.booleans()


def manual_norm_1(x, ord=None, axis=None, keepdims=False):
    return np.sum(np.sqrt(x ** 2), axis=axis, keepdims=keepdims)


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


@fwdprop_test_factory(
    mygrad_func=mg.linalg.norm,
    true_func=np.linalg.norm,
    num_arrays=1,
    kwargs=dict(axis=axis_strat, ord=ord_strat),
)
def test_linalg_fwd():
    pass


@backprop_test_factory(
    mygrad_func=mg.linalg.norm,
    true_func=manual_norm_1,
    assumptions=lambda x, axis, ord, keepdims: x.ndim > 0,
    num_arrays=1,
    kwargs=dict(
        axis=partial(axis_strat, permit_none=False), ord=1, keepdims=keepdims_strat
    ),
    vary_each_element=True,
)
def test_linalg_bkwd_ord_1():
    pass


@backprop_test_factory(
    mygrad_func=mg.linalg.norm,
    true_func=manual_norm_2,
    assumptions=lambda x, axis, ord, keepdims: x.ndim > 0,
    num_arrays=1,
    kwargs=dict(
        axis=partial(axis_strat, permit_none=False), ord=2, keepdims=keepdims_strat
    ),
    vary_each_element=True,
)
def test_linalg_bkwd_ord_2():
    pass


@backprop_test_factory(
    mygrad_func=mg.linalg.norm,
    true_func=np.linalg.norm,
    num_arrays=1,
    kwargs=dict(axis=axis_strat, ord=ord_strat),
    vary_each_element=True,
)
def test_linalg_bkwd():
    pass
