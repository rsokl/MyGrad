from functools import partial

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose
from pytest import raises

import mygrad as mg
from mygrad import amax, amin, cumprod, cumsum, mean, prod, std, sum, var
from tests.utils.functools import add_constant_passthrough
from tests.utils.numerical_gradient import numerical_gradient_full

from ...custom_strategies import tensors, valid_axes
from ...wrappers.uber import (
    backprop_test_factory as backprop_test_factory,
    fwdprop_test_factory as fwdprop_test_factory,
)


def axis_arg(*arrs, min_dim=0):
    """Wrapper for passing valid-axis search strategy to test factory"""
    if arrs[0].ndim:
        return valid_axes(arrs[0].ndim, min_dim=min_dim)
    else:
        return st.just(tuple())


def single_axis_arg(*arrs):
    """Wrapper for passing valid-axis (single-value only)
    search strategy to test factory"""
    if arrs[0].ndim:
        return valid_axes(arrs[0].ndim, single_axis_only=True)
    else:
        return st.none()


def keepdims_arg(*arrs):
    """Wrapper for passing keep-dims strategy to test factory"""
    return st.booleans()


def ddof_arg(*arrs):
    """Wrapper for passing ddof strategy to test factory
    (argument for var and std)"""
    min_side = min(arrs[0].shape) if arrs[0].shape else 0
    return st.integers(0, min_side - 1) if min_side else st.just(0)


@fwdprop_test_factory(
    mygrad_func=amax,
    true_func=np.amax,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
)
def test_max_fwd():
    pass


@fwdprop_test_factory(
    mygrad_func=amax,
    true_func=mg.no_autodiff(mg.max, to_numpy=True),
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
)
def test_max_no_graph_track_fwd():
    """tests no-graph mode transitively
    mg.max == np.max
    mg.max == mg.max(no-grad)"""


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.amax),  # exercises __array_function__
    true_func=np.amax,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
    vary_each_element=True,
    index_to_unique={0: True},
    elements_strategy=st.integers,
)
def test_max_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=amin,
    true_func=np.amin,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
)
def test_min_fwd():
    pass


@fwdprop_test_factory(
    mygrad_func=amin,
    true_func=mg.no_autodiff(mg.min, to_numpy=True),
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
)
def test_min_no_graph_track_fwd():
    """tests no-graph mode transitively
    mg.min == np.min
    mg.min == mg.min(no-grad)"""


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.amin),  # exercises __array_function__
    true_func=np.amin,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
    vary_each_element=True,
    index_to_unique={0: True},
    elements_strategy=st.integers,
)
def test_min_bkwd():
    pass


def test_min_max_aliases():
    assert mg.max == amax
    assert mg.min == amin


@fwdprop_test_factory(
    mygrad_func=sum,
    true_func=np.sum,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
)
def test_sum_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.sum),  # exercises __array_function__
    true_func=np.sum,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
    vary_each_element=True,
    atol=1e-5,
)
def test_sum_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=mean,
    true_func=np.mean,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
)
def test_mean_fwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.mean),  # exercises __array_function__
    true_func=np.mean,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
    index_to_bnds={0: (-10, 10)},
    vary_each_element=True,
)
def test_mean_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=var,
    true_func=np.var,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg, ddof=ddof_arg),
)
@pytest.mark.filterwarnings("ignore: Degrees of freedom")
@pytest.mark.filterwarnings("ignore: invalid value encountered in true_divide")
def test_var_fwd():
    pass


def _var(x, keepdims=False, axis=None, ddof=0):
    """Defines variance without using abs. Permits use of
    complex-step numerical derivative."""
    x = np.asarray(x)

    def mean(y, keepdims=False, axis=None, ddof=0):
        if isinstance(axis, int):
            axis = (axis,)
        N = y.size if axis is None else np.prod([y.shape[i] for i in axis])
        return y.sum(keepdims=keepdims, axis=axis) / (N - ddof)

    return mean(
        (x - x.mean(axis=axis, keepdims=True)) ** 2,
        keepdims=keepdims,
        axis=axis,
        ddof=ddof,
    )


@fwdprop_test_factory(
    mygrad_func=var,
    true_func=_var,
    num_arrays=1,
    kwargs=dict(
        axis=partial(axis_arg, min_dim=1), keepdims=keepdims_arg, ddof=ddof_arg
    ),
)
def test_custom_var_fwd():
    pass


# test keepdims=True/False explicitly to deal with coverage issues
@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.var),  # exercises __array_function__,
    true_func=_var,
    num_arrays=1,
    kwargs=dict(axis=partial(axis_arg, min_dim=1), keepdims=False, ddof=ddof_arg),
    vary_each_element=True,
    index_to_bnds={0: (-10, 10)},
)
def test_var_bkwd_without_keepdims():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.var),  # exercises __array_function__,,
    true_func=_var,
    num_arrays=1,
    kwargs=dict(axis=partial(axis_arg, min_dim=1), keepdims=True, ddof=ddof_arg),
    vary_each_element=True,
    index_to_bnds={0: (-10, 10)},
)
def test_var_bkwd_with_keepdims():
    pass


@given(
    x=tensors(
        dtype=np.float64,
        shape=hnp.array_shapes(),
        elements=st.floats(allow_infinity=False, allow_nan=False),
    )
)
def test_var_no_axis_fwd(x: mg.Tensor):
    o = mg.var(x, axis=())
    assert np.all(o == mg.zeros_like(x))


@given(
    x=tensors(
        dtype=np.float64,
        shape=hnp.array_shapes(),
        elements=st.floats(allow_infinity=False, allow_nan=False),
        constant=False,
    )
)
def test_var_no_axis_bkwrd(x: mg.Tensor):
    np.var(x, axis=()).backward()
    assert np.all(x.grad == mg.zeros_like(x))


@fwdprop_test_factory(
    mygrad_func=std,
    true_func=np.std,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg, ddof=ddof_arg),
)
@pytest.mark.filterwarnings("ignore: Degrees of freedom")
@pytest.mark.filterwarnings("ignore: invalid value encountered in true_divide")
def test_std_fwd():
    pass


def _std(x, keepdims=False, axis=None, ddof=0):
    """Defines standard dev without using abs. Permits use of
    complex-step numerical derivative."""
    x = np.asarray(x)

    def mean(y, keepdims=False, axis=None, ddof=0):
        if isinstance(axis, int):
            axis = (axis,)
        N = y.size if axis is None else np.prod([y.shape[i] for i in axis])
        return y.sum(keepdims=keepdims, axis=axis) / (N - ddof)

    return np.sqrt(
        mean(
            (x - x.mean(axis=axis, keepdims=True)) ** 2,
            keepdims=keepdims,
            axis=axis,
            ddof=ddof,
        )
    )


@fwdprop_test_factory(
    mygrad_func=std,
    true_func=_std,
    num_arrays=1,
    kwargs=dict(
        axis=partial(axis_arg, min_dim=1), keepdims=keepdims_arg, ddof=ddof_arg
    ),
)
def test_custom_std_fwd():
    pass


def _assume(*arrs, **kwargs):
    return all(i > 1 for i in arrs[0].shape)


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.std),  # exercises __array_function__,,
    true_func=_std,
    num_arrays=1,
    kwargs=dict(
        axis=partial(axis_arg, min_dim=1), keepdims=keepdims_arg, ddof=ddof_arg
    ),
    vary_each_element=True,
    index_to_bnds={0: (-10, 10)},
    elements_strategy=st.integers,
    index_to_unique={0: True},
    assumptions=_assume,
)
def test_std_bkwd():
    pass


@given(
    x=tensors(
        dtype=np.float64,
        shape=hnp.array_shapes(),
        elements=st.floats(allow_infinity=False, allow_nan=False),
    )
)
def test_std_no_axis_fwd(x):
    o = mg.std(x, axis=())
    assert np.all(o == mg.zeros_like(x))


@given(
    x=tensors(
        dtype=np.float64,
        shape=hnp.array_shapes(),
        elements=st.floats(allow_infinity=False, allow_nan=False),
        constant=False,
    )
)
def test_std_no_axis_bkwrd(x):
    mg.std(x, axis=()).backward()
    assert np.all(x.grad == mg.zeros_like(x))


@fwdprop_test_factory(
    mygrad_func=prod,
    true_func=np.prod,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
)
def test_prod_fwd():
    pass


@backprop_test_factory(
    mygrad_func=prod,
    true_func=np.prod,
    num_arrays=1,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
    vary_each_element=True,
    index_to_bnds={0: (-2, 2)},
)
def test_prod_bkwd():
    pass


@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.prod),  # exercises __array_function__,,
    true_func=np.prod,
    num_arrays=1,
    elements_strategy=st.integers,
    kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
    vary_each_element=True,
    index_to_bnds={0: (-2, 2)},
)
def test_multi_zero_prod_bkwd():
    """Drives tests cases with various configurations of zeros"""


def test_int_axis_cumprod():
    """check if numpy cumprod begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    "`np.cumprod` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        np.cumprod(x, axis=(0, 1))

    "`mygrad.cumprod` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        cumprod(x, axis=(0, 1))


@fwdprop_test_factory(
    mygrad_func=cumprod,
    true_func=np.cumprod,
    num_arrays=1,
    kwargs=dict(axis=single_axis_arg),
)
def test_cumprod_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.cumprod),  # exercises __array_function__,
    true_func=np.cumprod,
    num_arrays=1,
    kwargs=dict(axis=single_axis_arg),
    vary_each_element=True,
    index_to_bnds={0: (-2, 2)},
)
def test_cumprod_bkwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=cumprod,
    true_func=np.cumprod,
    num_arrays=1,
    kwargs=dict(axis=single_axis_arg),
    vary_each_element=True,
    index_to_bnds={0: (-0.5, 0.5)},
    index_to_unique={0: True},
    index_to_arr_shapes={0: hnp.array_shapes(max_side=5, max_dims=4)},
)
def test_cumprod_bkwd2():
    pass


def test_int_axis_cumsum():
    """check if numpy cumsum begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    "`np.cumsum` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        np.cumsum(x, axis=(0, 1))

    "`mygrad.cumsum` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        cumsum(x, axis=(0, 1))


@fwdprop_test_factory(
    mygrad_func=cumsum,
    true_func=np.cumsum,
    num_arrays=1,
    kwargs=dict(axis=single_axis_arg),
)
def test_cumsum_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=add_constant_passthrough(np.cumsum),  # exercises __array_function__,,
    true_func=np.cumsum,
    num_arrays=1,
    kwargs=dict(axis=single_axis_arg),
    vary_each_element=True,
    index_to_bnds={0: (-2, 2)},
    atol=1e-5,
)
def test_cumsum_bkwd():
    pass


def test_patch_coverage_on_var():
    # this path coverage through var was flaky
    x = mg.arange(6.0).reshape((2, 3))
    o = mg.var(x, axis=1, keepdims=False)
    o.backward()
    (num_grad,) = numerical_gradient_full(
        _var, x.data, kwargs=dict(axis=1, keepdims=False), back_grad=np.ones(o.shape)
    )
    assert_allclose(x.grad, num_grad)
