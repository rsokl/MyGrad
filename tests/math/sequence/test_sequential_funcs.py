from functools import partial

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from pytest import raises

import mygrad
from mygrad import amax, amin, cumprod, cumsum, mean, prod, std, sum, var

from ...custom_strategies import valid_axes
from ...wrappers.uber import (
    backprop_test_factory as backprop_test_factory,
    fwdprop_test_factory as fwdprop_test_factory,
)


def axis_arg(*arrs, min_dim=0):
    """ Wrapper for passing valid-axis search strategy to test factory"""
    return valid_axes(arrs[0].ndim, min_dim=min_dim)


def single_axis_arg(*arrs):
    """ Wrapper for passing valid-axis (single-value only)
    search strategy to test factory"""
    return valid_axes(arrs[0].ndim, single_axis_only=True)


def keepdims_arg(*arrs):
    """ Wrapper for passing keep-dims strategy to test factory"""
    return st.booleans()


def ddof_arg(*arrs):
    """ Wrapper for passing ddof strategy to test factory
    (argument for var and std)"""
    return st.integers(0, min(arrs[0].shape) - 1)


@fwdprop_test_factory(mygrad_func=amax, true_func=np.amax, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_max_fwd():
    pass


@backprop_test_factory(mygrad_func=amax, true_func=np.amax, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, index_to_unique={0: True},
                       elements_strategy=st.integers)
def test_max_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=amin, true_func=np.amin, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_min_fwd():
    pass


@backprop_test_factory(mygrad_func=amin, true_func=np.amin, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, index_to_unique={0: True},
                       elements_strategy=st.integers)
def test_min_bkwd():
    pass


def test_min_max_aliases():
    assert mygrad.max == amax
    assert mygrad.min == amin


@fwdprop_test_factory(mygrad_func=sum, true_func=np.sum, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_sum_fwd():
    pass


@backprop_test_factory(mygrad_func=sum, true_func=np.sum, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, atol=1e-5)
def test_sum_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=mean, true_func=np.mean, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_mean_fwd():
    pass


@backprop_test_factory(mygrad_func=mean, true_func=np.mean, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       index_to_bnds={0: (-10, 10)},
                       vary_each_element=True)
def test_mean_bkwd():
    pass


@fwdprop_test_factory(mygrad_func=var, true_func=np.var, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                                  ddof=ddof_arg))
def test_var_fwd():
    pass


def _var(x, keepdims=False, axis=None, ddof=0):
    """Defines variance without using abs. Permits use of
    complex-step numerical derivative."""
    def mean(y, keepdims=False, axis=None, ddof=0):
        N = y.size if axis is None else np.prod([y.shape[i] for i in axis])
        return y.sum(keepdims=keepdims, axis=axis) / (N - ddof)

    return mean((x - x.mean(axis=axis, keepdims=True))**2,
                keepdims=keepdims, axis=axis, ddof=ddof)


@fwdprop_test_factory(mygrad_func=var, true_func=_var, num_arrays=1,
                      kwargs=dict(axis=partial(axis_arg, min_dim=1), keepdims=keepdims_arg,
                                  ddof=ddof_arg))
def test_custom_var_fwd():
    pass


@backprop_test_factory(mygrad_func=var, true_func=_var, num_arrays=1,
                       kwargs=dict(axis=partial(axis_arg, min_dim=1), keepdims=keepdims_arg,
                                   ddof=ddof_arg),
                       vary_each_element=True, index_to_bnds={0: (-10, 10)})
def test_var_bkwd():
    pass


@given(x=hnp.arrays(dtype=np.float,
                    shape=hnp.array_shapes(),
                    elements=st.floats(allow_infinity=False,
                                       allow_nan=False)))
def test_var_no_axis_fwd(x):
    import mygrad as mg
    x = mg.Tensor(x, constant=False)
    o = mg.var(x, axis=())
    assert np.all(o.data == np.zeros_like(x.data))


@given(x=hnp.arrays(dtype=np.float,
                    shape=hnp.array_shapes(),
                    elements=st.floats(allow_infinity=False,
                                       allow_nan=False)))
def test_var_no_axis_bkwrd(x):
    import mygrad as mg
    x = mg.Tensor(x, constant=False)
    mg.var(x, axis=()).backward()
    assert np.all(x.grad == np.zeros_like(x.data))


@fwdprop_test_factory(mygrad_func=std, true_func=np.std, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg,
                                  ddof=ddof_arg))
def test_std_fwd():
    pass


def _std(x, keepdims=False, axis=None, ddof=0):
    """Defines standard dev without using abs. Permits use of
    complex-step numerical derivative."""
    def mean(y, keepdims=False, axis=None, ddof=0):
        N = y.size if axis is None else np.prod([y.shape[i] for i in axis])
        return y.sum(keepdims=keepdims, axis=axis) / (N - ddof)

    return np.sqrt(mean((x - x.mean(axis=axis, keepdims=True))**2,
                        keepdims=keepdims, axis=axis, ddof=ddof))


@fwdprop_test_factory(mygrad_func=std, true_func=_std, num_arrays=1,
                      kwargs=dict(axis=partial(axis_arg, min_dim=1), keepdims=keepdims_arg,
                                  ddof=ddof_arg))
def test_custom_std_fwd():
    pass


def _assume(*arrs, **kwargs):
    return all(i > 1 for i in arrs[0].shape)


@backprop_test_factory(mygrad_func=std, true_func=_std, num_arrays=1,
                       kwargs=dict(axis=partial(axis_arg, min_dim=1), keepdims=keepdims_arg,
                                   ddof=ddof_arg),
                       vary_each_element=True, index_to_bnds={0: (-10, 10)},
                       elements_strategy=st.integers,
                       index_to_unique={0: True},
                       assumptions=_assume)
def test_std_bkwd():
    pass


@given(x=hnp.arrays(dtype=np.float,
                    shape=hnp.array_shapes(),
                    elements=st.floats(allow_infinity=False,
                                       allow_nan=False)))
def test_std_no_axis_fwd(x):
    import mygrad as mg
    x = mg.Tensor(x, constant=False)
    o = mg.std(x, axis=())
    assert np.all(o.data == np.zeros_like(x.data))


@given(x=hnp.arrays(dtype=np.float,
                    shape=hnp.array_shapes(),
                    elements=st.floats(allow_infinity=False,
                                       allow_nan=False)))
def test_std_no_axis_bkwrd(x):
    import mygrad as mg
    x = mg.Tensor(x, constant=False)
    mg.std(x, axis=()).backward()
    assert np.all(x.grad == np.zeros_like(x.data))


@fwdprop_test_factory(mygrad_func=prod, true_func=np.prod, num_arrays=1,
                      kwargs=dict(axis=axis_arg, keepdims=keepdims_arg))
def test_prod_fwd():
    pass


@backprop_test_factory(mygrad_func=prod, true_func=np.prod, num_arrays=1,
                       kwargs=dict(axis=axis_arg, keepdims=keepdims_arg),
                       vary_each_element=True, index_to_bnds={0: (-2, 2)})
def test_prod_bkwd():
    pass


def test_int_axis_cumprod():
    """check if numpy cumprod begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]])
    "`np.cumprod` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        np.cumprod(x, axis=(0, 1))

    "`mygrad.cumprod` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        cumprod(x, axis=(0, 1))


@fwdprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod, num_arrays=1,
                      kwargs=dict(axis=single_axis_arg))
def test_cumprod_fwd(): pass


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod, num_arrays=1,
                       kwargs=dict(axis=single_axis_arg),
                       vary_each_element=True, index_to_bnds={0: (-2, 2)})
def test_cumprod_bkwd():
    pass


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumprod, true_func=np.cumprod, num_arrays=1,
                       kwargs=dict(axis=single_axis_arg),
                       vary_each_element=True, index_to_bnds={0: (-.5, .5)},
                       index_to_unique={0: True},
                       index_to_arr_shapes={0: hnp.array_shapes(max_side=5, max_dims=4)})
def test_cumprod_bkwd2():
    pass


def test_int_axis_cumsum():
    """check if numpy cumsum begins to support tuples for the axis argument"""

    x = np.array([[1, 1, 0],
                  [1, 1, 0],
                  [1, 1, 0]])
    "`np.cumsum` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        np.cumsum(x, axis=(0, 1))

    "`mygrad.cumsum` is expected to raise a TypeError "
    "when it is provided a tuple of axes."
    with raises(TypeError):
        cumsum(x, axis=(0, 1))


@fwdprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum, num_arrays=1,
                      kwargs=dict(axis=single_axis_arg))
def test_cumsum_fwd():
    pass


@settings(deadline=None)
@backprop_test_factory(mygrad_func=cumsum, true_func=np.cumsum, num_arrays=1,
                       kwargs=dict(axis=single_axis_arg),
                       vary_each_element=True, index_to_bnds={0: (-2, 2)},
                       atol=1e-5)
def test_cumsum_bkwd():
    pass
