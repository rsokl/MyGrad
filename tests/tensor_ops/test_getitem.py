from mygrad.tensor_base import Tensor
import numpy as np
from numpy.testing import assert_allclose

from ..wrappers.uber import fwdprop_test_factory, backprop_test_factory
from ..custom_strategies import basic_index, adv_integer_index
import hypothesis.extra.numpy as hnp


def test_getitem():
    x = Tensor([1, 2, 3])
    a, b, c = x
    f = 2*a + 3*b + 4*c
    f.backward()

    assert a.data == 1
    assert b.data == 2
    assert c.data == 3
    assert f.data == 20

    assert_allclose(a.grad, np.array(2))
    assert_allclose(b.grad, np.array(3))
    assert_allclose(c.grad, np.array(4))
    assert_allclose(x.grad, np.array([2, 3, 4]))


def get_item(*arrs, index):
    return arrs[0][index]


def basic_index_wrap(*arrs):
    return basic_index(arrs[0].shape)


def adv_index_int_wrap(*arrs):
    return adv_integer_index(arrs[0].shape)


def adv_index_bool_wrap(*arrs):
    return hnp.arrays(shape=arrs[0].shape, dtype=bool)


@fwdprop_test_factory(mygrad_func=get_item, true_func=get_item, num_arrays=1,
                      index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
                      kwargs=dict(index=basic_index_wrap))
def test_getitem_basicindex_fwdprop():
    pass


@backprop_test_factory(mygrad_func=get_item, true_func=get_item, num_arrays=1,
                       index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
                       kwargs=dict(index=basic_index_wrap))
def test_getitem_basicindex_bkwdprop():
    pass


@fwdprop_test_factory(mygrad_func=get_item, true_func=get_item, num_arrays=1,
                      index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
                      kwargs=dict(index=adv_index_int_wrap))
def test_getitem_advindex_int_fwdprop():
    pass


@backprop_test_factory(mygrad_func=get_item, true_func=get_item, num_arrays=1,
                       index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
                       kwargs=dict(index=adv_index_int_wrap))
def test_getitem_advindex_int_bkwdprop():
    pass


@fwdprop_test_factory(mygrad_func=get_item, true_func=get_item, num_arrays=1,
                      index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
                      kwargs=dict(index=adv_index_bool_wrap))
def test_getitem_advindex_int_fwdprop():
    pass


@backprop_test_factory(mygrad_func=get_item, true_func=get_item, num_arrays=1,
                       index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
                       kwargs=dict(index=adv_index_bool_wrap))
def test_getitem_advindex_int_bkwdprop():
    pass
