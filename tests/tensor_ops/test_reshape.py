from functools import partial
from numbers import Number

import numpy as np
import pytest

from mygrad import reshape
from mygrad.tensor_base import Tensor

from ..custom_strategies import valid_shapes
from ..wrappers.uber import backprop_test_factory, fwdprop_test_factory


def positional_reshape(arr, newshape, reshaper, **kwargs):
    return reshaper(arr, newshape, **kwargs)


def keyword_reshape(arr, newshape, reshaper, **kwargs):
    return reshaper(arr, newshape=newshape, **kwargs)


def unpacked_reshape(arr, newshape, reshaper, **kwargs):
    return reshaper(arr, *newshape, **kwargs)


def method_tuple_reshape(arr, newshape, reshaper, **kwargs):
    to_array = np.asarray if reshaper is np.reshape else Tensor
    if isinstance(arr, Number):
        arr = to_array(arr)
    return arr.reshape(newshape, **kwargs)


def method_unpacked_reshape(arr, newshape, reshaper, **kwargs):
    to_array = np.asarray if reshaper is np.reshape else Tensor
    if isinstance(arr, Number):
        arr = to_array(arr)

    return (
        arr.reshape(*newshape, **kwargs)
        if isinstance(newshape, tuple)
        else arr.reshape(newshape, **kwargs)
    )


@pytest.mark.parametrize(
    "reshape_type",
    [
        positional_reshape,
        keyword_reshape,
        method_tuple_reshape,
        method_unpacked_reshape,
    ],
)
def test_reshape_fwd(reshape_type):
    @fwdprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size)),
    )
    def test_fwd():
        pass

    test_fwd()


@pytest.mark.parametrize(
    "reshape_type",
    [
        positional_reshape,
        keyword_reshape,
        method_tuple_reshape,
        method_unpacked_reshape,
    ],
)
def test_reshape_bkwd(reshape_type):
    @backprop_test_factory(
        mygrad_func=partial(reshape_type, reshaper=reshape),
        true_func=partial(reshape_type, reshaper=np.reshape),
        num_arrays=1,
        kwargs=dict(newshape=lambda arrs: valid_shapes(arrs.size)),
        vary_each_element=True,
        atol=1e-9,
    )
    def test_bkwd():
        pass

    test_bkwd()


@pytest.mark.parametrize(
    "bad_input", [tuple(), ((2,), 2), (((2,), 2)), (2, (2,)), ((2, (2,)))]
)
def test_input_validation(bad_input):
    x = np.array([1, 2])

    with pytest.raises(TypeError):
        x.reshape(*bad_input)
