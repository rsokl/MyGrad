from functools import partial

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import mygrad as mg
from mygrad.math.arithmetic.ops import Positive, Square

from ..wrappers.uber import backprop_test_factory, fwdprop_test_factory


def custom_pow(x, p, constant=False):
    out = x ** p
    if isinstance(out, mg.Tensor):
        out._constant = constant
    return out


def any_scalar(*args, p):
    return st.sampled_from([int(p), float(p), np.array(p)])


@pytest.mark.parametrize("power, op", [(1, Positive), (2, Square)])
def test_pow_uses_special_case(power, op):
    @given(exp=st.sampled_from([int(power), float(power), np.array(power)]))
    def wrapped_func(exp):
        out = mg.arange(2) ** exp
        assert isinstance(out.creator, op)

    wrapped_func()


@fwdprop_test_factory(
    mygrad_func=custom_pow,
    true_func=custom_pow,
    num_arrays=1,
    kwargs={"p": partial(any_scalar, p=1)},
    permit_0d_array_as_float=False,
)
def test_pow_1_fwd():
    pass


@backprop_test_factory(
    mygrad_func=custom_pow,
    true_func=custom_pow,
    num_arrays=1,
    kwargs={"p": partial(any_scalar, p=1)},
)
def test_pow_1_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=custom_pow,
    true_func=custom_pow,
    num_arrays=1,
    permit_0d_array_as_float=False,
    kwargs={"p": partial(any_scalar, p=2)},
)
def test_pow_2_fwd():
    pass


@backprop_test_factory(
    mygrad_func=custom_pow,
    true_func=custom_pow,
    num_arrays=1,
    kwargs={"p": partial(any_scalar, p=2)},
)
def test_pow_2_bkwd():
    pass
