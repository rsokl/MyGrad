import functools
from typing import List, Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import mygrad as mg
import mygrad.math.misc.funcs
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def matmul_wrapper(*args, constant=None):
    return mygrad.math.misc.funcs.multi_matmul(args, constant=constant)


def multi_matmul_slow(*arrays, **kwargs):
    return functools.reduce(np.matmul, arrays)


@given(st.lists(st.just(mg.Tensor([0.0, 1.0])), min_size=0, max_size=1))
def test_input_validation_too_few_tensors(tensors: List[mg.Tensor]):
    """multi_matmul requires at least two input-tensors"""
    with pytest.raises(ValueError):
        mygrad.math.misc.funcs.multi_matmul(tensors)


@given(st.lists(hnp.array_shapes(min_dims=1, min_side=2, max_side=2), min_size=2))
def test_input_validation_large_dimensionality(shapes: List[Tuple[int, ...]]):
    """multi_matmul only operates on 1D and 2D tensors"""
    if all((1 <= len(x) <= 2) for x in shapes):
        shapes[0] = (1, 1, *shapes[0])
    tensors = [mg.ones(shape=shape) for shape in shapes]
    with pytest.raises(ValueError):
        mygrad.math.misc.funcs.multi_matmul(tensors)


@pytest.mark.parametrize(
    "signature",
    (
        "(a?,b),(b,e?)->(a?,e?)",
        "(a?,b),(b,c),(c,e?)->(a?,e?)",
        "(a?,b),(b,c),(c,d),(d,e?)->(a?,e?)",
    ),
)
def test_matmul_fwd(signature):
    @fwdprop_test_factory(
        mygrad_func=matmul_wrapper,
        true_func=multi_matmul_slow,
        shapes=hnp.mutually_broadcastable_shapes(signature=signature, max_dims=0),
        default_bnds=(-10, 10),
        atol=1e-5,
        rtol=1e-5,
        internal_view_is_ok=True,
    )
    def test_runner():
        pass

    test_runner()


@pytest.mark.parametrize(
    "signature",
    (
        "(a?,b),(b,e?)->(a?,e?)",
        "(a?,b),(b,c),(c,e?)->(a?,e?)",
        "(a?,b),(b,c),(c,d),(d,e?)->(a?,e?)",
    ),
)
def test_matmul_bkwd(signature):
    @backprop_test_factory(
        mygrad_func=matmul_wrapper,
        true_func=multi_matmul_slow,
        shapes=hnp.mutually_broadcastable_shapes(signature=signature, max_dims=0),
        default_bnds=(-10, 10),
        atol=1e-5,
        rtol=1e-5,
        vary_each_element=True,
    )
    def test_runner():
        pass

    test_runner()
