from typing import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note
from numpy.testing import assert_array_equal

from mygrad import Tensor
from tests.custom_strategies import tensors


@given(x=tensors(include_grad=True))
def test_involving_a_tensor_in_a_graph_nulls_its_gradient(x: Tensor):
    assert x.grad is not None
    _ = +x
    assert x.grad is None
    assert x._ops is not None


@given(x=tensors(elements=st.floats(-100, 100), include_grad=st.booleans()))
def test_backprop_clears_graph(x: Tensor):
    for num_fwd_pass in range(2):
        note(f"Forward-pass iteration: {num_fwd_pass}")
        y = 2 * x
        f = y + x ** 2
        f[...] = f[...]
        x ** 3  # no-op
        f.backward()
        if not x.constant:
            assert_array_equal(
                f.grad, np.ones_like(f), err_msg="f.grad is not the expected value"
            )
            assert_array_equal(
                y.grad, np.ones_like(y), err_msg="y.grad is not the expected value"
            )
            assert_array_equal(
                x.grad, 2 + 2 * x.data, err_msg="x.grad is not the expected value"
            )
        else:
            assert f.grad is None
            assert y.grad is None
            assert x.grad is None
        assert not f._ops
        assert f.creator is None

        assert not y._ops
        assert y.creator is None

        assert not x._ops
        assert x.creator is None
