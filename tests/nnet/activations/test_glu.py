import sys

from hypothesis import assume, given, settings, HealthCheck
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
import pytest

import mygrad as mg
from mygrad.nnet.activations import glu
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@pytest.mark.parametrize("axis", (None, 1j))
def test_input_validation(axis):
    with pytest.raises(TypeError):
        glu(2, axis=axis)


@given(arr=hnp.arrays(dtype=np.float32, shape=hnp.array_shapes()))
def test_bad_shape_dimension(arr):
    assume(any(x % 2 for x in arr.shape))
    idx = np.random.choice([i for i, axis in enumerate(arr.shape) if axis % 2]).item()
    with pytest.raises(ValueError):
        glu(arr, idx)


def _np_glu(x, axis):
    if isinstance(axis, (np.ndarray, mg.Tensor)):
        axis = axis.item()

    first_idx = list(slice(None) for _ in x.shape)
    second_idx = list(slice(None) for _ in x.shape)
    first_idx[axis] = slice(0, x.shape[axis] // 2)
    second_idx[axis] = slice(x.shape[axis] // 2, None)

    first_half = x[tuple(first_idx)]
    second_half = x[tuple(second_idx)]

    return first_half * (1 / (1 + np.exp(-second_half)))


@st.composite
def _axis_strategy(draw, arr):
    assume(any(not x % 2 for x in arr.shape))
    val = draw(st.sampled_from([i for i, axis in enumerate(arr.shape) if not axis % 2]))
    dtype = draw(st.sampled_from((np.array, mg.Tensor, int)))
    return dtype(val)


@fwdprop_test_factory(
    mygrad_func=glu,
    true_func=_np_glu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max), np.log(sys.float_info.max))},
    kwargs={"axis": lambda x: _axis_strategy(x),},
    assumptions=lambda arr, axis: any(not x % 2 for x in arr.shape),
)
def test_glu_fwd():
    pass


@backprop_test_factory(
    mygrad_func=glu,
    true_func=_np_glu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max), np.log(sys.float_info.max))},
    kwargs={"axis": lambda x: _axis_strategy(x)},
    assumptions=lambda arr, axis: any(not x % 2 for x in arr.shape),
    vary_each_element=True,
)
def test_glu_bkwd():
    pass
