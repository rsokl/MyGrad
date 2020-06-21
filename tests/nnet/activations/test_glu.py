import sys

from hypothesis import assume, settings, HealthCheck
import hypothesis.strategies as st
import numpy as np

from mygrad.nnet.activations import glu
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _np_glu(x, dim):
    first_idx = list(slice(None) for _ in x.shape)
    second_idx = list(slice(None) for _ in x.shape)
    first_idx[dim] = slice(0, x.shape[dim] // 2)
    second_idx[dim] = slice(x.shape[dim] // 2, None)

    first_half = x[tuple(first_idx)]
    second_half = x[tuple(second_idx)]

    return first_half * (1 / (1 + np.exp(-second_half)))


@st.composite
def _dim_strategy(draw, arr):
    assume(any(not x % 2 for x in arr.shape))
    return draw(st.sampled_from([i for i, dim in enumerate(arr.shape) if not dim % 2]))


@settings(suppress_health_check=(HealthCheck.filter_too_much,))
@fwdprop_test_factory(
    mygrad_func=glu,
    true_func=_np_glu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max), np.log(sys.float_info.max))},
    kwargs={"dim": lambda x: _dim_strategy(x),},
    assumptions=lambda arr, dim: any(not x % 2 for x in arr.shape),
)
def test_glu_fwd():
    pass


@backprop_test_factory(
    mygrad_func=glu,
    true_func=_np_glu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max), np.log(sys.float_info.max))},
    kwargs={"dim": lambda x: _dim_strategy(x),},
    assumptions=lambda arr, dim: any(not x % 2 for x in arr.shape),
    vary_each_element=True,
)
def test_glu_bkwd():
    pass
