import numpy as np
import pytest

import hypothesis.strategies as st
from mygrad.nnet.activations import leaky_relu
from mygrad import Tensor
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@pytest.mark.parametrize("slope", (None, 1j))
def test_input_validation(slope):
    with pytest.raises(TypeError):
        leaky_relu(2, slope=slope)


def _np_leaky_relu(x, slope):
    if isinstance(slope, Tensor):
        slope = slope.data
    return np.maximum(x, 0) + slope * np.minimum(x, 0)


_reasonable_floats = st.floats(-100, 100)


def _finite_params(arrs, slope):
    if isinstance(arrs, Tensor):
        arrs = arrs.data
    if isinstance(slope, Tensor):
        slope = slope.data
    return np.all(np.isfinite(slope * arrs))


@fwdprop_test_factory(
    mygrad_func=leaky_relu,
    true_func=_np_leaky_relu,
    num_arrays=1,
    kwargs={"slope": lambda x: _reasonable_floats | _reasonable_floats.map(np.array) | _reasonable_floats.map(Tensor)},
    assumptions=_finite_params,
)
def test_leaky_relu_fwd():
    pass


def _away_from_zero(*arrs, **kwargs):
    x = arrs[0]
    return np.all(np.abs(x.data) > 1e-8)


@backprop_test_factory(
    mygrad_func=leaky_relu,
    true_func=_np_leaky_relu,
    num_arrays=1,
    assumptions=lambda arrs, slope: _away_from_zero(arrs) and _finite_params(arrs, slope),
    kwargs={"slope": lambda x: _reasonable_floats | _reasonable_floats.map(np.array) | _reasonable_floats.map(Tensor)},
)
def test_leaky_relu_bkwd():
    pass
