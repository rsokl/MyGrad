import numpy as np
import pytest

import hypothesis.strategies as st
from mygrad.nnet.activations import leaky_relu
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@pytest.mark.parametrize("slope", (None, 1j))
def test_input_validation(slope):
    with pytest.raises(TypeError):
        leaky_relu(2, slope=slope)


def _np_leaky_relu(x, slope):
    return np.maximum(x, 0) + slope * np.minimum(x, 0)


finite_floats = st.floats(allow_infinity=False, allow_nan=False, width=32)


@fwdprop_test_factory(
    mygrad_func=leaky_relu,
    true_func=_np_leaky_relu,
    num_arrays=1,
    kwargs={"slope": lambda x: finite_floats | finite_floats.map(np.array)},
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
    assumptions=_away_from_zero,
    kwargs={"slope": lambda x: finite_floats | finite_floats.map(np.array)},
)
def test_leaky_relu_bkwd():
    pass
