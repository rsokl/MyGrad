import sys

import hypothesis.strategies as st
import numpy as np
import pytest

from mygrad import Tensor
from mygrad.nnet.activations import elu
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@pytest.mark.parametrize("alpha", (None, 1j))
def test_input_validation(alpha):
    with pytest.raises(TypeError):
        elu(2, alpha=alpha)


def _np_elu(x, alpha):
    if isinstance(alpha, Tensor):
        alpha = alpha.data
    return np.where(x < 0, alpha * (np.exp(x) - 1), x)


_reasonable_floats = st.floats(-100, 100)


@fwdprop_test_factory(
    mygrad_func=elu,
    true_func=_np_elu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max) / 100, np.log(sys.float_info.max) / 100)},
    kwargs={"alpha": lambda x: _reasonable_floats | _reasonable_floats.map(np.array) | _reasonable_floats.map(Tensor)},
)
def test_elu_fwd():
    pass


@backprop_test_factory(
    mygrad_func=elu,
    true_func=_np_elu,
    num_arrays=1,
    index_to_bnds={0: (-np.log(sys.float_info.max) / 100, np.log(sys.float_info.max) / 100)},
    kwargs={"alpha": lambda x: _reasonable_floats | _reasonable_floats.map(np.array) | _reasonable_floats.map(Tensor)},
)
def test_elu_bkwd():
    pass
