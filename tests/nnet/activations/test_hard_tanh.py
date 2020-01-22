import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from mygrad.nnet.activations import hard_tanh
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@fwdprop_test_factory(
    mygrad_func=hard_tanh,
    true_func=lambda x, lower_bound, upper_bound: np.maximum(
        np.minimum(x, upper_bound), lower_bound
    ),
    num_arrays=1,
    kwargs={
        "lower_bound": lambda x: st.floats(allow_infinity=False, allow_nan=False),
        "upper_bound": lambda x: st.floats(allow_infinity=False, allow_nan=False),
    },
)
def test_hard_tanh_fwd():
    pass


@backprop_test_factory(
    mygrad_func=hard_tanh,
    true_func=lambda x, lower_bound, upper_bound: np.maximum(
        np.minimum(x, upper_bound), lower_bound
    ),
    num_arrays=1,
    kwargs={
        "lower_bound": lambda x: st.floats(allow_infinity=False, allow_nan=False),
        "upper_bound": lambda x: st.floats(allow_infinity=False, allow_nan=False),
    },
)
def test_hard_tanh_bkwd():
    pass
