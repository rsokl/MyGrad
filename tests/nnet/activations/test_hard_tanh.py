import numpy as np
from hypothesis import given
import hypothesis.strategies as st

from mygrad.nnet.activations import hard_tanh
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@given(
    minimum=st.floats(allow_infinity=False, allow_nan=False),
    maximum=st.floats(allow_infinity=False, allow_nan=False),
)
def test_relu_fwd(minimum, maximum):
    @fwdprop_test_factory(
        mygrad_func=hard_tanh,
        true_func=lambda x: np.maximum(np.minimum(x, 1), -1),
        num_arrays=1,
    )
    def wrapped_test():
        pass


@given(
    minimum=st.floats(allow_infinity=False, allow_nan=False),
    maximum=st.floats(allow_infinity=False, allow_nan=False),
)
def test_relu_bkwd(minimum, maximum):
    @backprop_test_factory(
        mygrad_func=hard_tanh,
        true_func=lambda x: np.maximum(np.minimum(x, 1), -1),
        num_arrays=1,
    )
    def wrapped_test():
        pass
