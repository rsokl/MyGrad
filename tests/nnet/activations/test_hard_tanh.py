import hypothesis.strategies as st
import numpy as np
import pytest

from mygrad.nnet.activations import hard_tanh
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@pytest.mark.parametrize("lower_bound, upper_bound", [(None, 1), (1, None)])
def test_input_validation(lower_bound, upper_bound):
    with pytest.raises(TypeError):
        hard_tanh(2, lower_bound=lower_bound, upper_bound=upper_bound)


finite_floats = st.floats(allow_infinity=False, allow_nan=False)


@fwdprop_test_factory(
    mygrad_func=hard_tanh,
    true_func=lambda x, lower_bound, upper_bound: np.maximum(
        np.minimum(x, upper_bound), lower_bound
    ),
    num_arrays=1,
    kwargs={
        "lower_bound": lambda x: finite_floats | finite_floats.map(np.array),
        "upper_bound": lambda x: finite_floats | finite_floats.map(np.array),
    },
)
def test_hard_tanh_fwd():
    pass


def assume_data_not_on_bounds(x, lower_bound, upper_bound):
    return np.all(np.logical_and(x != lower_bound, x != upper_bound))


@backprop_test_factory(
    mygrad_func=hard_tanh,
    true_func=lambda x, lower_bound, upper_bound: np.maximum(
        np.minimum(x, upper_bound), lower_bound
    ),
    num_arrays=1,
    kwargs={
        "lower_bound": lambda x: finite_floats | finite_floats.map(np.array),
        "upper_bound": lambda x: finite_floats | finite_floats.map(np.array),
    },
    assumptions=assume_data_not_on_bounds,
)
def test_hard_tanh_bkwd():
    pass
