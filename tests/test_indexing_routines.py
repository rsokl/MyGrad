import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import mygrad as mg
from mygrad import where
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def mygrad_where(x, y, condition, constant=None):
    return where(condition, x, y, constant=constant)


def numpy_where(x, y, condition, constant=None):
    return np.where(condition, x, y)


def condition_strat(*arrs):
    shape = np.broadcast(*arrs).shape
    return hnp.arrays(shape=hnp.broadcastable_shapes(shape=shape), dtype=bool)


@fwdprop_test_factory(
    mygrad_func=mygrad_where,
    true_func=numpy_where,
    kwargs=dict(condition=condition_strat),
    num_arrays=2,
)
def test_where_fwd():
    pass


@backprop_test_factory(
    mygrad_func=numpy_where,  # exercises __array_function__ override
    true_func=numpy_where,
    kwargs=dict(condition=condition_strat),
    num_arrays=2,
)
def test_where_bkwd():
    pass


@given(condition=st.from_type(type) | hnp.arrays(shape=hnp.array_shapes(), dtype=int))
@pytest.mark.filterwarnings("ignore: Calling nonzero on 0d arrays is deprecated")
def test_where_condition_only_fwd(condition):
    """mygrad.where should merely mirror numpy.where when only `where(condition)`
    is specified."""
    tensor_condition = (
        mg.Tensor(condition) if isinstance(condition, np.ndarray) else condition
    )
    assert all(
        np.all(x == y) for x, y in zip(np.where(tensor_condition), np.where(condition))
    )


@given(
    condition=hnp.arrays(shape=hnp.array_shapes(min_dims=1), dtype=bool),
    x=st.none()
    | hnp.arrays(
        shape=hnp.array_shapes(min_dims=1),
        dtype=int,
    ),
    y=st.none()
    | hnp.arrays(
        shape=hnp.array_shapes(min_dims=1),
        dtype=int,
    ),
)
def test_where_input_validation(condition, x, y):
    args = [i for i in (x, y) if i is not None]

    try:
        np.where(condition, *args)
    except Exception as e:
        with pytest.raises(type(e)):
            where(condition, *args)
        return
