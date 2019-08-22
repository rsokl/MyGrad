import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_array_equal
from scipy import special

from mygrad.math._special import logsumexp
from tests.custom_strategies import valid_axes


@settings(deadline=None)
@given(
    data=st.data(),
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0),
        dtype=np.float,
        elements=st.floats(),
    ),
    keepdims=st.booleans(),
)
def test_logsumexp(data: st.SearchStrategy, x: np.ndarray, keepdims: bool):
    axes = data.draw(valid_axes(ndim=x.ndim), label="axes")
    mygrad_result = logsumexp(x, axis=axes, keepdims=keepdims)
    scipy_result = special.logsumexp(x, axis=axes, keepdims=keepdims)
    assert_array_equal(
        mygrad_result,
        scipy_result,
        err_msg="mygrad's implementation of logsumexp does "
        "not match that of scipy's",
    )
