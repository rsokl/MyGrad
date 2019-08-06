from numbers import Real

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings

from mygrad._utils import is_invalid_gradient
from tests.custom_strategies import everything_except


@pytest.mark.parametrize(
    ("grad", "is_invalid"),
    [
        (everything_except((np.ndarray, Real)), True),
        (None, True),
        (np.ndarray([1], dtype="O"), True),
        (
            hnp.arrays(
                shape=hnp.array_shapes(),
                dtype=hnp.floating_dtypes(),
                elements=st.floats(width=16),
            ),
            False,
        ),
        ((st.integers(min_value=-1e6, max_value=1e6) | st.floats()), False),
    ],
)
@settings(deadline=None, suppress_health_check=(HealthCheck.filter_too_much,))
@given(data=st.data())
def test_is_invalid_gradient(grad, is_invalid, data: st.DataObject):
    if isinstance(grad, st.SearchStrategy):
        grad = data.draw(grad, label="grad")

    assert is_invalid_gradient(grad) is is_invalid, grad
