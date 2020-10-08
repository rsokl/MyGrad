import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import multiply_sequence
from mygrad._utils.duplicating_graph import make_placeholder_tensor
from tests.custom_strategies import choices, tensors


@given(
    x=tensors(elements=st.floats(-100, 100), read_only=st.booleans()), data=st.data(),
)
def test_placeholder_tensor(x: mg.Tensor, data: st.DataObject):
    """
    Ensure the backpropagation is properly rerouted through placeholder tensor
    """
    was_writeable = x.data.flags.writeable

    y = mg.ones_like(x)
    # shuffle order
    # note that one of the entries is 2*x, which exercises the
    # process of rerouting the tensor for multiple ops
    seq = data.draw(
        choices([x, x, 2 * x, y, y], size=5, replace=False), label="sequence"
    )
    out = multiply_sequence(*seq).sum()
    placeholder = make_placeholder_tensor(x)

    assert placeholder is not x
    assert_array_equal(x, placeholder)
    assert placeholder.constant is x.constant
    assert placeholder.data.flags.writeable is x.data.flags.writeable

    if x.size:
        assert np.shares_memory(x, placeholder)

    out.backward()

    # note that writeability of `x` is not restored
    assert placeholder.data.flags.writeable is was_writeable

    assert x.grad is None

    if not y.constant:
        assert_allclose(y.grad, 4 * x.data ** 3, atol=1e-7)

    if not placeholder.constant:
        assert_allclose(placeholder.grad, 6 * x.data ** 2)
