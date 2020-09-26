import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import multiply_sequence
from tests.custom_strategies import choices, tensors


@given(
    x=tensors(elements=st.floats(-100, 100)), data=st.data(), make_copy=st.booleans()
)
def test_placeholder_tenosre(x: mg.Tensor, data: st.DataObject, make_copy: bool):
    """
    Ensure the backpropagation is properly rerouted through placeholder tensor
    """
    y = mg.ones_like(x)
    # shuffle order
    # note that one of the entries is 2*x, which exercises the
    # process of rerouting the tensor for multiple ops
    seq = data.draw(
        choices([x, x, 2 * x, y, y], size=5, replace=False), label="sequence"
    )
    out = multiply_sequence(*seq).sum()
    placeholder = x._make_placeholder_tensor(copy_data=make_copy)

    assert_array_equal(x, placeholder)
    assert placeholder.constant is x.constant

    if x.size:
        assert np.shares_memory(x, placeholder) is not make_copy

    out.backward()

    assert x.grad is None

    if not y.constant:
        assert_allclose(y.grad, 4 * x.data ** 3, atol=1e-7)

    if not placeholder.constant:
        assert_allclose(placeholder.grad, 6 * x.data ** 2)
