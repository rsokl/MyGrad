import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.tensor_base import Tensor


@given(
    x=st.floats(min_value=-1e-6, max_value=1e6),
    y=st.floats(min_value=-1e-6, max_value=1e6),
    z=st.floats(min_value=-1e-6, max_value=1e6),
    side_effects=st.booleans(),
)
def test_chainrule_scalar(x, y, z, side_effects):
    x = Tensor(x)
    y = Tensor(y)
    z = Tensor(z)

    f = x * y + z
    g = x + z * f * f

    if side_effects:
        # check side effects
        unused = 2 * g - f
        w = 1 * f
    else:
        unused = Tensor(0)
        w = Tensor(0)
    assert unused is not None

    g.backward()
    assert_allclose(f.grad, 2 * z.data * f.data)
    assert_allclose(x.grad, 1 + 2 * z.data * f.data * y.data)
    assert_allclose(y.grad, 2 * z.data * f.data * x.data)
    assert_allclose(z.grad, f.data ** 2 + z.data * 2 * f.data)

    assert w.grad is None


def test_identical_inputs():
    v1 = Tensor(2.0, constant=False)
    v2 = v1 + v1
    v3 = v2 + v2
    v3.backward(1.0)  # v3 = 4 * v1
    assert v3.data.item() == 8.0
    assert v1.grad.item() == 4.0


@given(data=st.floats(-10, 10), grad=(st.none() | st.floats(-10, 10)))
def test_non_broadcastable(data, grad):
    v1 = Tensor(data, constant=False)
    v2 = mg.exp(v1)
    v3 = mg.cos(v2)
    v3.backward(grad)

    if grad is None:
        grad = 1.0

    assert_allclose(actual=v2.data, desired=np.exp(v1.data))
    assert_allclose(actual=v3.data, desired=np.cos(np.exp(v1.data)))

    assert_allclose(actual=v3.grad, desired=grad)
    assert_allclose(actual=v2.grad, desired=-np.sin(v2.data) * grad)
    assert_allclose(actual=v1.grad, desired=np.exp(v1.data) * -np.sin(v2.data) * grad)
