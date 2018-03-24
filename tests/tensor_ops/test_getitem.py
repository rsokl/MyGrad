from mygrad.tensor_base import Tensor
import numpy as np
from numpy.testing import assert_allclose

def test_getitem():
    x = Tensor([1, 2, 3])
    a, b, c = x
    f = 2*a + 3*b + 4*c
    f.backward()

    assert a.data == 1
    assert b.data == 2
    assert c.data == 3
    assert f.data == 20

    assert_allclose(a.grad, np.array(2))
    assert_allclose(b.grad, np.array(3))
    assert_allclose(c.grad, np.array(4))
    assert_allclose(x.grad, np.array([2, 3, 4]))

