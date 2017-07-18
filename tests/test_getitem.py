from mygrad.tensor_base import Tensor
import numpy as np


def test_getitem():
    x = Tensor([1, 2, 3])
    a, b, c = x
    f = 2*a + 3*b + 4*c
    f.backward()

    assert a.data == 1
    assert b.data == 2
    assert c.data == 3
    assert f.data == 20

    assert np.allclose(a.grad, np.array(2))
    assert np.allclose(b.grad, np.array(3))
    assert np.allclose(c.grad, np.array(4))
    assert np.allclose(x.grad, np.array([2, 3, 4]))

