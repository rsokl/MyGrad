from mygrad.tensor_base import Tensor
from mygrad.math import multiply_sequence, add_sequence
import numpy as np


def test_seq_add():
    a = Tensor(3)
    b = Tensor([1, 2, 3])
    c = Tensor([[1, 2, 3], [2, 3, 4]])
    f = add_sequence(a, b, c)
    f.sum().backward()

    a1 = Tensor(3)
    b1 = Tensor([1, 2, 3])
    c1 = Tensor([[1, 2, 3], [2, 3, 4]])
    f1 = a1 + b1 + c1
    f1.sum().backward()

    assert np.allclose(f.data, f1.data)
    assert np.allclose(f.grad, f1.grad)
    assert np.allclose(a.grad, a1.grad)
    assert np.allclose(b.grad, b1.grad)
    assert np.allclose(c.grad, c1.grad)


def test_seq_mult():
    a = Tensor(3.)
    b = Tensor([1., 2., 3.])
    c = Tensor([[1., 2., 3.], [2., 3., 4.]])
    f = multiply_sequence(a, b, c)
    f.sum().backward()

    a1 = Tensor(3.)
    b1 = Tensor([1., 2., 3.])
    c1 = Tensor([[1., 2., 3.], [2., 3., 4.]])
    f1 = a1 * b1 * c1
    f1.sum().backward()

    assert np.allclose(f.data, f1.data)
    assert np.allclose(f.grad, f1.grad)
    assert np.allclose(a.grad, a1.grad)
    assert np.allclose(b.grad, b1.grad)
    assert np.allclose(c.grad, c1.grad)
