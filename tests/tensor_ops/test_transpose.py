from mygrad.tensor_base import Tensor
from mygrad import transpose
from numpy.testing import assert_allclose
import numpy as np

def test_transpose_property():
    dat = np.arange(6).reshape(2, 3)
    x = Tensor(dat)
    f = x.T
    f.backward(dat.T)

    assert_allclose(f.data, dat.T)
    assert_allclose(x.grad, dat)


def test_transpose_method():
    dat = np.arange(24).reshape(2, 3, 4)
    x = Tensor(dat)
    f = x.transpose(axes=(2, 1, 0))
    f.backward(dat.transpose((2, 1, 0)))

    assert_allclose(f.data, dat.transpose((2, 1, 0)))
    assert_allclose(x.grad, dat)


def test_transpose():
    dat = np.arange(24).reshape(2, 3, 4)
    x = Tensor(dat)
    f = transpose(x, axes=(2, 1, 0))
    f.backward(dat.transpose((2, 1, 0)))

    assert_allclose(f.data, np.transpose(dat, axes=(2, 1, 0)))
    assert_allclose(x.grad, dat)
