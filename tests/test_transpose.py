from mygrad.tensor_base import Tensor
import numpy as np

def test_transpose_property():
    dat = np.arange(6).reshape(2, 3)
    x = Tensor(dat)
    f = x.T()
    f.backward(dat.T)

    assert np.allclose(f.data, dat.T)
    assert np.allclose(x.grad, dat)


def test_transpose():
    dat = np.arange(24).reshape(2, 3, 4)
    x = Tensor(dat)
    f = x.transpose(axes=(2, 1, 0))
    f.backward(dat.transpose((2, 1, 0)))

    assert np.allclose(f.data, dat.transpose((2, 1, 0)))
    assert np.allclose(x.grad, dat)
