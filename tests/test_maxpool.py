import numpy as np

from mygrad.nnet.layers import max_pool
from mygrad.tensor_base import Tensor
from pytest import raises


def test_maxpool():

    # case 1
    x = np.zeros((1, 1, 2, 2))
    pool = 2
    stride = 1

    a = Tensor(x)
    f = max_pool(a, pool, stride)
    assert np.all(f.data == np.zeros((1, 1, 1, 1)))

    f.backward()
    assert np.all(a.grad == np.array([1, 0, 0, 0]).reshape(1, 1, 2, 2))

    # case 2
    x = np.arange(2*4*3).reshape(1, 2, 4, 3)
    x *= x[..., ::-1, ::-1]
    x[0, 0, 0, 1] = 400

    out = np.array([[[[400, 400],
                      [ 30,  28]],

                     [[304, 306],
                      [306, 304]]]])

    pool = 2
    stride = [2, 1]

    a = Tensor(x)
    f = max_pool(a, pool, stride)

    assert np.all(f.data == out)

    f.backward(np.arange(1, 9).reshape(1, 2, 2, 2))

    da = np.array([[[[0, 3, 0],
                     [0, 0, 0],
                     [3, 4, 0],
                     [0, 0, 0]],

                    [[0, 0, 0],
                     [0, 5, 6],
                     [7, 8, 0],
                     [0, 0, 0]]]])

    assert np.all(da == a.grad)


def test_bad_max_shapes():
    x = Tensor(np.zeros((1, 2, 2, 2)))
    with raises(ValueError):
        max_pool(x, 3, 1)  # large filter

    with raises(AssertionError):
        max_pool(x, 2, 0)  # bad stride

    with raises(AssertionError):
        max_pool(x, 2, [1, 2, 3])  # bad stride

    with raises(ValueError):
        max_pool(x, 1, 3)  # shape mismatch
