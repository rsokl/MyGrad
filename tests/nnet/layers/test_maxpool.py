import numpy as np
from numpy.testing import assert_allclose
from pytest import raises

from mygrad.nnet.layers import max_pool
from mygrad.tensor_base import Tensor


def text_constant():
    x = np.array(
        [
            [
                [17, 10, 15, 28, 25, 23],
                [44, 26, 18, 16, 39, 34],
                [5, 42, 36, 0, 2, 46],
                [30, 20, 1, 31, 35, 43],
            ],
            [
                [6, 7, 45, 27, 11, 8],
                [37, 4, 41, 22, 9, 33],
                [47, 3, 13, 32, 21, 38],
                [19, 12, 40, 24, 14, 29],
            ],
        ]
    )
    x = Tensor(x)
    pool = (3,)
    stride = (1,)
    assert max_pool(x, pool, stride, constant=True).constant is True
    assert max_pool(x, pool, stride, constant=False).constant is False


def test_1d_case():
    x = np.array(
        [
            [
                [17, 10, 15, 28, 25, 23],
                [44, 26, 18, 16, 39, 34],
                [5, 42, 36, 0, 2, 46],
                [30, 20, 1, 31, 35, 43],
            ],
            [
                [6, 7, 45, 27, 11, 8],
                [37, 4, 41, 22, 9, 33],
                [47, 3, 13, 32, 21, 38],
                [19, 12, 40, 24, 14, 29],
            ],
        ]
    )
    x = Tensor(x)
    pool = (3,)
    stride = (1,)
    out = max_pool(x, pool, stride)
    out.backward(np.arange(out.data.size).reshape(out.shape))

    fwd_ans = np.array(
        [
            [[17, 28, 28, 28], [44, 26, 39, 39], [42, 42, 36, 46], [30, 31, 35, 43]],
            [[45, 45, 45, 27], [41, 41, 41, 33], [47, 32, 32, 38], [40, 40, 40, 29]],
        ]
    )

    bkc_ans = np.array(
        [
            [
                [0.0, 0.0, 0.0, 6.0, 0.0, 0.0],
                [4.0, 5.0, 0.0, 0.0, 13.0, 0.0],
                [0.0, 17.0, 10.0, 0.0, 0.0, 11.0],
                [12.0, 0.0, 0.0, 13.0, 14.0, 15.0],
            ],
            [
                [0.0, 0.0, 51.0, 19.0, 0.0, 0.0],
                [0.0, 0.0, 63.0, 0.0, 0.0, 23.0],
                [24.0, 0.0, 0.0, 51.0, 0.0, 27.0],
                [0.0, 0.0, 87.0, 0.0, 0.0, 31.0],
            ],
        ]
    )
    assert isinstance(out, Tensor)
    assert_allclose(fwd_ans, out.data)
    assert_allclose(bkc_ans, x.grad)
    assert max_pool(x, pool, stride, constant=True).constant is True
    assert max_pool(x, pool, stride, constant=False).constant is False


def test_2d_case():
    x = np.array(
        [
            [
                [17, 10, 15, 28, 25, 23],
                [44, 26, 18, 16, 39, 34],
                [5, 42, 36, 0, 2, 46],
                [30, 20, 1, 31, 35, 43],
            ],
            [
                [6, 7, 45, 27, 11, 8],
                [37, 4, 41, 22, 9, 33],
                [47, 3, 13, 32, 21, 38],
                [19, 12, 40, 24, 14, 29],
            ],
        ]
    )
    x = Tensor(x)
    pool = (2, 3)
    stride = (2, 1)
    out = max_pool(x, pool, stride)
    out.sum().backward()

    fwd_ans = np.array(
        [[[44, 28, 39, 39], [42, 42, 36, 46]], [[45, 45, 45, 33], [47, 40, 40, 38]]]
    )

    bkc_ans = np.array(
        [
            [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 2.0, 0.0],
                [0.0, 2.0, 1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    assert isinstance(out, Tensor)
    assert_allclose(fwd_ans, out.data)
    assert_allclose(bkc_ans, x.grad)


def test_3d_case():
    x = np.array(
        [
            [
                [33, 27, 47, 11, 7, 36],
                [20, 18, 9, 2, 3, 17],
                [45, 31, 24, 12, 25, 19],
                [28, 1, 8, 16, 34, 14],
            ],
            [
                [37, 39, 40, 41, 21, 13],
                [35, 15, 6, 4, 23, 30],
                [43, 46, 32, 10, 26, 42],
                [38, 5, 44, 29, 0, 22],
            ],
        ]
    )
    x = Tensor(x)
    pool = (2, 2, 2)
    stride = (1, 1, 2)
    out = max_pool(x, pool, stride)
    g = np.arange(out.data.size)
    out.backward(g.reshape(out.shape))

    fwd_ans = np.array([[[39, 47, 36], [46, 32, 42], [46, 44, 42]]])

    bkc_ans = np.array(
        [
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 9.0, 4.0, 0.0, 0.0, 13.0],
                [0.0, 0.0, 7.0, 0.0, 0.0, 0.0],
            ],
        ]
    )
    assert isinstance(out, Tensor)
    assert_allclose(fwd_ans, out.data)
    assert_allclose(bkc_ans, x.grad)


def test_bad_max_shapes():
    x = Tensor(np.zeros((1, 2, 2, 2)))
    with raises(ValueError):
        max_pool(x, (3,) * 3, (1,) * 3)  # large filter

    with raises(AssertionError):
        max_pool(x, (2,) * 3, (0,) * 3)  # bad stride

    with raises(AssertionError):
        max_pool(x, (2,) * 2, [1, 2, 3])  # bad stride

    with raises(ValueError):
        max_pool(x, (1,) * 3, (3,) * 3)  # shape mismatch
