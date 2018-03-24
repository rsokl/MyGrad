import numpy as np
from numpy.testing import assert_allclose
from mygrad import Tensor
from mygrad.nnet.activations import softmax, logsoftmax


def test_static_softmax():
    # Verified against theano.tensor.softmax

    skew = np.array([[ 0.87566484,  0.53596079,  0.85693981,  0.09526036],
                     [ 0.32024455,  0.81532148,  0.2480434 ,  0.85119342],
                     [ 0.57943085,  0.33958252,  0.95864464,  0.22881712]])

    x = np.array([[  0.,   1.,   2.,   3.],
                  [  4.,   5.,   6.,   7.],
                  [  8.,   9.,  10.,  11.]])

    x = Tensor(x)
    f = (softmax(x) * skew).sum()

    out = np.array(1.449875865467131)
    assert_allclose(f.data, out)

    f.backward()
    dx = np.array([[ 0.01720112,  0.01715422,  0.12266443, -0.15701977],
                   [-0.01179518,  0.01108053, -0.10425844,  0.10497309],
                   [ 0.00502799, -0.00723393,  0.12698131, -0.12477536]])

    assert_allclose(x.grad, dx)


def test_static_logsoftmax():
    # Verified against theano.tensor.softmax
    skew = np.array([[ 0.87566484,  0.53596079,  0.85693981,  0.09526036],
                     [ 0.32024455,  0.81532148,  0.2480434 ,  0.85119342],
                     [ 0.57943085,  0.33958252,  0.95864464,  0.22881712]])

    x = np.array([[  0.,   1.,   2.,   3.],
                  [  4.,   5.,   6.,   7.],
                  [  8.,   9.,  10.,  11.]])

    x = Tensor(x)
    f = (logsoftmax(x) * skew).sum()

    out = np.array(-13.722895761739732)
    assert_allclose(f.data, out)

    f.backward()
    dx = np.array([[ 0.79988389,  0.3299668 ,  0.29699009, -1.42684078],
                   [ 0.24859989,  0.62057111, -0.281343  , -0.587828  ],
                   [ 0.5119002 ,  0.15601518,  0.45965687, -1.12757225]])

    assert_allclose(x.grad, dx)

