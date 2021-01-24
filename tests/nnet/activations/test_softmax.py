import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

from mygrad import Tensor
from mygrad.nnet.activations import logsoftmax, softmax
from tests import is_float_arr
from tests.custom_strategies import valid_axes
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory

log_largest = np.log(np.finfo(np.float64).max)


@given(
    arr=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0, min_side=0, max_side=0),
        dtype=hnp.floating_dtypes() | hnp.integer_dtypes(),
        elements=dict(min_value=-10, max_value=10),
    ),
    data=st.data(),
)
def test_softmax_on_empty_arrays(arr: np.ndarray, data: st.DataObject):
    axes = data.draw(valid_axes(arr.ndim))
    out = softmax(arr, axis=axes)
    expected_dtype = arr.dtype if is_float_arr(arr) else np.dtype(np.float64)
    assert out.shape == arr.shape
    assert out.dtype == expected_dtype


@given(
    hnp.arrays(
        shape=hnp.array_shapes(min_dims=0, min_side=0),
        dtype=hnp.integer_dtypes(),
        elements=dict(min_value=-10, max_value=10),
    )
)
def test_softmax_on_ints(arr: np.ndarray):
    actual = softmax(arr)
    desired = softmax(arr.astype(np.float))
    assert desired.dtype == actual.dtype
    assert_allclose(desired, actual, atol=1e-3, rtol=1e-3)


@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0),
        dtype=np.float64,
        elements=st.floats(-log_largest, log_largest),
    ),
    data=st.data(),
)
def test_softmax_numerical_stability(x: np.ndarray, data: st.DataObject):
    axis = data.draw(valid_axes(x.ndim), label="axis")
    out = softmax(x, axis=axis).data
    assert np.all(np.logical_and(0 <= out, out <= 1))
    assert_allclose(out.sum(axis=axis), 1.0)


@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0),
        dtype=np.float64,
        elements=st.floats(-log_largest, log_largest),
    ),
    data=st.data(),
)
def test_log_softmax_numerical_stability(x: np.ndarray, data: st.DataObject):
    axis = data.draw(valid_axes(x.ndim), label="axis")
    out = np.exp(logsoftmax(x, axis=axis).data)
    assert np.all(np.logical_and(0 <= out, out <= 1)), out
    assert_allclose(out.sum(axis=axis), 1.0)


def numpy_softmax(x, axis):
    x = np.asarray(x)
    x = np.exp(x - x.max(axis, keepdims=True))
    return x / x.sum(axis, keepdims=True)


def numpy_logsoftmax(x, axis):
    return np.log(numpy_softmax(x, axis))


@fwdprop_test_factory(
    mygrad_func=softmax,
    true_func=numpy_softmax,
    num_arrays=1,
    kwargs=dict(axis=lambda arrs: valid_axes(arrs.ndim)),
)
def test_softmax_fwd():
    pass


@backprop_test_factory(
    mygrad_func=softmax,
    true_func=numpy_softmax,
    num_arrays=1,
    kwargs=dict(axis=lambda arrs: valid_axes(arrs.ndim)),
    vary_each_element=True,
)
def test_softmax_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=logsoftmax,
    true_func=numpy_logsoftmax,
    num_arrays=1,
    kwargs=dict(axis=lambda arrs: valid_axes(arrs.ndim)),
    index_to_bnds={0: (-10, 10)},
)
def test_logsoftmax_fwd():
    pass


@backprop_test_factory(
    mygrad_func=logsoftmax,
    true_func=numpy_logsoftmax,
    num_arrays=1,
    kwargs=dict(axis=lambda arrs: valid_axes(arrs.ndim)),
    vary_each_element=True,
    index_to_bnds={0: (-10, 10)},
)
def test_logsoftmax_bkwd():
    pass


def test_static_softmax1d():
    # Verified against theano.tensor.softmax

    skew = np.array([0.87566484, 0.53596079, 0.85693981, 0.09526036])
    x = np.array([0.0, 1.0, 2.0, 3.0])

    x = Tensor(x)
    f = (softmax(x, constant=False) * skew).sum()

    out = np.array(0.33911235096116465)
    assert_allclose(actual=f.data, desired=out)

    f.backward()
    dx = np.array([0.01720112, 0.01715422, 0.12266443, -0.15701977])

    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)


def test_static_softmax2d():
    # Verified against theano.tensor.softmax

    skew = np.array(
        [
            [0.87566484, 0.53596079, 0.85693981, 0.09526036],
            [0.32024455, 0.81532148, 0.2480434, 0.85119342],
            [0.57943085, 0.33958252, 0.95864464, 0.22881712],
        ]
    )

    x = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]])

    x = Tensor(x)
    f = (softmax(x, constant=False) * skew).sum()

    out = np.array(1.449875865467131)
    assert_allclose(actual=f.data, desired=out)

    f.backward()
    dx = np.array(
        [
            [0.01720112, 0.01715422, 0.12266443, -0.15701977],
            [-0.01179518, 0.01108053, -0.10425844, 0.10497309],
            [0.00502799, -0.00723393, 0.12698131, -0.12477536],
        ]
    )

    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)


def test_static_logsoftmax1d():
    # Verified against theano.tensor.softmax

    skew = np.array([0.87566484, 0.53596079, 0.85693981, 0.09526036])
    x = np.array([0.0, 1.0, 2.0, 3.0])

    x = Tensor(x)
    f = (logsoftmax(x, constant=False) * skew).sum()

    out = np.array(-5.596387676353177)
    assert_allclose(actual=f.data, desired=out)

    f.backward()
    dx = np.array([0.79988389, 0.3299668, 0.29699009, -1.42684078])

    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)


def test_static_logsoftmax2d():
    # Verified against theano.tensor.softmax
    skew = np.array(
        [
            [0.87566484, 0.53596079, 0.85693981, 0.09526036],
            [0.32024455, 0.81532148, 0.2480434, 0.85119342],
            [0.57943085, 0.33958252, 0.95864464, 0.22881712],
        ]
    )

    x = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]])

    x = Tensor(x)
    f = (logsoftmax(x, constant=False) * skew).sum()

    out = np.array(-13.722895761739732)
    assert_allclose(actual=f.data, desired=out)

    f.backward()
    dx = np.array(
        [
            [0.79988389, 0.3299668, 0.29699009, -1.42684078],
            [0.24859989, 0.62057111, -0.281343, -0.587828],
            [0.5119002, 0.15601518, 0.45965687, -1.12757225],
        ]
    )

    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)
