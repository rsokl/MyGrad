from mygrad.tensor_base import Tensor

from ..custom_strategies import valid_axes
import numpy as np
import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp


@given(keepdims=st.booleans(),
       d=st.data()
       )
def test_sum_fwd(keepdims, d):
    a = d.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                          dtype=float,
                          elements=st.floats(-100, 100)))
    axes = d.draw(valid_axes(a.ndim))
    a = Tensor(a)

    np_out = a.data.sum(axis=axes, keepdims=keepdims)
    mygrad_out = a.sum(axis=axes, keepdims=keepdims).data
    if mygrad_out.ndim == 0:
        mygrad_out = np.asscalar(mygrad_out)

    assert np.allclose(np_out, mygrad_out)


@given(keepdims=st.booleans(),
       d=st.data()
       )
def test_sum_bkwd(keepdims, d):
    a = d.draw(hnp.arrays(shape=hnp.array_shapes(max_side=1, max_dims=5),
                          dtype=float,
                          elements=st.floats(-100, 100)))
    axes = d.draw(valid_axes(a.ndim))
    a = Tensor(a)

    mygrad_out = a.sum(axis=axes, keepdims=keepdims)

    grad = d.draw(hnp.arrays(shape=mygrad_out.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    mygrad_out.backward(grad)

    if not keepdims and mygrad_out.ndim:
        index = [slice(None) for i in range(a.ndim)]
        for i in axes:
            index[i] = np.newaxis
        grad = grad[index]

    grad = np.broadcast_to(grad, a.shape)

    assert np.allclose(grad, mygrad_out.grad)


@given(keepdims=st.booleans(),
       d=st.data()
       )
def test_mean_fwd(keepdims, d):
    a = d.draw(hnp.arrays(shape=hnp.array_shapes(max_side=1, max_dims=5),
                          dtype=float,
                          elements=st.floats(-100, 100)))
    axes = d.draw(valid_axes(a.ndim))
    a = Tensor(a)

    np_out = a.data.mean(axis=axes, keepdims=keepdims)
    mygrad_out = a.mean(axis=axes, keepdims=keepdims).data
    if mygrad_out.ndim == 0:
        mygrad_out = np.asscalar(mygrad_out)

    assert np.allclose(np_out, mygrad_out)


@given(keepdims=st.booleans(),
       d=st.data()
       )
def test_mean_bkwd(keepdims, d):
    a = d.draw(hnp.arrays(shape=hnp.array_shapes(max_side=1, max_dims=5),
                          dtype=float,
                          elements=st.floats(-100, 100)))
    axes = d.draw(valid_axes(a.ndim))
    a = Tensor(a)
    b = Tensor(a)

    mygrad_out = a.mean(axis=axes, keepdims=keepdims)

    n = a.data.size if not axes else np.prod([a.shape[i] for i in axes])
    sum_out = b.sum(axis=axes, keepdims=keepdims) / n

    grad = d.draw(hnp.arrays(shape=mygrad_out.shape,
                             dtype=float,
                             elements=st.floats(-100, 100)))

    mygrad_out.backward(grad)
    sum_out.backward(grad)

    assert np.allclose(sum_out.grad, mygrad_out.grad)