import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose

import mygrad as mg

from ...custom_strategies import tensors, valid_axes

dtype_strat_numpy = st.sampled_from(
    (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
)


@given(
    tensor=tensors(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    ),
    data=st.data(),
)
def test_argmin(tensor: mg.Tensor, data: st.DataObject):
    axis = data.draw(valid_axes(ndim=tensor.ndim, single_axis_only=True), label="axis")

    a = tensor.data

    # tensor input
    assert_allclose(mg.argmin(tensor, axis=axis), np.argmin(a, axis=axis))

    # tensor method
    assert_allclose(tensor.argmin(axis=axis), a.argmin(axis=axis))

    # array input
    assert_allclose(mg.argmin(a, axis=axis), np.argmin(a, axis=axis))


@given(
    tensor=tensors(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    ),
    data=st.data(),
)
def test_argmax(tensor: mg.Tensor, data: st.DataObject):
    axis = data.draw(valid_axes(ndim=tensor.ndim, single_axis_only=True), label="axis")

    a = tensor.data

    # tensor input
    assert_allclose(mg.argmax(tensor, axis=axis), np.argmax(a, axis=axis))

    # tensor method
    assert_allclose(tensor.argmax(axis=axis), a.argmax(axis=axis))

    # array input
    assert_allclose(mg.argmax(a, axis=axis), np.argmax(a, axis=axis))


@given(
    tensor=tensors(
        shape=hnp.array_shapes(max_side=4, max_dims=5), dtype=dtype_strat_numpy
    ),
    data=st.data(),
    keepdims=st.booleans(),
)
def test_any(tensor: mg.Tensor, data: st.DataObject, keepdims):
    axis = data.draw(valid_axes(ndim=tensor.ndim), label="axis")
    a = tensor.data

    # tensor input
    assert_allclose(
        mg.any(tensor, axis=axis, keepdims=keepdims),
        np.any(a, axis=axis, keepdims=keepdims),
    )

    # tensor method
    assert_allclose(
        tensor.any(axis=axis, keepdims=keepdims), a.any(axis=axis, keepdims=keepdims)
    )

    # array input
    assert_allclose(
        mg.any(a, axis=axis, keepdims=keepdims), np.any(a, axis=axis, keepdims=keepdims)
    )

    # test when keepdims is not specified
    assert_allclose(mg.any(a, axis=axis), np.any(a, axis=axis))
