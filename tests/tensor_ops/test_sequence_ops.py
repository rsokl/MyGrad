from functools import reduce
from typing import Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, note, settings
from numpy.testing import assert_allclose

from mygrad import add_sequence, multiply_sequence
from mygrad.tensor_base import Tensor

# TODO: migrate to factory wrappers once mutually-broadcastable strat is available


@given(st.lists(st.just(1.0), max_size=1))
def test_input_validation(arrays):
    with pytest.raises(ValueError):
        add_sequence(*arrays)

    with pytest.raises(ValueError):
        multiply_sequence(*arrays)


def _broadcast_shapes(shape_a, shape_b):
    longest, shortest = (
        (shape_a, shape_b) if len(shape_a) >= len(shape_b) else (shape_b, shape_a)
    )
    shortest = (len(longest) - len(shortest)) * (1,) + shortest
    return tuple(a if a > b else b for a, b in zip(longest, shortest))


@settings(deadline=None)
@given(
    shape_1=hnp.array_shapes(min_dims=0), num_arrays=st.integers(1, 4), data=st.data()
)
def test_seq_add(shape_1: Tuple[int, ...], num_arrays: int, data: st.DataObject):
    shape_2 = data.draw(hnp.broadcastable_shapes(shape_1), label="shape_2")
    shapes = [shape_1, shape_2]

    pair = shapes
    # ensure sequence of shapes is mutually-broadcastable
    for i in range(num_arrays):
        broadcasted = _broadcast_shapes(*pair)
        shapes.append(
            data.draw(
                hnp.broadcastable_shapes(broadcasted), label="shape_{}".format(i + 3)
            )
        )
        pair = [broadcasted, shapes[-1]]

    tensors = [
        Tensor(
            data.draw(
                hnp.arrays(
                    shape=shape, dtype=np.float32, elements=st.floats(-10, 10, width=32)
                )
            )
        )
        for shape in shapes
    ]
    note("tensors: {}".format(tensors))
    tensors_copy = [x.copy() for x in tensors]

    f = add_sequence(*tensors)
    f1 = sum(tensors_copy)

    assert_allclose(f.data, f1.data)

    f.sum().backward()
    f1.sum().backward()

    assert_allclose(f.data, f1.data, rtol=1e-4, atol=1e-4)

    for n, (expected, actual) in enumerate(zip(tensors_copy, tensors)):
        assert_allclose(
            expected.grad,
            actual.grad,
            rtol=1e-4,
            atol=1e-4,
            err_msg="tensor-{}".format(n),
        )

    f.null_gradients()
    assert all(x.grad is None for x in tensors)
    assert all(not x._ops for x in tensors)


@settings(deadline=None)
@given(
    shape_1=hnp.array_shapes(min_dims=0), num_arrays=st.integers(1, 4), data=st.data()
)
def test_seq_mult(shape_1: Tuple[int, ...], num_arrays: int, data: st.DataObject):
    shape_2 = data.draw(hnp.broadcastable_shapes(shape_1), label="shape_2")
    shapes = [shape_1, shape_2]

    pair = shapes

    for i in range(num_arrays):

        # ensure sequence of shapes is mutually-broadcastable
        broadcasted = _broadcast_shapes(*pair)
        shapes.append(
            data.draw(
                hnp.broadcastable_shapes(broadcasted), label="shape_{}".format(i + 3)
            )
        )
        pair = [broadcasted, shapes[-1]]

    tensors = [
        Tensor(
            data.draw(
                hnp.arrays(
                    shape=shape, dtype=np.float32, elements=st.floats(-10, 10, width=32)
                )
            )
        )
        for shape in shapes
    ]
    note("tensors: {}".format(tensors))
    tensors_copy = [x.copy() for x in tensors]

    f = multiply_sequence(*tensors)
    f1 = reduce(lambda x, y: x * y, (var for n, var in enumerate(tensors_copy)))

    assert_allclose(f.data, f1.data)

    f.sum().backward()
    f1.sum().backward()

    assert_allclose(f.data, f1.data, rtol=1e-4, atol=1e-4)

    for n, (expected, actual) in enumerate(zip(tensors_copy, tensors)):
        assert_allclose(
            expected.grad,
            actual.grad,
            rtol=1e-3,
            atol=1e-3,
            err_msg="tensor-{}".format(n),
        )

    f.null_gradients()
    assert all(x.grad is None for x in tensors)
    assert all(not x._ops for x in tensors)
