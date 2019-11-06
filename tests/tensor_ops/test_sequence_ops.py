from functools import reduce
from typing import Callable, Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

from mygrad import add_sequence, multiply_sequence
from mygrad.tensor_base import Tensor


@given(st.lists(st.just(1.0), max_size=1))
def test_input_validation(arrays):
    with pytest.raises(ValueError):
        add_sequence(*arrays)

    with pytest.raises(ValueError):
        multiply_sequence(*arrays)


def prod(seq):
    return reduce(lambda x, y: x * y, seq)


@pytest.mark.parametrize(
    "sequential_function", ((add_sequence, sum), (multiply_sequence, prod))
)
@settings(deadline=None)
@given(
    shapes=st.integers(2, 4).flatmap(
        lambda n: hnp.mutually_broadcastable_shapes(num_shapes=n, min_dims=0)
    ),
    data=st.data(),
)
def test_sequential_arithmetic(
    sequential_function: Tuple[Callable, Callable],
    shapes: hnp.BroadcastableShapes,
    data: st.DataObject,
):
    mygrad_func, true_func = sequential_function
    tensors = data.draw(
        st.tuples(
            *(
                hnp.arrays(
                    shape=shape, dtype=np.float32, elements=st.floats(-10, 10, width=32)
                ).map(Tensor)
                for shape in shapes.input_shapes
            )
        ),
        label="tensors",
    )

    tensors_copy = [x.copy() for x in tensors]

    f = mygrad_func(*tensors)
    f1 = true_func(tensors_copy)

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
