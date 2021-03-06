from functools import partial
from typing import Optional, Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad.typing import ArrayLike
from tests.custom_strategies import array_likes, tensors


@st.composite
def concatable_tensors(
    draw, array_like_strat=array_likes
) -> st.SearchStrategy[Tuple[Tuple[ArrayLike, ...], Optional[int]]]:
    """draws valid inputs for numpy.concatenate"""
    axis_is_none = draw(st.booleans())
    if axis_is_none:
        inputs = draw(
            st.lists(
                array_like_strat(
                    shape=hnp.array_shapes(
                        min_dims=0, min_side=0, max_dims=3, max_side=3
                    ),
                    dtype=float,
                    elements=st.floats(-100, 100),
                ),
                min_size=1,
                max_size=3,
            ).map(tuple)
        )
        return inputs, None

    master_shape: list = draw(hnp.array_shapes(min_dims=1, min_side=0).map(list))
    N = len(master_shape) + 1
    axis = draw(st.integers(-N, N - 1))
    pos_axis = axis % N
    shapes = []
    for size in draw(st.lists(st.integers(0, 3), min_size=1, max_size=4)):
        shape = master_shape.copy()
        shape.insert(pos_axis, size)
        shapes.append(shape)

    concatable_arrs = st.tuples(
        *(
            array_like_strat(shape=shape, dtype=float, elements=st.floats(-100, 100))
            for shape in shapes
        )
    )
    return draw(concatable_arrs), axis


@given(concatable_tensors(hnp.arrays))
def test_concatable_tensors_strat(x: Tuple[Tuple[ArrayLike, ...], Optional[int]]):
    """Ensures strat does not produce bad results"""
    arrs, axis = x
    np.concatenate(arrs, axis=axis)


def not_tensor(x):
    return x if not isinstance(x, mg.Tensor) else x.data


@given(concatable_tensors(array_likes))
def test_concatenate_fwd(x: Tuple[Tuple[ArrayLike, ...], Optional[int]]):
    arrs, axis = x
    mygrad_out = mg.concatenate(arrs, axis=axis)

    # `not_tensor` ensures that mygrad override doesn't take place here
    numpy_out = np.concatenate(tuple(not_tensor(x) for x in arrs), axis=axis)
    assert isinstance(mygrad_out, mg.Tensor)
    assert (
        mygrad_out.base is None and numpy_out.base is None
    ), "concatenate returns a view"
    assert_array_equal(mygrad_out, numpy_out)


@given(concatable_tensors(partial(tensors, constant=False)))
def test_concatenate_bkwd(x: Tuple[Tuple[ArrayLike, ...], Optional[int]]):
    arrs, axis = x
    mygrad_out = np.concatenate(
        arrs, axis=axis
    )  # exercises __array_function__ override
    mygrad_out.backward(mygrad_out.data)
    for n, t in enumerate(arrs):
        assert_array_equal(t.grad, t.data), f"tensor {n}"


def test_concatenate_with_dtype():
    if np.__version__ < "1.20":
        pytest.skip("concatenate does not support dtype until numpy 1.20")
    x = mg.tensor([1.0, 2.0])
    assert mg.concatenate([x, x], dtype="float32").dtype == np.float32


def test_concatenate_with_inplace_target():
    if np.__version__ < "1.20":
        pytest.skip("concatenate does not support dtype until numpy 1.20")
    x = mg.tensor([1.0, 2.0])
    y = mg.empty((4,))
    np.concatenate([x, x], out=y)
    y.backward()

    assert_array_equal(x.grad, [2.0, 2.0])
    assert_array_equal(y, mg.concatenate([x, x]))
