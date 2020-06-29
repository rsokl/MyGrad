from typing import Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad.nnet.losses import margin_ranking_loss


@pytest.mark.parametrize(
    "args",
    [
        dict(x1=st.sampled_from([np.ones((1,)), np.ones((1, 1, 1))])),
        dict(x1=np.ones((3, 4), dtype=int)),
        dict(x2=np.ones((3, 4), dtype=int)),
        dict(x2=np.ones((3, 5))),
        dict(margin=None),
        dict(margin=-1),
        dict(y=np.ones((3, 4))),
        dict(y=np.ones((5,))),
    ],
)
@given(data=st.data())
def test_input_validation(args, data: st.DataObject):
    kwargs = dict(x1=np.ones((3, 4)), y=mg.Tensor(1), margin=0.5)

    if "x2" not in args:
        args["x2"] = np.ones_like(kwargs["x1"])

    kwargs.update(
        (k, (data.draw(v, label=k)) if isinstance(v, st.SearchStrategy) else v)
        for k, v in args.items()
    )

    with pytest.raises((ValueError, TypeError)):
        margin_ranking_loss(**kwargs)


def simple_loss(x1, x2, y, margin):
    """
    x1 : mygrad.Tensor, shape=(N, D)
    x2 : mygrad.Tensor, shape=(N, D)
    y : Union[int, numpy.ndarray, mygrad.Tensor], scalar or shape=(N,)
    margin : float

    Returns
    -------
    mygrad.Tensor, shape=()
    """
    if isinstance(y, mg.Tensor):
        y = y.data
    y = np.asarray(y)
    if y.ndim:
        assert y.size == 1 or len(y) == len(x1)
        if x1.ndim == 2:
            y = y.reshape(-1, 1)

    return mg.mean(mg.maximum(0, margin - y * (x1 - x2)))


@given(
    shape=hnp.array_shapes(min_dims=1, max_dims=2),
    margin=st.floats(0, 1000),
    labels_as_tensor=st.booleans(),
    data=st.data(),
)
def test_ranked_margin(
    shape: Tuple[int, ...], margin: float, labels_as_tensor: bool, data: st.DataObject
):
    x1 = data.draw(
        hnp.arrays(shape=shape, dtype=float, elements=st.floats(-1000, 1000)),
        label="x1",
    )
    x2 = data.draw(
        hnp.arrays(shape=shape, dtype=float, elements=st.floats(-1000, 1000)),
        label="x2",
    )
    y = data.draw(
        st.sampled_from((-1, 1))
        | hnp.arrays(
            shape=shape[:1],
            dtype=hnp.integer_dtypes(),
            elements=st.sampled_from((-1, 1)),
        ).map(lambda x: mg.Tensor(x) if labels_as_tensor else x),
        label="y",
    )

    x1_copy = np.copy(x1)
    x2_copy = np.copy(x2)
    y_copy = y.copy() if isinstance(y, mg.Tensor) else np.copy(y)

    x1_dum = mg.Tensor(x1)
    x2_dum = mg.Tensor(x2)

    x1_real = mg.Tensor(x1)
    x2_real = mg.Tensor(x2)

    loss_dum = simple_loss(x1_dum, x2_dum, y, margin)

    loss_real = margin_ranking_loss(x1_real, x2_real, y, margin).mean()

    assert_allclose(
        actual=loss_real.data, desired=loss_dum.data, err_msg="losses don't match"
    )

    assert_array_equal(x1, x1_copy, err_msg="`x1` was mutated by forward")
    assert_array_equal(x2, x2_copy, err_msg="`x2` was mutated by forward")
    if isinstance(y, np.ndarray):
        assert_array_equal(y, y_copy, err_msg="`y` was mutated by forward")

    loss_dum.backward()
    loss_real.backward()

    assert_allclose(
        actual=x1_real.grad, desired=x1_dum.grad, err_msg="x1.grad doesn't match"
    )
    assert_allclose(
        actual=x2_real.grad, desired=x2_dum.grad, err_msg="x2.grad doesn't match"
    )

    assert_array_equal(x1, x1_copy, err_msg="`x1` was mutated by backward")
    assert_array_equal(x2, x2_copy, err_msg="`x2` was mutated by backward")

    if isinstance(y, mg.Tensor):
        y = y.data

    if isinstance(y, np.ndarray):
        assert_array_equal(y, y_copy, err_msg="`y` was mutated by backward")

    loss_real.null_gradients()
    assert x1_real.grad is None
    assert x2_real.grad is None
