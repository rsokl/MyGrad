import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose
from pytest import raises

import mygrad as mg
from mygrad.nnet.losses import multiclass_hinge
from mygrad.tensor_base import Tensor


@pytest.mark.parametrize(
    ("data", "labels"),
    [
        (np.ones((2,), dtype=float), np.zeros((2,), dtype=int)),  # 1D data
        (np.ones((2, 1), dtype=float), np.zeros((2,), dtype=float)),  # non-int labels
        (np.ones((2, 1), dtype=float), np.zeros((2, 1), dtype=int)),  # bad label-ndim
        (np.ones((2, 1), dtype=float), np.zeros((3,), dtype=int)),  # bad label-shape
    ],
)
def test_input_validation(data, labels):
    with raises((ValueError, TypeError)):
        multiclass_hinge(data, labels)


@given(st.data())
def test_multiclass_hinge(data):
    """Test the built-in implementation of multiclass hinge
    against the pure mygrad version"""
    s = data.draw(
        hnp.arrays(
            shape=hnp.array_shapes(max_side=10, min_dims=2, max_dims=2),
            dtype=float,
            elements=st.floats(-100, 100),
        )
    )
    loss = data.draw(
        hnp.arrays(
            shape=(s.shape[0],),
            dtype=hnp.integer_dtypes(),
            elements=st.integers(min_value=0, max_value=s.shape[1] - 1),
        )
    )
    hinge_scores = Tensor(s)
    hinge_loss = multiclass_hinge(hinge_scores, loss, constant=False).mean()
    hinge_loss.backward()

    mygrad_scores = Tensor(s)
    correct_labels = (range(len(loss)), loss)
    correct_class_scores = mygrad_scores[correct_labels]  # Nx1

    Lij = mygrad_scores - correct_class_scores[:, np.newaxis] + 1.0  # NxC margins
    Lij[Lij <= 0] = 0
    Lij[correct_labels] = 0

    mygrad_loss = Lij.sum() / mygrad_scores.shape[0]
    mygrad_loss.backward()
    assert_allclose(hinge_loss.data, mygrad_loss.data)
    assert_allclose(mygrad_scores.grad, hinge_scores.grad)


@given(shape=st.sampled_from([(3, 1), (3, 4), tuple()]))
def test_bad_label_shape(shape):
    """
    Ensures that `multiclass_hinge` checks for shape-(N,) `y_true`
    """
    scores = mg.arange(12).reshape(3, 4)
    labels = mg.zeros(shape, dtype=int)
    with raises(ValueError):
        multiclass_hinge(scores, labels)


@given(type=st.sampled_from([bool, float, np.float32]))
def test_bad_label_type(type):
    """
    Ensures that `multiclass_hinge` checks integer-type `y_true`
    """
    scores = mg.arange(12).reshape(3, 4)
    labels = np.zeros((3,), dtype=type)
    with raises(TypeError):
        multiclass_hinge(scores, labels)
