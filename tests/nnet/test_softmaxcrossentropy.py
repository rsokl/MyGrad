import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given
from numpy.testing import assert_allclose
from pytest import raises

import mygrad as mg
from mygrad import log
from mygrad.nnet.activations import softmax
from mygrad.nnet.losses import softmax_crossentropy
from mygrad.tensor_base import Tensor


@given(st.data())
def test_softmax_crossentropy(data):
    """ Test the built-in implementation of multiclass hinge against the pure pygrad version"""
    s = data.draw(
        hnp.arrays(
            shape=hnp.array_shapes(max_side=10, min_dims=2, max_dims=2),
            dtype=float,
            elements=st.floats(-100, 100),
        )
    )
    l = data.draw(
        hnp.arrays(
            shape=(s.shape[0],),
            dtype=hnp.integer_dtypes(),
            elements=st.integers(min_value=0, max_value=s.shape[1] - 1),
        )
    )
    scores = Tensor(s)
    softmax_cross = softmax_crossentropy(scores, l, constant=False)
    softmax_cross.backward()

    pygrad_scores = Tensor(s)
    probs = softmax(pygrad_scores)

    correct_labels = (range(len(l)), l)
    truth = np.zeros(pygrad_scores.shape)
    truth[correct_labels] = 1

    pygrad_cross = (-1 / s.shape[0]) * (log(probs) * truth).sum()
    pygrad_cross.backward()
    assert_allclose(softmax_cross.data, pygrad_cross.data, atol=1e-5, rtol=1e-5)
    assert_allclose(scores.grad, pygrad_scores.grad, atol=1e-5, rtol=1e-5)


@given(shape=st.sampled_from([(3, 1), (3, 4), tuple()]))
def test_bad_label_shape(shape):
    """
    Ensures that softmax_crossentropy checks for shape-(N,) `y_true`
    """
    scores = mg.arange(12).reshape(3, 4)
    labels = mg.zeros(shape, dtype=int)
    with raises(ValueError):
        softmax_crossentropy(scores, labels)


@given(type=st.sampled_from([bool, float, np.float32]))
def test_bad_label_type(type):
    """
    Ensures that softmax_crossentropy checks integer-type `y_true`
    """
    scores = mg.arange(12).reshape(3, 4)
    labels = np.zeros((3,), dtype=type)
    with raises(TypeError):
        softmax_crossentropy(scores, labels)
