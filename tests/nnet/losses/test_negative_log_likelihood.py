import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose
from pytest import raises

import mygrad as mg
from mygrad.nnet.losses import negative_log_likelihood, softmax_crossentropy
from mygrad.tensor_base import Tensor


@pytest.mark.parametrize(
    ("data", "labels", "weights"),
    [
        (np.ones((2,), dtype=float), np.zeros((2,), dtype=int), None),  # 1D data
        (np.ones((2, 1), dtype=float), np.zeros((2,), dtype=float), None),  # non-int labels
        (np.ones((2, 1), dtype=float), np.zeros((2, 1), dtype=int), None),  # bad label-ndim
        (np.ones((2, 1), dtype=float), np.zeros((3,), dtype=int), None),  # bad label-shape
        (np.ones((2, 2), dtype=float), np.zeros((2,), dtype=int), np.ones((1,)))  # bad weight shape
    ],
)
def test_input_validation(data, labels, weights):
    with raises((ValueError, TypeError)):
        negative_log_likelihood(data, labels, weights=weights)


@given(data=st.data(), labels_as_tensor=st.booleans())
def test_negative_log_likelihood(data: st.DataObject, labels_as_tensor: bool):
    s = data.draw(
        hnp.arrays(
            shape=hnp.array_shapes(max_side=10, min_dims=2, max_dims=2),
            dtype=float,
            elements=st.floats(-100, 100),
        )
    )
    y_true = data.draw(
        hnp.arrays(
            shape=(s.shape[0],),
            dtype=hnp.integer_dtypes(),
            elements=st.integers(min_value=0, max_value=s.shape[1] - 1),
        ).map(Tensor if labels_as_tensor else lambda x: x)
    )
    scores = Tensor(s)
    nll = negative_log_likelihood(mg.log(mg.nnet.softmax(scores)), y_true).mean()
    nll.backward()

    cross_entropy_scores = Tensor(s)
    ce = softmax_crossentropy(cross_entropy_scores, y_true).mean()
    ce.backward()

    assert_allclose(nll.data, ce.data, atol=1e-5, rtol=1e-5)
    assert_allclose(scores.grad, cross_entropy_scores.grad, atol=1e-5, rtol=1e-5)


@given(data=st.data(), labels_as_tensor=st.booleans())
def test_weighted_negative_log_likelihood(data: st.DataObject, labels_as_tensor: bool):
    s = data.draw(
        hnp.arrays(
            shape=hnp.array_shapes(min_side=1, max_side=10, min_dims=2, max_dims=2),
            dtype=float,
            elements=st.floats(-100, 100),
        )
    )
    y_true = data.draw(
        hnp.arrays(
            shape=(s.shape[0],),
            dtype=hnp.integer_dtypes(),
            elements=st.integers(min_value=0, max_value=s.shape[1] - 1),
        ).map(Tensor if labels_as_tensor else lambda x: x)
    )
    weights = data.draw(
        hnp.arrays(
            shape=(s.shape[1],),
            dtype=float,
            elements=st.floats(1e-8, 100),
        )
    )
    scores = Tensor(s)
    weights = Tensor(weights)

    for score, y in zip(scores, y_true):
        score = mg.log(mg.nnet.softmax(score.reshape(1, -1)))
        y = y.reshape(-1)
        nll = negative_log_likelihood(score, y)
        weighted_nll = negative_log_likelihood(score, y, weights=weights)
        assert np.isclose(weighted_nll.data, weights[y.data].data * nll.data)
