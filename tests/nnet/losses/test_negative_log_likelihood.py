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
        negative_log_likelihood(data, labels)


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
    nll = negative_log_likelihood(mg.log(mg.nnet.softmax(scores)), y_true)
    nll.backward()

    cross_entropy_scores = Tensor(s)
    ce = softmax_crossentropy(cross_entropy_scores, y_true)
    ce.backward()

    assert_allclose(nll.data, ce.data, atol=1e-5, rtol=1e-5)
    assert_allclose(scores.grad, cross_entropy_scores.grad, atol=1e-5, rtol=1e-5)
