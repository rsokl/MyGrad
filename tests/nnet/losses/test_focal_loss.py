import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import pytest

from mygrad import Tensor, sum, log
from mygrad.nnet.activations import softmax
from mygrad.nnet.losses import softmax_focal_loss, focal_loss


@pytest.mark.parametrize("fn", (softmax_focal_loss, focal_loss))
@pytest.mark.parametrize(
    ("data", "labels"),
    [
        (np.ones((2,), dtype=float), np.zeros((2,), dtype=int)),  # 1D data
        (np.ones((2, 1), dtype=float), np.zeros((2,), dtype=float)),  # non-int labels
        (np.ones((2, 1), dtype=float), np.zeros((2, 1), dtype=int)),  # bad label-ndim
        (np.ones((2, 1), dtype=float), np.zeros((3,), dtype=int)),  # bad label-shape
    ],
)
def test_input_validation(fn, data, labels):
    with pytest.raises((ValueError, TypeError)):
        fn(data, labels)


@given(
    num_datum=st.integers(1, 100),
    num_classes=st.integers(1, 15),
    alpha=st.floats(-1, 1),
    gamma=st.floats(.0, 5),
    data=st.data(),
    grad=st.floats(-1, 1),
)
def test_focal_loss(num_datum, num_classes, alpha, gamma, data, grad):
    scores = data.draw(
        hnp.arrays(shape=(num_datum, num_classes), dtype=float, elements=st.floats(1, 100))
    )
    assume((abs(scores.sum(axis=1)) > 0.001).all())

    scores_mygrad = Tensor(scores)
    scores_nn = Tensor(scores)

    truth = np.zeros((num_datum, num_classes))
    targets = data.draw(st.tuples(*(st.integers(0, num_classes - 1) for i in range(num_datum))))

    truth[range(num_datum), targets] = 1

    probs = softmax(scores_mygrad)
    mygrad_focal_loss = sum(truth * (-alpha * (1 - probs + 1e-14)**gamma * log(probs))) / num_datum
    mygrad_focal_loss.backward(grad)

    nn_loss = softmax_focal_loss(scores_nn, targets, alpha=alpha, gamma=gamma)
    nn_loss.backward(grad)

    assert isinstance(nn_loss, Tensor) and nn_loss.ndim == 0
    assert_allclose(nn_loss.data, mygrad_focal_loss.data, atol=1e-4, rtol=1e-4)
    assert_allclose(scores_nn.grad, scores_mygrad.grad, atol=1e-4, rtol=1e-4)

    nn_loss.null_gradients()
    assert scores_nn.grad is None
