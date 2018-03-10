from mygrad.tensor_base import Tensor
from mygrad.nnet.losses import softmax_crossentropy
from mygrad.nnet.activations import softmax
from mygrad.math import log

import numpy as np
import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp


@given(st.data())
def test_softmax_crossentropy(data):
    """ Test the built-in implementation of multiclass hinge against the pure pygrad version"""
    s = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=10, min_dims=2, max_dims=2),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    l = data.draw(hnp.arrays(shape=(s.shape[0],),
                             dtype=int,
                             elements=st.integers(min_value=0, max_value=s.shape[1] - 1)))
    scores = Tensor(s)
    softmax_cross = softmax_crossentropy(scores, l)
    softmax_cross.backward()

    pygrad_scores = Tensor(s)
    probs = softmax(pygrad_scores)

    correct_labels = (range(len(l)), l)
    truth = np.zeros(pygrad_scores.shape)
    truth[correct_labels] = 1

    pygrad_cross = (-1/s.shape[0]) * (log(probs) * truth).sum()
    pygrad_cross.backward()
    assert np.allclose(softmax_cross.data, pygrad_cross.data)
    assert np.allclose(scores.grad, pygrad_scores.grad)
