import mygrad as mg
from mygrad.tensor_base import Tensor
from mygrad.nnet.losses import multiclass_hinge

import numpy as np
from numpy.testing import assert_allclose
import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

from pytest import raises

@given(st.data())
def test_multiclass_hinge(data):
    """ Test the built-in implementation of multiclass hinge against the pure pygrad version"""
    s = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=10, min_dims=2, max_dims=2),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    l = data.draw(hnp.arrays(shape=(s.shape[0],),
                             dtype=hnp.integer_dtypes(),
                             elements=st.integers(min_value=0, max_value=s.shape[1] - 1)))
    hinge_scores = Tensor(s)
    hinge_loss = multiclass_hinge(hinge_scores, l, constant=False)
    hinge_loss.backward()

    pygrad_scores = Tensor(s)
    correct_labels = (range(len(l)), l)
    correct_class_scores = pygrad_scores[correct_labels]  # Nx1

    Lij = pygrad_scores - correct_class_scores[:, np.newaxis] + 1.  # NxC margins
    Lij[Lij <= 0] = 0
    Lij[correct_labels] = 0

    pygrad_loss = Lij.sum() / pygrad_scores.shape[0]
    pygrad_loss.backward()
    assert_allclose(hinge_loss.data, pygrad_loss.data)
    assert_allclose(pygrad_scores.grad, hinge_scores.grad)


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
