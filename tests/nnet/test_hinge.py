from mygrad.tensor_base import Tensor
from mygrad.nnet.losses import multiclass_hinge

import numpy as np
from numpy.testing import assert_allclose
import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp


@given(st.data())
def test_multiclass_hinge(data):
    """ Test the built-in implementation of multiclass hinge against the pure pygrad version"""
    s = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=10, min_dims=2, max_dims=2),
                             dtype=float,
                             elements=st.floats(-100, 100)))
    l = data.draw(hnp.arrays(shape=(s.shape[0],),
                             dtype=int,
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
