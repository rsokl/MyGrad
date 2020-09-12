import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose
from pytest import raises

import mygrad as mg
from mygrad import log
from mygrad.nnet.activations import softmax
from mygrad.nnet.losses import softmax_crossentropy
from mygrad.tensor_base import Tensor
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@st.composite
def targets(draw, scores):
    as_tensor = draw(st.booleans())
    out = draw(
        st.lists(
            st.integers(0, scores.shape[1] - 1),
            min_size=len(scores),
            max_size=len(scores),
        )
    )
    return mg.Tensor(out) if as_tensor else np.array(out)


@fwdprop_test_factory(
    mygrad_func=softmax_crossentropy,
    true_func=mg.no_autodiff(softmax_crossentropy),
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=2)},
    index_to_bnds={0: (1e-14, 100)},
    num_arrays=1,
    kwargs=dict(y_true=lambda scores: targets(scores=scores),),
)
def test_softmax_crossentropy_fwd():
    pass


@backprop_test_factory(
    mygrad_func=softmax_crossentropy,
    true_func=mg.no_autodiff(softmax_crossentropy, to_numpy=True),
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=2)},
    index_to_bnds={0: (1e-14, 100)},
    num_arrays=1,
    kwargs=dict(y_true=lambda scores: targets(scores=scores),),
    vary_each_element=True,
)
def test_softmax_crossentropy_bkwd():
    pass


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
        softmax_crossentropy(data, labels)


@given(data=st.data(), labels_as_tensor=st.booleans())
def test_softmax_crossentropy_via_mygrad_ops(
    data: st.DataObject, labels_as_tensor: bool
):
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
    softmax_cross = softmax_crossentropy(scores, y_true, constant=False)
    softmax_cross.backward()

    mygrad_scores = Tensor(s)
    probs = softmax(mygrad_scores)

    correct_labels = (range(len(y_true)), y_true.data if labels_as_tensor else y_true)
    truth = np.zeros(mygrad_scores.shape)
    truth[correct_labels] = 1

    mygrad_cross = (-1 / s.shape[0]) * (log(probs) * truth).sum()
    mygrad_cross.backward()
    assert_allclose(softmax_cross.data, mygrad_cross.data, atol=1e-5, rtol=1e-5)
    assert_allclose(scores.grad, mygrad_scores.grad, atol=1e-5, rtol=1e-5)


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
