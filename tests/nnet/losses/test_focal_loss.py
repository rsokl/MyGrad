import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given

import mygrad as mg
from mygrad.nnet.activations import softmax
from mygrad.nnet.losses import focal_loss, softmax_focal_loss
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def numpy_focal_loss(
    scores: np.ndarray, targets: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    targets = mg.asarray(targets)
    rows = np.arange(len(scores))
    pc = scores[rows, targets]
    return -alpha * np.clip(1 - pc, a_min=0, a_max=1) ** gamma * np.log(pc)


def numpy_softmax_focal_loss(
    scores: np.ndarray, targets: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    targets = mg.asarray(targets)
    scores = softmax(scores).data
    rows = np.arange(len(scores))
    pc = scores[rows, targets]
    return -alpha * np.clip(1 - pc, a_min=0, a_max=1) ** gamma * np.log(pc)


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


@given(gamma=st.floats(max_value=0, exclude_max=True) | st.lists(st.floats()))
def test_raises_on_bad_gamma(gamma: float):
    with pytest.raises(ValueError):
        focal_loss(np.array([[1.0]]), np.array([0]), gamma=gamma)


@fwdprop_test_factory(
    mygrad_func=focal_loss,
    true_func=numpy_focal_loss,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=2)},
    index_to_bnds={0: (1e-14, 100)},
    num_arrays=1,
    kwargs=dict(
        targets=lambda scores: targets(scores=scores),
        alpha=lambda scores: st.floats(-2, 2),
        gamma=lambda scores: st.floats(0, 10),
    ),
)
def test_focal_fwd():
    pass


@backprop_test_factory(
    mygrad_func=focal_loss,
    true_func=numpy_focal_loss,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=2)},
    index_to_bnds={0: (1e-14, 1)},
    num_arrays=1,
    kwargs=dict(
        targets=lambda scores: targets(scores=scores),
        alpha=lambda scores: st.floats(-2, 2),
        gamma=lambda scores: st.floats(0, 10) | st.sampled_from([0.0, 1.0]),
    ),
    vary_each_element=True,
)
def test_focal_bkwd():
    pass


@fwdprop_test_factory(
    mygrad_func=softmax_focal_loss,
    true_func=numpy_softmax_focal_loss,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=2)},
    index_to_bnds={0: (1e-14, 100)},
    num_arrays=1,
    kwargs=dict(
        targets=lambda scores: targets(scores=scores),
        alpha=lambda scores: st.floats(-2, 2),
        gamma=lambda scores: st.floats(0, 10),
    ),
)
def test_softmax_focal_fwd():
    pass


@backprop_test_factory(
    mygrad_func=softmax_focal_loss,
    true_func=numpy_softmax_focal_loss,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=2, max_dims=2)},
    index_to_bnds={0: (1e-14, 1)},
    num_arrays=1,
    kwargs=dict(
        targets=lambda scores: targets(scores=scores),
        alpha=lambda scores: st.floats(-2, 2),
        gamma=lambda scores: st.floats(0, 10),
    ),
    vary_each_element=True,
)
def test_softmax_focal_bkwd():
    pass
