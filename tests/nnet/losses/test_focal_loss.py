import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest

from mygrad.nnet.activations import softmax
from mygrad.nnet.losses import focal_loss, softmax_focal_loss
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def numpy_focal_loss(
    scores: np.ndarray, targets: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    rows = np.arange((len(scores)))
    pc = scores[rows, targets]
    return -alpha * np.clip(1 - pc, a_min=0, a_max=1) ** gamma * np.log(pc)


def numpy_softmax_focal_loss(
    scores: np.ndarray, targets: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    scores = softmax(scores).data
    rows = np.arange((len(scores)))
    pc = scores[rows, targets]
    return -alpha * np.clip(1 - pc, a_min=0, a_max=1) ** gamma * np.log(pc)


@st.composite
def targets(draw, scores):
    return draw(
        st.lists(
            st.integers(0, scores.shape[1] - 1),
            min_size=len(scores),
            max_size=len(scores),
        )
    )


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
        gamma=lambda scores: st.floats(0, 10),
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
