import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from numpy.testing import assert_allclose

from mygrad import maximum, minimum
from mygrad.tensor_base import Tensor
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def is_not_close(arr0: Tensor, arr1: Tensor) -> bool:
    return not np.any(np.isclose(arr0.data, arr1.data))


@fwdprop_test_factory(mygrad_func=maximum, true_func=np.maximum, num_arrays=2)
def test_maximum_fwd():
    pass


@backprop_test_factory(
    mygrad_func=maximum, true_func=np.maximum, num_arrays=2, assumptions=is_not_close
)
def test_maximum_bkwd():
    pass


def test_maximum_bkwd_equal():
    """ regression test for documented behavior of maximum/minimum where
        x == y"""

    x = Tensor([1.0, 0.0, 2.0])
    y = Tensor([2.0, 0.0, 1.0])

    o = maximum(x, y)
    o.backward()

    assert_allclose(x.grad, [0.0, 0.0, 1])
    assert_allclose(y.grad, [1.0, 0.0, 0])
    o.null_gradients()

    # ensure branch covered for equal scalars
    x = Tensor(1.0)
    y = Tensor(1.0)

    o = maximum(x, y)
    o.backward()

    assert_allclose(x.grad, 0.0)
    assert_allclose(y.grad, 0.0)
    o.null_gradients()


@fwdprop_test_factory(mygrad_func=minimum, true_func=np.minimum, num_arrays=2)
def test_minimum_fwd():
    pass


@backprop_test_factory(
    mygrad_func=minimum, true_func=np.minimum, num_arrays=2, assumptions=is_not_close
)
def test_minimum_bkwd():
    pass


def test_minimum_bkwd_equal():
    """ regression test for documented behavior of minimum/minimum where
        x == y"""

    x = Tensor([1.0, 0.0, 2.0])
    y = Tensor([2.0, 0.0, 1.0])

    o = minimum(x, y)
    o.backward()

    assert_allclose(x.grad, [1.0, 0.0, 0.0])
    assert_allclose(y.grad, [0.0, 0.0, 1.0])
    o.null_gradients()

    # ensure branch covered for equal scalars
    x = Tensor(1.0)
    y = Tensor(1.0)

    o = minimum(x, y)
    o.backward()

    assert_allclose(x.grad, 0.0)
    assert_allclose(y.grad, 0.0)
    o.null_gradients()


def to_min_max(arr: np.ndarray) -> st.SearchStrategy:
    bnd_shape = hnp.broadcastable_shapes(
        shape=arr.shape, max_dims=arr.ndim, max_side=min(arr.shape) if arr.ndim else 1
    )
    bnd_strat = hnp.arrays(
        shape=bnd_shape, elements=st.floats(-1e6, 1e6), dtype=np.float64
    )
    return st.fixed_dictionaries(dict(a_min=bnd_strat, a_max=bnd_strat))
