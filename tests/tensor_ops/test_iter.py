import hypothesis.extra.numpy as hnp
import numpy as np
import pytest

from mygrad.tensor_base import Tensor
from tests.utils import adds_constant_arg

from ..wrappers.uber import backprop_test_factory, fwdprop_test_factory


# Test https://github.com/rsokl/MyGrad/issues/210
def test_iter_over_0d_raises():
    x = Tensor(3)
    with pytest.raises(TypeError):
        sum(x)


@adds_constant_arg
def _sum(x):
    out = sum(x)
    if isinstance(x, Tensor):
        if not isinstance(out, Tensor):
            # Hack to deal with summing over an empty tensor.
            # `sum(Tensor([]))` returns 0, which is fine
            out = Tensor(float(out))
    else:
        out = np.asarray(out)  # ensure numpy-output is array
    return out


@fwdprop_test_factory(
    mygrad_func=_sum,
    true_func=_sum,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=1, min_side=0)},
)
def test_builtin_sum_fwd():
    """Ensures equivalent iter behavior of tensor and array"""


@backprop_test_factory(
    mygrad_func=_sum,
    true_func=_sum,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(min_dims=1, min_side=0)},
    assumptions=lambda x: x.shape[0] != 0,  # avoid sum(tensor) = 0 (int)
    vary_each_element=True,
)
def test_builtin_sum_bkwd():
    pass
