import numpy as np

from mygrad import abs, absolute
from tests.wrappers.uber import backprop_test_factory


def _is_non_zero(x):
    return np.all(np.abs(x.data) > 1e-8)


@backprop_test_factory(
    mygrad_func=abs,
    true_func=np.abs,
    num_arrays=1,
    index_to_bnds={0: (-10, 10)},
    assumptions=_is_non_zero,
    atol=1e-5,
    use_finite_difference=True,
    h=1e-8,
)
def test_abs_backward():
    pass


@backprop_test_factory(
    mygrad_func=absolute,
    true_func=np.absolute,
    num_arrays=1,
    index_to_bnds={0: (-100, 100)},
    index_to_no_go={0: (0,)},
    atol=1e-5,
    assumptions=_is_non_zero,
    use_finite_difference=True,
    h=1e-8,
)
def test_absolute_backward():
    pass
