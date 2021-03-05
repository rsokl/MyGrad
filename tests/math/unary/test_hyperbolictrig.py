import numpy as np

from mygrad import coth, csch, sech
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


def _is_nonzero(x):
    return np.all(np.abs(x.data) > 1e-8)


@fwdprop_test_factory(
    mygrad_func=csch,
    true_func=lambda x: 1 / np.sinh(x),
    index_to_bnds={0: (0.001, 10)},
    num_arrays=1,
)
def test_csch_fwd():
    pass


@backprop_test_factory(
    mygrad_func=csch,
    true_func=lambda x: 1 / np.sinh(x),
    index_to_bnds={0: (0.001, 10)},
    num_arrays=1,
)
def test_csch_backward():
    pass


@fwdprop_test_factory(
    mygrad_func=sech,
    true_func=lambda x: 1 / np.cosh(x),
    index_to_bnds={0: (-10, 10)},
    num_arrays=1,
)
def test_sech_fwd():
    pass


@backprop_test_factory(
    mygrad_func=sech,
    true_func=lambda x: 1 / np.cosh(x),
    index_to_bnds={0: (0.001, 10)},
    atol=1e-5,
    num_arrays=1,
)
def test_sech_backward():
    pass


@fwdprop_test_factory(
    mygrad_func=coth,
    true_func=lambda x: 1 / np.tanh(x),
    index_to_bnds={0: (-10, 10)},
    assumptions=_is_nonzero,
    num_arrays=1,
)
def test_coth_fwd():
    pass


@backprop_test_factory(
    mygrad_func=coth,
    true_func=lambda x: 1 / np.tanh(x),
    index_to_bnds={0: (0.001, 10)},
    atol=1e-5,
    num_arrays=1,
)
def test_coth_backward():
    pass
