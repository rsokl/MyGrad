import hypothesis.strategies as st
import numpy as np
from hypothesis import settings
from numpy.testing import assert_almost_equal

from mygrad import matmul
from tests.custom_strategies import broadcastable_shape
from tests.wrappers.uber import backprop_test_factory, fwdprop_test_factory


@st.composite
def special_shape(draw, static_shape, shape=tuple(), min_dim=0, max_dim=5):
    """ search strategy that permits broadcastable dimensions to
        be prepended to a static shape - for the purposes of 
        drawing diverse shaped-arrays for matmul
        
        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy
            -> Tuple[int, ...]"""
    return draw(broadcastable_shape(shape, min_dim, max_dim)) + static_shape


@fwdprop_test_factory(mygrad_func=matmul, true_func=np.matmul,
                      num_arrays=2, index_to_arr_shapes={0: (2, 3),
                                                         1: special_shape((3, 4))})
def test_matmul_fwd():
    """ a is n-d, b is n-d, a can broadcast into b"""

def test_matmul_fwd_static():
    a = [4., 3.]
    b = [1.3, 4.]
    assert_almost_equal(actual=matmul(a, b).data, desired=np.matmul(a, b))

    a = [[4., 3.], [1., 2.]]
    b = [4., 3.]
    assert_almost_equal(actual=matmul(a, b).data, desired=np.matmul(a, b))


@backprop_test_factory(mygrad_func=matmul, true_func=np.matmul, num_arrays=2,
                       index_to_arr_shapes={0: (4,), 1: (4,)},
                       vary_each_element=True,
                       index_to_bnds={0: (-10, 10), 1: (-10, 10)})
def test_matmul_bkwd_1d_1d():
    """ a is 1-d, b is 1-d"""


@backprop_test_factory(mygrad_func=matmul, true_func=np.matmul, num_arrays=2, 
                       index_to_arr_shapes={0: special_shape((4,), min_dim=1, max_dim=2), 
                                            1: (4,)},
                       vary_each_element=True,
                       index_to_bnds={0: (-10, 10), 1: (-10, 10)})
def test_matmul_bkwd_nd_1d():
    """ a is n-d, b is 1-d"""


@backprop_test_factory(mygrad_func=matmul, true_func=np.matmul, num_arrays=2, 
                       index_to_arr_shapes={1: special_shape((4, 1), min_dim=1, max_dim=2), 
                                            0: (4,)},
                       vary_each_element=True,
                       index_to_bnds={0: (-10, 10), 1: (-10, 10)})
def test_matmul_bkwd_1d_nd():
    """ a is 1-d, b is n-d"""


@settings(deadline=None)
@backprop_test_factory(mygrad_func=matmul, true_func=np.matmul, num_arrays=2, 
                       index_to_arr_shapes={0: special_shape((4,), min_dim=1, max_dim=2), 
                                            1: (4, 5)},
                       vary_each_element=True,
                       index_to_bnds={0: (-10, 10), 1: (-10, 10)})
def test_matmul_bkwd_nd_nd():
    """ a is n-d, b is n-d; b can broadcast into a"""


@settings(deadline=None)
@backprop_test_factory(mygrad_func=matmul, true_func=np.matmul, num_arrays=2, 
                       index_to_arr_shapes={0: (2, 4), 
                                            1: special_shape((4, 5), max_dim=2)},
                       vary_each_element=True,
                       index_to_bnds={0: (-10, 10), 1: (-10, 10)})
def test_matmul_bkwd_nd_nd2():
    """ a is n-d, b is n-d; a can broadcast into b"""


@settings(deadline=None)
@backprop_test_factory(mygrad_func=matmul, true_func=np.matmul, num_arrays=2, 
                       index_to_arr_shapes={0: (2, 1, 3, 4), 
                                            1: (1, 2, 4, 2)},
                       vary_each_element=True,
                       index_to_bnds={0: (-10, 10), 1: (-10, 10)})
def test_matmul_bkwd_nd_nd3():
    """ a is n-d, b is n-d; a and b broadcast mutually via singleton dimensions"""
