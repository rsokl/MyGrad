import numpy as np
from numpy.testing import assert_array_equal
from mygrad.tensor_creation.funcs import *
from mygrad import Tensor
from hypothesis import given
import hypothesis.strategies as st


def check_tensor_array(tensor, array, constant):
    assert isinstance(tensor, Tensor)
    assert_array_equal(tensor.data, array)
    assert tensor.dtype is array.dtype
    assert tensor.constant is constant


@given(constant=st.booleans(),
       dtype=st.sampled_from((np.int32, np.float64)))
def test_all_tensor_creation(constant, dtype):
    x = np.array([1, 2, 3])

    e = empty((3, 2), dtype=dtype, constant=constant)
    assert e.shape == (3, 2)
    assert e.constant is constant

    e = empty_like(e, dtype=dtype, constant=constant)
    assert e.shape == (3, 2)
    assert e.constant is constant

    check_tensor_array(eye(3, dtype=dtype, constant=constant), 
                       np.eye(3, dtype=dtype), constant)

    check_tensor_array(identity(3, dtype=dtype, constant=constant), 
                       np.identity(3, dtype=dtype), constant)

    check_tensor_array(ones((4, 5, 6), dtype=dtype, constant=constant), 
                       np.ones((4, 5, 6), dtype=dtype), constant)

    check_tensor_array(ones_like(x, dtype=dtype, constant=constant), 
                       np.ones_like(x, dtype=dtype), constant)

    check_tensor_array(zeros((4, 5, 6), dtype=dtype, constant=constant), 
                       np.zeros((4, 5, 6), dtype=dtype), constant)

    check_tensor_array(zeros_like(x, dtype=dtype, constant=constant), 
                       np.zeros_like(x, dtype=dtype), constant)
    
    check_tensor_array(full((4, 5, 6), 5., dtype=dtype, constant=constant),
                       np.full((4, 5, 6), 5., dtype=dtype), constant)

    check_tensor_array(full_like(x, 5., dtype=dtype, constant=constant),
                       np.full_like(x, 5., dtype=dtype), constant)

    check_tensor_array(arange(3, 7, dtype=dtype, constant=constant),
                       np.arange(3, 7, dtype=dtype), constant)

    check_tensor_array(linspace(3, 7, dtype=dtype, constant=constant),
                       np.linspace(3, 7, dtype=dtype), constant)

    check_tensor_array(logspace(3, 7, dtype=dtype, constant=constant),
                       np.logspace(3, 7, dtype=dtype), constant)

    check_tensor_array(geomspace(3, 7, dtype=dtype, constant=constant),
                       np.geomspace(3, 7, dtype=dtype), constant)
