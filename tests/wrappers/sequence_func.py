from ..utils.numerical_gradient import numerical_gradient_sequence
from ..custom_strategies import valid_axes

from mygrad import Tensor
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from copy import copy

from functools import wraps


class fwdprop_test_factory():
    def __init__(self, *, mygrad_func, true_func,
                 xbnds=(-100, 100),
                 x_no_go=(),
                 max_dims=4,
                 max_side=5,
                 unique=False,
                 no_axis=False,
                 no_keepdims=False,
                 single_axis_only=False,
                 draw_from_int=True):

        self.op = mygrad_func
        self.true_func = true_func
        self.xbnds = xbnds
        self.x_no_go = x_no_go
        self.max_dims = max_dims
        self.max_side = max_side
        self.no_keepdims = no_keepdims
        self.no_axis = no_axis
        self.unique = unique
        self.single_axis_only = single_axis_only
        self.draw_from = st.integers if draw_from_int else st.floats

    def __call__(self, f):
        @given(x=hnp.arrays(shape=hnp.array_shapes(max_side=self.max_side,
                                                   max_dims=self.max_dims),
                            dtype=float,
                            elements=st.floats(*self.xbnds),
                            unique=self.unique),
               data=st.data())
        @wraps(f)
        def wrapper(x, data):
            x_copy = copy(x)
            axis = np.nan if self.no_axis else data.draw(valid_axes(x.ndim,
                                                                    single_axis_only=self.single_axis_only))
            keepdims = np.nan if self.no_keepdims else data.draw(st.booleans())

            for value in self.x_no_go:
                assume(np.all(x != value))

            kwargs = dict()
            if not self.no_axis:
                kwargs["axis"] = axis
            if not self.no_keepdims:
                kwargs["keepdims"] = keepdims

            o = self.op(x, **kwargs)
            tensor_out = o.data
            true_out = self.true_func(x, **kwargs)

            assert_array_equal(x, x_copy,
                               err_msg="`x` was mutated during forward prop")
            assert isinstance(o, Tensor), "`mygrad_func` returned type {}, should return `mygrad.Tensor`".format(type(o))
            assert_allclose(tensor_out, true_out,
                            err_msg="`mygrad_func(x)` and `true_func(x)` produce different results")
        return wrapper


class backprop_test_factory():
    def __init__(self, *, mygrad_func, true_func,
                 xbnds=(-1000, 1000),
                 x_no_go=(),
                 h=1e-8,
                 rtol=1e-05,
                 atol=1e-08,
                 max_dims=4,
                 max_side=5,
                 unique=False,
                 no_axis=False,
                 no_keepdims=False,
                 single_axis_only=False,
                 draw_from_int=True):

        self.op = mygrad_func
        self.func = true_func
        self.xbnds = xbnds
        self.x_no_go = x_no_go
        self.h = h
        self.tolerances = dict(rtol=rtol, atol=atol)
        self.max_dims = max_dims
        self.max_side = max_side
        self.no_keepdims = no_keepdims
        self.no_axis = no_axis
        self.unique = unique
        self.single_axis_only = single_axis_only
        self.draw_from = st.integers if draw_from_int else st.floats

    def __call__(self, f):
        @given(x=hnp.arrays(shape=hnp.array_shapes(max_side=self.max_side,
                                                   max_dims=self.max_dims),
                            dtype=float,
                            elements=self.draw_from(*self.xbnds),
                            unique=self.unique),
               data=st.data())
        @wraps(f)
        def wrapper(x, data):
            """ Performs hypothesis unit test for checking back-propagation
                through a `mygrad` op.

                Raises
                ------
                AssertionError"""

            x_copy = copy(x)
            axis = np.nan if self.no_axis else data.draw(valid_axes(x.ndim,
                                                                    single_axis_only=self.single_axis_only))
            keepdims = np.nan if self.no_keepdims else data.draw(st.booleans())

            for value in self.x_no_go:
                assume(np.all(x != value))

            x = x.astype(float)
            x = Tensor(x)

            # some sequential functions do not accep
            kwargs = dict()
            if not self.no_axis:
                kwargs["axis"] = axis
            if not self.no_keepdims:
                kwargs["keepdims"] = keepdims

            out = self.op(x, **kwargs)

            # gradient to be backpropped through this operation
            grad = data.draw(hnp.arrays(shape=out.shape,
                                        dtype=float,
                                        elements=st.floats(1, 10)))
            grad_copy = copy(grad)

            out.backward(grad)
            my_grad = x.grad

            dx = numerical_gradient_sequence(self.func, x=x.data, back_grad=grad,
                                             axis=axis, keepdims=keepdims, h=self.h,
                                             no_keepdims=self.no_keepdims, no_axis=self.no_axis)

            assert_array_equal(grad, grad_copy,
                               err_msg="`grad` was mutated during backprop")
            assert_array_equal(x, x_copy,
                               err_msg="`x` was mutated during backprop")
            assert_allclose(my_grad, dx,
                            err_msg="x: numerical derivative and mygrad derivative do not match",
                            **self.tolerances)


        return wrapper
