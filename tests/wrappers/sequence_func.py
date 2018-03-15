from ..utils.numerical_gradient import numerical_gradient_sequence
from ..custom_strategies import valid_axes

from mygrad import Tensor
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
from functools import wraps


class fwdprop_test_factory():
    def __init__(self, *, mygrad_func, true_func, xbnds=(-100, 100), x_no_go=()):

        self.op = mygrad_func
        self.true_func = true_func
        self.xbnds = xbnds
        self.x_no_go = x_no_go

    def __call__(self, f):
        @given(x=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                            dtype=float,
                            elements=st.floats(*self.xbnds),
                            unique=True),
               data=st.data(),
               keepdims=st.booleans())
        @wraps(f)
        def wrapper(x, data, keepdims):
            axis = data.draw(valid_axes(x.ndim))

            for value in self.x_no_go:
                assume(np.all(x != value))

            o = self.op(x, axis=axis, keepdims=keepdims)
            tensor_out = o.data
            true_out = self.true_func(x, axis=axis, keepdims=keepdims)

            assert isinstance(o, Tensor), "`mygrad_func` returned type {}, should return `mygrad.Tensor`".format(type(o))
            assert np.allclose(tensor_out, true_out), "`mygrad_func(x)` and `true_func(x)` produce different results"
        return wrapper


class backprop_test_factory():
    def __init__(self, *, mygrad_func, true_func, xbnds=(-1000, 1000),
                 x_no_go=(), h=1e-8, rtol=1e-05, atol=1e-08):

        self.op = mygrad_func
        self.func = true_func
        self.xbnds = xbnds
        self.x_no_go = x_no_go
        self.h = h
        self.tolerances = dict(rtol=rtol, atol=atol)

    def __call__(self, f):
        @given(x=hnp.arrays(shape=hnp.array_shapes(max_side=5, max_dims=4),
                            dtype=float,
                            elements=st.integers(*self.xbnds),
                            unique=True),
               data=st.data(),
               keepdims=st.booleans())
        @wraps(f)
        def wrapper(x, data, keepdims):
            """ Performs hypothesis unit test for checking back-propagation
                through a `mygrad` op.

                Raises
                ------
                AssertionError"""

            axis = data.draw(valid_axes(x.ndim))

            for value in self.x_no_go:
                assume(np.all(x != value))

            x = x.astype(float)
            x = Tensor(x)
            out = self.op(x, axis=axis, keepdims=keepdims)


            # gradient to be backpropped through this operation
            grad = data.draw(hnp.arrays(shape=out.shape,
                                        dtype=float,
                                        elements=st.floats(1, 10)))
            grad = grad

            out.backward(grad)
            my_grad = x.grad

            dx = numerical_gradient_sequence(self.func, x=x.data, back_grad=grad,
                                             axis=axis, keepdims=keepdims, h=self.h)

            assert np.allclose(my_grad, dx, **self.tolerances), \
                "x: numerical derivative and mygrad derivative do not match"

        return wrapper
