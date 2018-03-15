from ..utils.numerical_gradient import numerical_gradient
from ..custom_strategies import broadcastable_shape

from mygrad import Tensor
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
from functools import wraps

class fwdprop_test_factory():
    def __init__(self, *, mygrad_func, true_func, xbnds=(-100, 100),
                 ybnds=(-100, 100), x_no_go=(), y_no_go=()):

        self.op = mygrad_func
        self.true_func = true_func
        self.xbnds = xbnds
        self.ybnds = ybnds
        self.x_no_go = x_no_go
        self.y_no_go = y_no_go

    def __call__(self, f):
        @given(x=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                            dtype=float,
                            elements=st.floats(*self.xbnds)),
               data=st.data())
        @wraps(f)
        def wrapper(x, data):
            y = data.draw(hnp.arrays(shape=broadcastable_shape(x.shape),
                                     dtype=float,
                                     elements=st.floats(*self.ybnds)))
            for value in self.x_no_go:
                assume(np.all(x != value))

            for value in self.y_no_go:
                assume(np.all(y != value))

            o = self.op(x, y)
            tensor_out = o.data
            true_out = self.true_func(x, y)
            assert isinstance(o, Tensor), "`mygrad_func` returned type {}, should return `mygrad.Tensor`".format(type(o))
            assert np.allclose(tensor_out, true_out), "`mygrad_func(x)` and `true_func(x)` produce different results"
        return wrapper


class backprop_test_factory():
    def __init__(self, *, mygrad_func, true_func, xbnds=(-100, 100),
                 ybnds=(-100, 100), x_no_go=(), y_no_go=(), h=1e-8, rtol=1e-05, atol=1e-08):

        self.op = mygrad_func
        self.func = true_func
        self.xbnds = xbnds
        self.ybnds = ybnds
        self.x_no_go = x_no_go
        self.y_no_go = y_no_go
        self.h = h
        self.tolerances = dict(rtol=rtol, atol=atol)

    def __call__(self, f):
        @given(data=st.data(),
               x=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                            dtype=float,
                            elements=st.floats(*self.xbnds)))
        @wraps(f)
        def wrapper(data, x):
            """ Performs hypothesis unit test for checking back-propagation
                through a `mygrad` op.

                Raises
                ------
                AssertionError"""

            y = data.draw(hnp.arrays(shape=broadcastable_shape(x.shape),
                                     dtype=float,
                                     elements=st.floats(*self.ybnds)))

            for value in self.x_no_go:
                assume(np.all(x != value))

            for value in self.y_no_go:
                assume(np.all(y != value))

            # gradient to be backpropped through this operation
            x = Tensor(x)
            y = Tensor(y)
            out = self.op(x, y)

            grad = data.draw(hnp.arrays(shape=out.shape,
                                        dtype=float,
                                        elements=st.floats(1, 10)))

            if any(out.shape != i.shape for i in (x, y)):
                # broadcasting occurred, must reduce `out` to scalar
                # first multiply by `grad` to simulate non-trivial back-prop
                (grad * out).sum().backward()
            else:
                out.backward(grad)

            dx, dy = numerical_gradient(self.func, x.data, y.data, back_grad=grad)

            assert np.allclose(x.grad, dx, **self.tolerances), \
                "x: numerical derivative and mygrad derivative do not match"
            assert np.allclose(y.grad, dy, **self.tolerances), \
                "y: numerical derivative and mygrad derivative do not match"
        return wrapper
