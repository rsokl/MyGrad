from ..custom_strategies import numerical_derivative

from mygrad import Tensor
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
from functools import wraps


class fwdprop_test_factory():
    def __init__(self, *, mygrad_func, true_func, xbnds=(-100, 100), no_go=()):

        self.op = mygrad_func
        self.true_func = true_func
        self.xbnds = xbnds
        self.no_go = no_go

    def __call__(self, f):
        @given(st.data())
        @wraps(f)
        def wrapper(data):
            x = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                                     dtype=float,
                                     elements=st.floats(*self.xbnds)))
            for value in self.no_go:
                assume(np.all(x != value))

            o = self.op(x)
            tensor_out = o.data
            true_out = self.true_func(x)
            assert isinstance(o, Tensor)
            assert np.allclose(tensor_out, true_out)
        return wrapper


class backprop_test_factory():
    def __init__(self, *, mygrad_func, true_func=None, xbnds=(-100, 100), no_go=(),
                 h=1e-8, rtol=1e-05, atol=1e-08):

        self.op = mygrad_func
        self.func = true_func if true_func is not None else lambda x: mygrad_func(float(x)).data.item()
        self.xbnds = xbnds
        self.no_go = no_go
        self.h = h
        self.tolerances = dict(rtol=rtol, atol=atol)

    def __call__(self, f):
        @given(st.data())
        @wraps(f)
        def wrapper(data):
            # gradient to be backpropped through this operation
            grad = data.draw(st.decimals(min_value=-100, max_value=100))

            # sampled x coord, and df/fx evaluated at x (numerically)
            # both are of type Decimal
            x, dx = data.draw(numerical_derivative(self.func, xbnds=self.xbnds, no_go=(0,)))
            numerical_grad = float(dx * grad)

            # compute df/dx via mygrad
            var = Tensor(float(x))
            self.op(var).backward(float(grad))
            tensor_grad = var.grad.item()

            assert np.isclose(numerical_grad, tensor_grad, **self.tolerances)
        return wrapper