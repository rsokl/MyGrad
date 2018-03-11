from mygrad import Tensor
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
from functools import wraps
from decimal import Decimal, getcontext

getcontext().prec = 14


def numerical_derivative(f, x, h=1e-8):
    """ Computes the numerical derivate of f(x) at `x`::

                  dfdx = (f(x + h) - f(x - h)) / (2h)

        Makes use of `decimal.Decimal` for high-precision arithmetic.

        Parameters
        ----------
        f : Callable[[Real], Real]
            A unary function: f(x)

        x : Decimal
            The value at at which the derivative is computed

        h : Real, optional (default=1e-8)
            Approximating infinitesimal.

        Returns
        -------
        Decimal
            df/dx @ `x` """

    h = Decimal(h)
    dx = (Decimal(f(x + h)) - Decimal(f(x - h))) / (Decimal(2) * h)
    return dx


class fwdprop_test_factory():
    def __init__(self, *, mygrad_func, true_func, xbnds=(-100, 100), no_go=()):

        self.op = mygrad_func
        self.true_func = true_func
        self.xbnds = xbnds
        self.no_go = no_go

    def __call__(self, f):
        @given(x=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=3),
                            dtype=float,
                            elements=st.floats(*self.xbnds)))
        @wraps(f)
        def wrapper(x):
            for value in self.no_go:
                assume(np.all(x != value))

            o = self.op(x)
            tensor_out = o.data
            true_out = self.true_func(x)
            assert isinstance(o, Tensor), "`mygrad_func` returned type {}, should return `mygrad.Tensor`".format(type(o))
            assert np.allclose(tensor_out, true_out), "`mygrad_func(x)` and `true_func(x)` produce different results"
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
        @given(grad=st.decimals(min_value=-100, max_value=100),
               x=st.decimals(min_value=self.xbnds[0], max_value=self.xbnds[1]))
        @wraps(f)
        def wrapper(grad, x):
            """ Performs hypothesis unit test for checking back-propagation
                through a `mygrad` op.

                Parameters
                ----------
                grad : Decimal
                    Gradient-value to be backpropped through this operation
                x : Decimal
                    The value at which df/dx is evaluated

                Raises
                ------
                AssertionError"""
            for x_val in self.no_go:
                assume(x != x_val)
            # gradient to be backpropped through this operation

            # sampled x coord, and df/fx evaluated at x (numerically)
            # both are of type Decimal
            dx = numerical_derivative(self.func, x)
            numerical_grad = float(dx * grad)

            # compute df/dx via mygrad
            var = Tensor(float(x))
            self.op(var).backward(float(grad))
            tensor_grad = var.grad.item()

            assert np.isclose(numerical_grad, tensor_grad, **self.tolerances), \
                "numerical derivative and mygrad derivative do not match"
        return wrapper