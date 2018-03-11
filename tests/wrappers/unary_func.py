from mygrad import Tensor
from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np
from functools import wraps
from decimal import Decimal, getcontext

getcontext().prec = 14

def numerical_derivative(f, x, h=1e-8):
    """ Hypothesis search strategy: Sample x from specified bounds,
        and compute::

                  dfdx = (f(x + h) - f(x - h)) / (2h)

        Returning a search-strategy for: (x, dfdx)

        Makes use of `decimal.Decimal` for high-precision arithmetic.

        Note: The parameter `draw` is reserved for use by `hypothesis` - thus it
              it excluded from the function signature.

        Parameters
        ----------
        f : Callable[[Real], Real]
            A differentiable unary function: f(x)

        xbnds : Tuple[Real, Real], optional (default=(-100, 100))
            Defines the domain bounds (inclusive) from which `x` is drawn.

        no_go : Iterable[Real, ...], optional (default=())
            An iterable of values from which `x` will not be drawn.

        h : Real, optional (default=1e-8)
            Approximating infinitesimal.

        Returns
        -------
        Decimal
            -> Tuple[decimals.Decimal, decimals.Decimal]
            (x, df/dx) """
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
        @given(grad=st.decimals(min_value=-100, max_value=100),
               x=st.decimals(min_value=self.xbnds[0], max_value=self.xbnds[1]))
        @wraps(f)
        def wrapper(grad, x):
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

            assert np.isclose(numerical_grad, tensor_grad, **self.tolerances)
        return wrapper