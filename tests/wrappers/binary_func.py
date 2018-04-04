from ..utils.numerical_gradient import numerical_gradient_full, numerical_gradient
from ..custom_strategies import broadcastable_shape

from mygrad import Tensor

from copy import copy

from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from functools import wraps


class fwdprop_test_factory():
    """ Decorator

        Randomly draw arrays x and y, to verify that a binary mygrad function,
        `f(x, y, **kwargs)` returns a correct result.

        This decorator is extremely uber: the decorated function's body is
        never executed. The function definition is effectively there to name
        the test being constructed. This constructs the entire test

        Examples
        --------
        >>> from mygrad import add
        >>> import numpy as np
        >>> @fwdprop_test_factory(mygrad_func=add, true_func=np.add)
        ... def test_add():
        ...     pass"""
    def __init__(self, *, mygrad_func, true_func, xbnds=(-100, 100),
                 ybnds=(-100, 100), x_no_go=(), y_no_go=(),
                 kwargs={}):
        """
        Parameters
        ----------
        mygrad_func : Callable[[numpy.ndarray, numpy.ndarray], mygrad.Tensor]
            The mygrad function whose forward pass validity is being checked.
        true_func : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
            A known correct version of the function
        xbnds : Tuple[int, int], optional (default=(-100, 100))
            Bounds for values of x
        ybnds : Tuple[int, int], optional (default=(-100, 100))
            Permissible bounds for values of y
        x_no_go : Tuple[int, ...]
            Values that x cannot posses
        y_no_go : Tuple[int, ...]
            Values that y cannot possess
        kwargs : Dict[str, Union[hypothesis.searchstrategy.SearchStrategy, Any]]
            Keyword arguments and their values to be passed to the functions.
            The values can be hypothesis search strategies, in which case
            a value when be drawn at test time for that argument.
        """
        self.op = mygrad_func
        self.true_func = true_func
        self.xbnds = xbnds
        self.ybnds = ybnds
        self.x_no_go = x_no_go
        self.y_no_go = y_no_go
        self.kwargs = kwargs

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
            x_copy = copy(x)
            y_copy = copy(y)

            S = st.SearchStrategy
            kwargs = {k: (data.draw(v) if isinstance(v, S) else v) for k, v in self.kwargs}

            for value in self.x_no_go:
                assume(np.all(x != value))

            for value in self.y_no_go:
                assume(np.all(y != value))

            o = self.op(x, y, **kwargs)
            tensor_out = o.data
            true_out = self.true_func(x, y, **kwargs)
            assert isinstance(o, Tensor), "`mygrad_func` returned type {}, should return `mygrad.Tensor`".format(type(o))
            assert_allclose(tensor_out, true_out,
                            err_msg="`mygrad_func(x)` and `true_func(x)` produce different results")

            assert_array_equal(x, x_copy,
                               err_msg="`x` was mutated during forward prop")
            assert_array_equal(y, y_copy,
                               err_msg="`y` was mutated during forward prop")
        return wrapper


class backprop_test_factory():
    """ Decorator

        Randomly draw arrays x and y, to verify that a binary mygrad function,
        `f(x, y, **kwargs)` performs backpropagation appropriately.

        x.grad and y.grad are compared against numerical derivatives of f.
        THe numerical derivative is arrived at

        **IMPORTANT**
        f must be a trivial mapping over the individual parameters of x and y,
        such as vectorized add or multiply. An example of an invalid

        This decorator is extremely uber: the decorated function's body is
        never executed. The function definition is effectively there to name
        the test being constructed. This constructs the entire test

        Examples
        --------
        >>> from mygrad import add
        >>> import numpy as np
        >>> @backprop_test_factory(mygrad_func=add, true_func=np.add)
        ... def test_add():
        ...     pass"""

    def __init__(self, *, mygrad_func, true_func, xbnds=(-100, 100),
                 ybnds=(-100, 100),
                 x_no_go=(),
                 y_no_go=(),
                 h=1e-8,
                 rtol=1e-05,
                 atol=1e-08,
                 func_is_mapping=True,
                 as_decimal=True,
                 kwargs={}):
        """
        Parameters
        ----------
        mygrad_func : Callable[[numpy.ndarray, numpy.ndarray], mygrad.Tensor]
            The mygrad function whose forward pass validity is being checked.

        true_func : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
            A known correct version of the function

        xbnds : Tuple[int, int], optional (default=(-100, 100))
            Bounds for values of x

        ybnds : Tuple[int, int], optional (default=(-100, 100))
            Permissible bounds for values of y

        x_no_go : Tuple[float, ...]
            Values that x cannot posses

        y_no_go : Tuple[float, ...]
            Values that y cannot possess

        kwargs : Dict[str, Union[hypothesis.searchstrategy.SearchStrategy, Any]]
            Keyword arguments and their values to be passed to the functions.
            The values can be hypothesis search strategies, in which case
            a value when be drawn at test time for that argument.

        func_is_mapping : bool, optional (default=True)
            If True, then use a faster numerical derivative that varies entire
            arrays at once; valid only for functions that map over entries, like
            'add' and 'sum'.

        as_decimal : bool, optional (default=False)
            If False, x is passed to f as a Decimal-type array. This
            improves numerical precision, but is not permitted by some functions.
        """

        self.op = mygrad_func
        self.func = true_func
        self.xbnds = xbnds
        self.ybnds = ybnds
        self.x_no_go = x_no_go
        self.y_no_go = y_no_go
        self.h = h
        self.tolerances = dict(rtol=rtol, atol=atol)
        self.func_is_mapping = func_is_mapping
        self.as_decimal = as_decimal
        self.kwargs = kwargs

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

            S = st.SearchStrategy
            kwargs = {k: (data.draw(v) if isinstance(v, S) else v) for k, v in self.kwargs}

            # gradient to be backpropped through this operation
            x = Tensor(x)
            y = Tensor(y)
            out = self.op(x, y)

            grad = data.draw(hnp.arrays(shape=out.shape,
                                        dtype=float,
                                        elements=st.floats(1, 10)))

            x_copy = copy(x)
            y_copy = copy(y)
            grad_copy = copy(grad)
            if any(out.shape != i.shape for i in (x, y)):
                # broadcasting occurred, must reduce `out` to scalar
                # first multiply by `grad` to simulate non-trivial back-prop
                (grad * out).sum().backward()
            else:
                out.backward(grad)

            numerical_grad = numerical_gradient if self.func_is_mapping else numerical_gradient_full
            if self.func_is_mapping:
                dx, dy = numerical_grad(self.func, x.data, y.data, back_grad=grad, kwargs=kwargs)
            else:
                dx, dy = numerical_gradient_full(self.func, x.data, y.data,
                                                 back_grad=grad, kwargs=kwargs,
                                                 as_decimal=self.as_decimal)

            assert_allclose(x.grad, dx, **self.tolerances,
                            err_msg="x: numerical derivative and mygrad derivative do not match")
            assert_allclose(y.grad, dy, **self.tolerances,
                            err_msg="y: numerical derivative and mygrad derivative do not match")

            out.null_gradients()
            assert all(i.grad is None for i in (x, y)), "null_gradients failed"

            assert_array_equal(x, x_copy,
                               err_msg="`x` was mutated during backward prop")
            assert_array_equal(y, y_copy,
                               err_msg="`y` was mutated during backward prop")
            assert_array_equal(grad, grad_copy,
                               err_msg="`grad` was mutated during backward prop")
        return wrapper
