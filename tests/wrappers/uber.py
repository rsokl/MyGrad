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

        Randomly draw N arrays x, (...) to verify that a mygrad function,
        `f(x, ..., **kwargs)` returns a correct result through forward-propagation.

        By default, all arrays will have shapes that are broadcast-compatible with x.

        This decorator is extremely uber: the decorated function's body is
        never executed; this decorator builds the entire unit tests The
        function definition is effectively there to name the test being
        constructed. This decorator constructs the entire test.

        The user specifies the number of arrays to be generated (at least one must be
        generated), along with the bounds on their elements, their shapes, as well
        as the keyword-args to be passed to the function.

        Examples
        --------
        >>> from mygrad import add
        >>> import numpy as np
        >>> @fwdprop_test_factory(mygrad_func=add, true_func=np.add, num_array_args=2)
        ... def test_add():
        ...     pass"""
    def __init__(self, *,
                 mygrad_func,
                 true_func,
                 num_arrays,
                 index_to_bnds={},
                 index_to_no_go={},
                 kwargs={},
                 index_to_arr_shapes={}):
        """
        Parameters
        ----------
        mygrad_func : Callable[[numpy.ndarray, numpy.ndarray], mygrad.Tensor]
            The mygrad function whose forward pass validity is being checked.

        true_func : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
            A known correct version of the function

        index_to_bnds : Dict[int, Tuple[int, int]]
            Indicate the lower and upper bounds from which the elements
            for array-i is drawn. By default, [-10, 10].

        index_to_no_go : Dict[int, Sequence[int]]
            Values that array-i cannot possess. By default, no values are
            excluded.

        index_to_arr_shapes : Dict[i, Union[Sequence[int], hypothesis.searchstrategy.SearchStrategy]]
            The shape for array-i. This can be an exact shape or a hypothesis search
            strategy that draws shapes.
                Default for array-0: `hnp.array_shapes(max_side=3, max_dims=3)`
                Default for array-i: `broadcastable_shape(arr-0.shape)`

        kwargs : Dict[str, Union[Any, Callable[[Any], hypothesis.searchstrategy.SearchStrategy]]]
            Keyword arguments and their values to be passed to the functions.
            The values can be hypothesis search strategies, in which case
            a value when be drawn at test time for that argument.

            Note that any search strategy must be "wrapped" in a function, which
            will be called, passing it the list of arrays as an input argument, such
            that the strategy can draw based on those particular arrays.
        """
        assert num_arrays > 0
        self.op = mygrad_func
        self.true_func = true_func

        self.index_to_bnds = index_to_bnds
        self.index_to_no_go = index_to_no_go
        self.index_to_arr_shapes = index_to_arr_shapes
        self.kwargs = kwargs
        self.num_arrays = num_arrays

    def __call__(self, f):
        @given(x=hnp.arrays(shape=self.index_to_arr_shapes.get(0,
                                                               hnp.array_shapes(max_side=3, max_dims=3)),
                            dtype=float,
                            elements=st.floats(*self.index_to_bnds.get(0, (-10., 10.)))),
               data=st.data())
        @wraps(f)
        def wrapper(x, data):
            arrs = [x]
            for i in range(1, self.num_arrays):
                y = data.draw(hnp.arrays(shape=self.index_to_arr_shapes.get(i,
                                                                            broadcastable_shape(x.shape)),
                                         dtype=float,
                                         elements=st.floats(*self.index_to_bnds.get(i, (-10., 10.)))),
                              label="array-{}".format(i))
                arrs.append(y)

            arr_copies = [copy(arr) for arr in arrs]
            kwargs = {k: (data.draw(v(*arrs), label="kwarg: {}".format(k)) if callable(v) else v)
                      for k, v in self.kwargs.items()}

            for i, arr in enumerate(arrs):
                for value in self.index_to_no_go.get(i, ()):
                    assume(np.all(arr != value))

            o = self.op(*(Tensor(i) for i in arrs), **kwargs)
            tensor_out = o.data
            true_out = self.true_func(*arrs, **kwargs)
            assert isinstance(o, Tensor), "`mygrad_func` returned type {}, should return `mygrad.Tensor`".format(type(o))
            assert_allclose(tensor_out, true_out,
                            err_msg="`mygrad_func(x)` and `true_func(x)` produce different results")

            for n, (arr, arr_copy) in enumerate(zip(arrs, arr_copies)):
                assert_array_equal(arr, arr_copy,
                                   err_msg="arr-{} was mutated during forward prop".format(n))
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

    def __init__(self, *,
                 mygrad_func,
                 true_func,
                 num_arrays,
                 index_to_bnds={},
                 index_to_no_go={},
                 index_to_arr_shapes={},
                 index_to_unique={},
                 kwargs={},
                 h=1e-8,
                 rtol=1e-05,
                 atol=1e-08,
                 vary_each_element=False,
                 as_decimal=True):
        """
        Parameters
        ----------
        mygrad_func : Callable[[numpy.ndarray, numpy.ndarray], mygrad.Tensor]
            The mygrad function whose forward pass validity is being checked.

        true_func : Callable[[numpy.ndarray, numpy.ndarray], numpy.ndarray]
            A known correct version of the function

        index_to_bnds : Dict[int, Tuple[int, int]]
            Indicate the lower and upper bounds from which the elements
            for array-i is drawn. By default, [-10, 10].

        index_to_no_go : Dict[int, Sequence[int]]
            Values that array-i cannot possess. By default, no values are
            excluded.

        index_to_arr_shapes : Dict[i, Union[Sequence[int], hypothesis.searchstrategy.SearchStrategy]]
            The shape for array-i. This can be an exact shape or a hypothesis search
            strategy that draws shapes.
                Default for array-0: `hnp.array_shapes(max_side=3, max_dims=3)`
                Default for array-i: `broadcastable_shape(arr-0.shape)`

        kwargs : Dict[str, Union[Any, Callable[[Any], hypothesis.searchstrategy.SearchStrategy]]]
            Keyword arguments and their values to be passed to the functions.
            The values can be hypothesis search strategies, in which case
            a value when be drawn at test time for that argument.

            Note that any search strategy must be "wrapped" in a function, which
            will be called, passing it the list of arrays as an input argument, such
            that the strategy can draw based on those particular arrays.

        vary_each_element : bool, optional (default=False)
            If False, then use a faster numerical derivative that varies entire
            arrays at once: arr -> arr + h; valid only for functions that map over
            entries, like 'add' and 'sum'. Otherwise, the gradient is constructed
            by varying each element of each array independently.

        as_decimal : bool, optional (default=True)
            If True, x is passed to f as a Decimal-type array. This
            improves numerical precision, but is not permitted by some functions.
        """

        assert num_arrays > 0
        self.op = mygrad_func
        self.true_func = true_func

        if isinstance(index_to_bnds, (tuple, list, np.ndarray)):
            index_to_bnds = {k: index_to_bnds for k in range(num_arrays)}
        self.index_to_bnds = index_to_bnds

        if isinstance(index_to_no_go, (tuple, list, np.ndarray)):
            index_to_no_go = {k: index_to_no_go for k in range(num_arrays)}
        self.index_to_no_go = index_to_no_go

        if isinstance(index_to_arr_shapes, (tuple, list, np.ndarray, st.SearchStrategy)):
            index_to_arr_shapes = {k: index_to_arr_shapes for k in range(num_arrays)}
            self.index_to_arr_shapes = index_to_arr_shapes
        self.index_to_arr_shapes = index_to_arr_shapes

        if isinstance(index_to_unique, bool):
            index_to_unique = {k: index_to_unique for k in range(num_arrays)}
        self.index_to_unique = index_to_unique
        self.kwargs = kwargs
        self.num_arrays = num_arrays

        self.h = h
        self.tolerances = dict(rtol=rtol, atol=atol)
        self.vary_each_element = vary_each_element
        self.as_decimal = as_decimal

    def __call__(self, f):
        @given(x=hnp.arrays(shape=self.index_to_arr_shapes.get(0,
                                                               hnp.array_shapes(max_side=3, max_dims=3)),
                            dtype=float,
                            elements=st.floats(*self.index_to_bnds.get(0, (-10., 10.))),
                            unique=self.index_to_unique.get(0, False)),
               data=st.data())
        @wraps(f)
        def wrapper(x, data):

            arrs = [x]
            for i in range(1, self.num_arrays):
                y = data.draw(hnp.arrays(shape=self.index_to_arr_shapes.get(i,
                                                                            broadcastable_shape(x.shape)),
                                         dtype=float,
                                         elements=st.floats(*self.index_to_bnds.get(i, (-10., 10.)))),
                              label="array-{}".format(i))
                arrs.append(y)

            arrs = tuple(Tensor(arr) for arr in arrs)
            arr_copies = tuple(copy(arr) for arr in arrs)
            kwargs = {k: (data.draw(v(*arrs), label="kwarg: {}".format(k)) if callable(v) else v)
                      for k, v in self.kwargs.items()}

            for i, arr in enumerate(arrs):
                for value in self.index_to_no_go.get(i, ()):
                    assume(np.all(arr != value))

            # gradient to be backpropped through this operation
            print(kwargs)
            out = self.op(*arrs, **kwargs)

            grad = data.draw(hnp.arrays(shape=out.shape,
                                        dtype=float,
                                        elements=st.floats(-10, 10),
                                        unique=True),
                             label="grad")

            grad_copy = copy(grad)
            if any(out.shape != i.shape for i in arrs):
                # broadcasting occurred, must reduce `out` to scalar
                # first multiply by `grad` to simulate non-trivial back-prop
                (grad * out).sum().backward()
            else:
                out.backward(grad)

            numerical_grad = numerical_gradient if self.vary_each_element else numerical_gradient_full
            if self.vary_each_element:
                grads_numerical = numerical_gradient_full(self.true_func, *(i.data for i in arrs),
                                                          back_grad=grad, kwargs=kwargs,
                                                          as_decimal=self.as_decimal)
            else:
                grads_numerical = numerical_grad(self.true_func, *(i.data for i in arrs), back_grad=grad,
                                                 kwargs=kwargs, as_decimal=self.as_decimal)

            for n, (arr, d_num) in enumerate(zip(arrs, grads_numerical)):
                assert_allclose(arr.grad, d_num, **self.tolerances,
                                err_msg="arr-{}: numerical derivative and mygrad derivative do not match".format(n))

            out.null_gradients()
            assert all(i.grad is None for i in arrs), "null_gradients failed"

            for n, (arr, arr_copy) in enumerate(zip(arrs, arr_copies)):
                assert_array_equal(arr, arr_copy,
                                   err_msg="arr-{} was mutated during forward prop".format(n))
            assert_array_equal(grad, grad_copy,
                               err_msg="`grad` was mutated during backward prop")
        return wrapper