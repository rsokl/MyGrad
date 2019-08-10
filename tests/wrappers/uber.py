from copy import copy
from functools import wraps
from itertools import combinations
from numbers import Real
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import assume, given, note
from hypothesis.strategies import SearchStrategy
from numpy.testing import assert_allclose, assert_array_equal

from mygrad import Tensor

from ..custom_strategies import broadcastable_shape
from ..utils.numerical_gradient import (
    finite_difference,
    numerical_gradient,
    numerical_gradient_full,
)


class fwdprop_test_factory:
    """ Decorator

        Randomly draw N arrays x, (...) to verify that a mygrad function,
        `f(x, ..., **kwargs)` returns a correct result through forward-propagation.

        By default, all arrays will have shapes that are broadcast-compatible with x.

        This decorator is extremely uber: the decorated function's body is
        never executed; this decorator builds the entire unit tests The
        function definition is effectively there to name the test being
        constructed.

        The user specifies the number of arrays to be generated (at least one must be
        generated), along with the bounds on their elements, their shapes, as well
        as the keyword-args to be passed to the function.

        Examples
        --------
        Writing a test that compares mygrad's add to numpy's add

        >>> from mygrad import add
        >>> import numpy as np
        >>> @fwdprop_test_factory(mygrad_func=add, true_func=np.add, num_array_args=2)
        ... def test_add():
        ...     pass
        """

    def __init__(
        self,
        *,
        mygrad_func: Callable[[Tensor], Tensor],
        true_func: Callable[[np.ndarray], np.ndarray],
        num_arrays: int,
        index_to_bnds: Dict[int, Tuple[int, int]] = {},
        index_to_no_go: Dict[int, Sequence[int]] = {},
        kwargs: Union[
            Callable, Dict[str, Union[Any, Callable[[Any], SearchStrategy]]]
        ] = {},
        index_to_arr_shapes: Dict[int, Union[Sequence[int], SearchStrategy]] = {},
        assumptions: Optional[Callable[..., bool]] = None
    ):
        """
        Parameters
        ----------
        mygrad_func : Callable[[numpy.ndarray, ...], mygrad.Tensor]
            The mygrad function whose forward pass validity is being checked.

        true_func : Callable[[numpy.ndarray, ...], numpy.ndarray]
            A known correct version of the function

        index_to_bnds : Dict[int, Tuple[int, int]]
            Indicate the lower and upper bounds from which the elements
            for array-i is drawn. By default, [-10, 10].

        index_to_no_go : Dict[int, Sequence[int]]
            Values that array-i cannot possess. By default, no values are
            excluded.

        index_to_arr_shapes : Dict[int, Union[Sequence[int], hypothesis.searchstrategy.SearchStrategy]]
            The shape for array-i. This can be an exact shape or a hypothesis search
            strategy that draws shapes.
                Default for array-0: `hnp.array_shapes(max_side=3, max_dims=3)`
                Default for array-i: `broadcastable_shape(arr-0.shape)`

        kwargs : Union[Callable, Dict[str, Union[Any, Callable[[Any], SearchStrategy]]]]
            Keyword arguments and their values to be passed to the functions.
            The values can be hypothesis search-strategies, in which case
            a value when be drawn at test time for that argument using the provided
            strategy.

            Note that any search strategy must be "wrapped" in a function, which
            will be called, passing it the list of arrays as an input argument, such
            that the strategy can draw based on those particular arrays.

        assumptions : Optional[Callable[[arrs, **kwargs], bool]]
            A callable that is fed the generated arrays and keyword arguments that will
            be fed to ``mygrad_func``. If ``assumptions`` returns ``False``, that test
            case will be marked as skipped by hypothesis.
        """
        assert num_arrays > 0
        self.op = mygrad_func
        self.true_func = true_func

        self.index_to_bnds = index_to_bnds
        self.index_to_no_go = index_to_no_go
        self.index_to_arr_shapes = index_to_arr_shapes
        self.kwargs = kwargs
        self.num_arrays = num_arrays
        self.assumptions = assumptions

    def gen_first_array(self) -> st.SearchStrategy:
        """
        Hypothesis search strategy for drawing the array x to be passed to f(x, ...)

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy"""
        return hnp.arrays(
            shape=self.index_to_arr_shapes.get(
                0, hnp.array_shapes(max_side=3, min_dims=0, max_dims=3)
            ),
            dtype=float,
            elements=st.floats(*self.index_to_bnds.get(0, (-10.0, 10.0))),
        )

    def gen_other_array(self, x: np.ndarray, i: int) -> st.SearchStrategy:
        """
        Hypothesis search strategy for drawing an array y to be passed to f(x, ..., y_i,...).
        By default, y is drawn to have a shape that is broadcast-compatible with x.

        Parameters
        ----------
        x : numpy.ndarray
        i : int
            The argument index-location of y in the signature of f.

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy"""
        return hnp.arrays(
            shape=self.index_to_arr_shapes.get(i, broadcastable_shape(x.shape)),
            dtype=float,
            elements=st.floats(*self.index_to_bnds.get(i, (-10.0, 10.0))),
        )

    def __call__(self, f):
        @given(x=self.gen_first_array(), constant=st.booleans(), data=st.data())
        @wraps(f)
        def wrapper(x, constant, data):
            arrs = [x]  # list of drawn arrays to feed to functions

            for i in range(
                1, self.num_arrays
            ):  # draw additional arrays according to `num_arrays`
                y = data.draw(self.gen_other_array(x, i), label="array-{}".format(i))
                arrs.append(y)

            arr_copies = [
                copy(arr) for arr in arrs
            ]  # list of array-copies to check for mutation

            if callable(self.kwargs):
                kwargs = data.draw(self.kwargs(*arrs))
                if not isinstance(kwargs, dict):
                    raise TypeError(
                        "`kwargs` was a search strategy. This needs to draw dictionaries,"
                        "instead drew: {}".format(kwargs)
                    )
            else:
                # set or draw keyword args to be passed to functions
                kwargs = {
                    k: (
                        data.draw(v(*arrs), label="kwarg: {}".format(k))
                        if callable(v)
                        else v
                    )
                    for k, v in self.kwargs.items()
                }

            if self.assumptions is not None:
                assume(self.assumptions(*arrs, **kwargs))

            for i, arr in enumerate(
                arrs
            ):  # assure arrays don't contain forbidden values
                for value in self.index_to_no_go.get(i, ()):
                    assume(np.all(arr != value))

            # execute mygrad and "true" functions. Compare outputs and check mygrad behavior
            o = self.op(*(Tensor(i) for i in arrs), **kwargs, constant=constant)
            tensor_out = o.data
            true_out = self.true_func(*arrs, **kwargs)

            assert isinstance(
                o, Tensor
            ), "`mygrad_func` returned type {}, should return `mygrad.Tensor`".format(
                type(o)
            )
            assert (
                o.constant is constant
            ), "`mygrad_func` returned tensor.constant={}, should be constant={}".format(
                o.constant, constant
            )

            assert_allclose(
                actual=tensor_out,
                desired=true_out,
                err_msg="`mygrad_func(x)` and `true_func(x)` produce different results",
                atol=1e-7,
            )

            for n, (arr, arr_copy) in enumerate(zip(arrs, arr_copies)):
                assert_array_equal(
                    arr,
                    arr_copy,
                    err_msg="arr-{} was mutated during forward prop".format(n),
                )

        return wrapper


class backprop_test_factory:
    """ Decorator

        Randomly draw arrays x, ... to verify that a binary mygrad function,
        `f(x, ..., **kwargs)` performs back-propagation appropriately.

        x.grad, ... are compared against numerical derivatives of f.

        This decorator is extremely uber: the decorated function's body is
        never executed. The function definition is effectively there to name
        the test being constructed. This constructs the entire test

        Notes
        -----
        By default this wrapper dispatches a numerical derivative that utilizes the complex
        step methodology. This requires that the function being tested be analytic and have
        a complex value implementation. See `tests.utils.numerical_gradient` for more details.

        Examples
        --------
        >>> from mygrad import add
        >>> import numpy as np
        >>> @backprop_test_factory(mygrad_func=add, true_func=np.add)
        ... def test_add():
        ...     pass"""

    def __init__(
        self,
        *,
        mygrad_func: Callable[[Tensor], Tensor],
        true_func: Callable[[np.ndarray], np.ndarray],
        num_arrays: int,
        index_to_bnds: Optional[Dict[int, Tuple[int, int]]] = None,
        index_to_no_go: Optional[Dict[int, Sequence[int]]] = None,
        index_to_arr_shapes: Optional[
            Dict[int, Union[Sequence[int], SearchStrategy]]
        ] = None,
        index_to_unique: Optional[Union[Dict[int, bool], bool]] = None,
        elements_strategy: Optional[SearchStrategy] = None,
        kwargs: Optional[
            Union[Callable, Dict[str, Union[Any, Callable[[Any], SearchStrategy]]]]
        ] = None,
        h: float = 1e-20,
        rtol: float = 1e-8,
        atol: float = 1e-8,
        vary_each_element: bool = False,
        finite_difference=False,
        assumptions: Optional[Callable[..., bool]] = None
    ):
        """
        Parameters
        ----------
        mygrad_func : Callable[[numpy.ndarray, ...], mygrad.Tensor]
            The mygrad function whose backward pass validity is being checked.

        true_func : Callable[[numpy.ndarray, ...], numpy.ndarray]
            A known correct version of the function, which is used to compute
            numerical derivatives.

        num_arrays : int
            The number of arrays that must be passed to ``mygrad_func``

        index_to_bnds : Optional[Dict[int, Tuple[int, int]]]
            Indicate the lower and upper bounds from which the elements
            for array-i is drawn. By default, [-100, 100].

        index_to_no_go : Optional[Dict[int, Sequence[int]]]
            Values that array-i cannot possess. By default, no values are
            excluded.

        index_to_arr_shapes : Optional[Dict[int, Union[Sequence[int], SearchStrategy]]]
            The shape for array-i. This can be an exact shape or a hypothesis search
            strategy that draws shapes.
                Default for array-0: `hnp.array_shapes(max_side=3, max_dims=3)`
                Default for array-i: `broadcastable_shape(arr-0.shape)`

        index_to_unique : Optional[Union[Dict[int, bool], bool]]
            Determines whether the elements drawn for each of the input-arrays are
            required to be unique or not. By default this is `False` for each array.
            If a single boolean value is supplied, this is applied for every array.

        elements_strategy : Optional[Union[SearchStrategy]
            The hypothesis-type-strategy used to draw the array elements.
            The default value is ``hypothesis.strategies.floats``.

        kwargs : Optional[Dict[str, Union[Any, Callable[[Any], SearchStrategy]]]]
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

        finite_difference : bool, optional (default=False)
            If True, the finite-difference method will be used to compute the numerical
            derivative instead of the complex step method (default). This is necessary
            if the function being tested is not analytic or does not have a complex-value
            implementation.

        assumptions : Optional[Callable[[arrs, **kwargs], bool]]
            A callable that is fed the generated arrays and keyword arguments that will
            be fed to ``mygrad_func``. If ``assumptions`` returns ``False``, that test
            case will be marked as skipped by hypothesis.
        """

        index_to_bnds = index_to_bnds if index_to_bnds is not None else {}
        index_to_no_go = index_to_no_go if index_to_no_go is not None else {}
        index_to_arr_shapes = (
            index_to_arr_shapes if index_to_arr_shapes is not None else {}
        )
        index_to_unique = index_to_unique if index_to_unique is not None else {}
        self.elements_strategy = (
            elements_strategy if elements_strategy is not None else st.floats
        )
        kwargs = kwargs if kwargs is not None else {}

        assert num_arrays > 0
        self.op = mygrad_func
        self.true_func = true_func

        if isinstance(index_to_bnds, (tuple, list, np.ndarray)):
            index_to_bnds = {k: index_to_bnds for k in range(num_arrays)}
        self.index_to_bnds = index_to_bnds

        if isinstance(index_to_no_go, (tuple, list, np.ndarray)):
            index_to_no_go = {k: index_to_no_go for k in range(num_arrays)}
        self.index_to_no_go = index_to_no_go

        if isinstance(
            index_to_arr_shapes, (tuple, list, np.ndarray, st.SearchStrategy)
        ):
            index_to_arr_shapes = {k: index_to_arr_shapes for k in range(num_arrays)}
            self.index_to_arr_shapes = index_to_arr_shapes
        self.index_to_arr_shapes = index_to_arr_shapes

        if isinstance(index_to_unique, bool):
            index_to_unique = {k: index_to_unique for k in range(num_arrays)}
        self.index_to_unique = index_to_unique
        self.kwargs = kwargs
        self.num_arrays = num_arrays

        assert isinstance(h, Real) and h > 0
        self.h = h

        self.tolerances = dict(rtol=rtol, atol=atol)

        assert isinstance(vary_each_element, bool)
        self.vary_each_element = vary_each_element

        assert assumptions is None or callable(assumptions)
        self.assumptions = assumptions

        assert isinstance(finite_difference, bool)
        self.finite_difference = finite_difference

        if finite_difference and vary_each_element:
            raise NotImplementedError(
                "`finite_difference` does not have an implementation supporting "
                "\n`vary_each_element=True`"
            )

        if finite_difference and h < 1e-8:
            from warnings import warn

            warn(
                "The `finite_difference` method is being used with an h-value of {}."
                "\nThis is likely too small, and was intended for use with the complex-step "
                "\nmethod. Please update `h` in this call to `backprop_test_factory`".format(
                    h
                )
            )

    def gen_first_array(self) -> st.SearchStrategy:
        """
        Hypothesis search strategy for drawing the array x to be passed to f(x, ...)

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy"""
        return hnp.arrays(
            shape=self.index_to_arr_shapes.get(
                0, hnp.array_shapes(max_side=3, min_dims=0, max_dims=3)
            ),
            dtype=float,
            elements=self.elements_strategy(*self.index_to_bnds.get(0, (-100, 100))),
            unique=self.index_to_unique.get(0, False),
        )

    def gen_other_array(self, x: np.ndarray, i: int) -> st.SearchStrategy:
        """
        Hypothesis search strategy for drawing an array y to be passed to f(x, ..., y_i,...).
        By default, y is drawn to have a shape that is broadcast-compatible with x.

        Parameters
        ----------
        x : numpy.ndarray
        i : int
            The argument index-location of y in the signature of f.

        Returns
        -------
        hypothesis.searchstrategy.SearchStrategy"""
        return hnp.arrays(
            shape=self.index_to_arr_shapes.get(i, broadcastable_shape(x.shape)),
            dtype=float,
            elements=self.elements_strategy(*self.index_to_bnds.get(i, (-100, 100))),
            unique=self.index_to_unique.get(i, False),
        )

    def __call__(self, f):
        @given(x=self.gen_first_array(), data=st.data())
        @wraps(f)
        def wrapper(x, data):
            arrs = [x]  # list of drawn arrays to feed to functions
            # draw additional arrays according to `num_arrays`
            for i in range(1, self.num_arrays):
                arrs.append(
                    data.draw(self.gen_other_array(x, i), label="array-{}".format(i))
                )

            arrs = tuple(Tensor(arr) for arr in arrs)
            arr_copies = tuple(copy(arr) for arr in arrs)

            if callable(self.kwargs):
                kwargs = data.draw(self.kwargs(*arrs), label="kwargs")
                if not isinstance(kwargs, dict):
                    raise TypeError(
                        "`kwargs` was a search strategy. This needs to draw dictionaries,"
                        "instead drew: {}".format(kwargs)
                    )
            else:
                # The keyword args to be passed to `self.op`. If any provided argument is callable
                # it is assumed to by a hypothesis search strategy, and all of the drawn arrays will
                # be passed to the strategy, in order to draw a value for that keyword argument.
                # Otherwise the provided value is used as-is.
                kwargs = {
                    k: (
                        data.draw(v(*arrs), label="kwarg: {}".format(k))
                        if callable(v)
                        else v
                    )
                    for k, v in self.kwargs.items()
                }

            if self.assumptions is not None:
                assume(self.assumptions(*arrs, **kwargs))

            for i, arr in enumerate(
                arrs
            ):  # assure arrays don't contain forbidden values
                for value in self.index_to_no_go.get(i, ()):
                    assume(np.all(arr != value))

            # forward pass of the function
            out = self.op(*arrs, **kwargs)

            # gradient to be backpropped through this operation
            grad = data.draw(
                hnp.arrays(
                    shape=out.shape,
                    dtype=float,
                    elements=st.floats(-10, 10),
                    unique=True,
                ),
                label="grad",
            )
            grad_copy = copy(grad)  # keep a copy to check for later mutations

            # compute analytic derivatives via mygrad-backprop
            if any(out.shape != i.shape for i in arrs):
                # Broadcasting occurred
                # Must reduce `out` to scalar
                # first multiply by `grad` to simulate non-trivial back-prop
                (grad * out).sum().backward()
            else:
                out.backward(grad)

            if not self.finite_difference:
                # compute derivatives via numerical approximation of derivative
                # using the complex-step method
                numerical_grad = (
                    numerical_gradient_full
                    if self.vary_each_element
                    else numerical_gradient
                )

            else:
                numerical_grad = finite_difference
            grads_numerical = numerical_grad(
                self.true_func, *(i.data for i in arrs), back_grad=grad, kwargs=kwargs
            )

            # check that the analytic and numeric derivatives match
            for n, (arr, d_num) in enumerate(zip(arrs, grads_numerical)):
                assert_allclose(
                    arr.grad,
                    d_num,
                    **self.tolerances,
                    err_msg="arr-{}: mygrad derivative and numerical derivative do not match".format(
                        n
                    )
                )

                # check that none of the set derivatives is a view of `grad`
                assert not np.shares_memory(
                    arr.grad, grad
                ), "arr-{}.grad stores a view of grad".format(n)

            # check that none of the set derivatives are views of one another
            for arr_i, arr_j in combinations(arrs, 2):
                assert not np.shares_memory(
                    arr_i.grad, arr_j.grad
                ), "two input arrays were propagated views of the same gradient"

            # verify that null_gradients works
            out.null_gradients()
            assert all(i.grad is None for i in arrs), "null_gradients failed"

            # check if any of the input-arrays were mutated
            for n, (arr, arr_copy) in enumerate(zip(arrs, arr_copies)):
                assert_array_equal(
                    arr.data,
                    arr_copy.data,
                    err_msg="arr-{} was mutated during backward prop".format(n),
                )

            # check if `grad` was mutated
            assert_array_equal(
                grad, grad_copy, err_msg="`grad` was mutated during backward prop"
            )

        return wrapper
