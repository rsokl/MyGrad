from collections import defaultdict
from functools import partial, reduce
from itertools import zip_longest
from typing import Mapping, Optional, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.typing import Shape
from mygrad.ufuncs import MyGradBinaryUfunc, MyGradUnaryUfunc
from tests.custom_strategies import array_likes, no_value, tensors
from tests.utils.functools import MinimalArgs, populate_args
from tests.utils.numerical_gradient import numerical_gradient
from tests.wrappers.uber import backprop_test_factory


def _broadcast_two_shapes(shape_a: Shape, shape_b: Shape) -> Shape:
    result = []
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if a != b and (a != 1) and (b != 1):
            raise ValueError(
                f"shapes {shape_a!r} and {shape_b!r} are not broadcast-compatible"
            )
        result.append(a if a != 1 else b)
    return tuple(reversed(result))


def _broadcast_shapes(*shapes):
    """Returns the shape resulting from broadcasting the
    input shapes together.
    Raises ValueError if the shapes are not broadcast-compatible"""
    assert len(shapes)
    return reduce(_broadcast_two_shapes, shapes, ())


@st.composite
def populates_ufunc(
    draw: st.DataObject.draw,
    ufunc: Union[MyGradUnaryUfunc, MyGradBinaryUfunc],
    arg_index_to_elements: Optional[Mapping[int, st.SearchStrategy]] = None,
    tensor_only=False,
    min_side=0,
    min_dims=0,
) -> st.SearchStrategy[MinimalArgs]:
    ind_to_elements = defaultdict(lambda: st.floats(-1e-9, 1e9))

    if arg_index_to_elements is not None:
        for k, v in arg_index_to_elements.items():
            ind_to_elements[k] = v

    shapes: hnp.BroadcastableShapes = draw(
        hnp.mutually_broadcastable_shapes(
            num_shapes=ufunc.nin + 1,
            min_side=min_side,
            min_dims=min_dims,
            max_dims=3,
            max_side=3,
        ),
        label="shapes",
    )

    array_strat = (
        array_likes if tensor_only is False else partial(tensors, constant=False)
    )
    args = populate_args(
        *(
            draw(array_strat(shape=shape, dtype=float, elements=ind_to_elements[n]))
            for n, shape in enumerate(shapes.input_shapes[:-1])
        ),
        where=draw(
            no_value()
            | st.booleans().map(lambda x: not x)
            | hnp.arrays(
                shape=shapes.input_shapes[-1], dtype=bool, elements=st.booleans()
            )
        ),
    )
    fill_value = draw(st.integers(0, 1))

    where = args.kwargs.get("where", True)
    if where is not True:
        # the predicted results shape can be wrong if the inputs don't `where`
        # broadcast against `where`
        out_shape = (
            shapes.result_shape
            if not isinstance(where, bool)
            else _broadcast_shapes(*(mg.asarray(a).shape for a in args.args))
        )
        args["out"] = np.full(out_shape, fill_value=fill_value, dtype=float)
    return args


not_zero = st.floats(-1e9, 1e9).filter(lambda x: not np.isclose(x, 0, atol=1e-5))

log_largest = np.log(np.finfo(np.float64).max)
log2_largest = np.log2(np.finfo(np.float64).max)
valid_log_inputs = st.floats(np.finfo(np.float64).eps, 1e20)
valid_exp_inputs = st.floats(min_value=-1e100, max_value=log_largest)
valid_exp2_inputs = st.floats(min_value=-1e100, max_value=log2_largest)

ufuncs_and_domains = [
    (mg.add, None),
    (mg.divide, {1: not_zero}),
    (mg.exp, {0: valid_exp_inputs}),
    (mg.exp2, {0: st.floats(min_value=-1e9, max_value=log2_largest)}),
    (mg.expm1, {0: valid_exp_inputs}),
    (mg.log, {0: valid_log_inputs}),
    (mg.log2, {0: valid_log_inputs}),
    (mg.log10, {0: valid_log_inputs}),
    (mg.log1p, {0: st.floats(-1 + np.finfo(np.float64).eps, 1e20)}),
    (
        mg.logaddexp,
        {
            0: st.floats(min_value=-log_largest / 2, max_value=log_largest / 2),
            1: st.floats(min_value=-log_largest / 2, max_value=log_largest / 2),
        },
    ),
    (
        mg.logaddexp2,
        {
            0: st.floats(min_value=-log2_largest / 2, max_value=log_largest / 2),
            1: st.floats(min_value=-log2_largest / 2, max_value=log_largest / 2),
        },
    ),
    (mg.multiply, None),
    (mg.negative, None),
    (mg.positive, None),
    (
        mg.power,
        {0: st.floats(0.001, 1e9), 1: st.sampled_from([1, 2]) | st.floats(-3, 3)},
    ),
    (mg.reciprocal, {0: not_zero}),
    (mg.square, None),
    (mg.subtract, None),
    (mg.true_divide, {1: not_zero}),
    (mg.cos, None),
    (mg.sin, None),
    (mg.tan, {0: st.floats(-np.pi / 2 + 1e-5, np.pi / 2 - 1e-5)}),
    (mg.arccos, {0: st.floats(-0.99, 0.99)}),
    (mg.arcsin, {0: st.floats(-0.99, 0.99)}),
    (mg.arctan, None),
]


@pytest.mark.parametrize("ufunc, domains", ufuncs_and_domains)
@given(data=st.data())
def test_ufunc_fwd(
    data: st.DataObject,
    ufunc: Union[MyGradUnaryUfunc, MyGradBinaryUfunc],
    domains: Optional[Mapping[int, st.SearchStrategy]],
):
    """
    Checks:
    - mygrad implementation of ufunc against numpy
    - op doesn't mutate inputs
    - tensor base matches array base
    - constant propagates as expected
    - grad shape consistency
    """
    args = data.draw(
        populates_ufunc(ufunc, arg_index_to_elements=domains), label="ufunc call sig"
    )
    args.make_arrays_read_only()  # guards against mutation
    # Explicitly retrieve numpy's ufunc of the same name.
    # Don't trust that mygrad ufunc is binds the correct numpy ufunc under the hood
    numpy_ufunc = getattr(np, ufunc.__name__)

    numpy_out = numpy_ufunc(*args.args_as_no_mygrad(), **args.kwargs)
    mygrad_out = ufunc(*args.args, **args.kwargs)

    # Check that numpy and mygrad implementations agree
    assert_allclose(actual=mygrad_out, desired=numpy_out)
    assert mygrad_out.base is None and numpy_out.base is None

    # Check that constant propagated properly
    expected_constant = all(t.constant is True for t in args.tensors_only())
    assert mygrad_out.constant is expected_constant
    mygrad_out.backward()

    # Check grad shape consistency
    for n, t in enumerate(args.tensors_only()):
        assert t.constant or t.grad.shape == t.shape, f"arg: {n}"


@pytest.mark.parametrize(
    "ufunc, domains",
    [
        (mg.add, None),
        (mg.divide, {1: not_zero}),
        (mg.exp, {0: st.floats(min_value=-1e9, max_value=log_largest / 10)}),
        (mg.exp2, {0: st.floats(min_value=-1e9, max_value=log2_largest / 100)}),
        (mg.expm1, {0: st.floats(min_value=-1e9, max_value=log_largest / 10)}),
        (mg.log, {0: valid_log_inputs}),
        (mg.log2, {0: valid_log_inputs}),
        (mg.log10, {0: valid_log_inputs}),
        (mg.log1p, {0: st.floats(-1 + np.finfo(np.float64).eps, 1e20)}),
        (mg.multiply, None),
        (mg.negative, None),
        (mg.positive, None),
        (
            mg.power,
            {0: st.floats(0.001, 1e9), 1: st.sampled_from([1, 2]) | st.floats(-3, 3)},
        ),
        (mg.reciprocal, {0: not_zero}),
        (mg.square, None),
        (mg.subtract, None),
        (mg.true_divide, {1: not_zero}),
        (mg.cos, None),
        (mg.sin, None),
        (mg.tan, {0: st.floats(-np.pi / 2 + 1e-5, np.pi / 2 - 1e-5)}),
        (mg.arccos, {0: st.floats(-0.99, 0.99)}),
        (mg.arcsin, {0: st.floats(-0.99, 0.99)}),
        (mg.arctan, None),
    ],
)
@given(data=st.data())
def test_ufunc_bkwd(
    data: st.DataObject,
    ufunc: Union[MyGradUnaryUfunc, MyGradBinaryUfunc],
    domains: Optional[Mapping[int, st.SearchStrategy]],
):
    """
    Checks:
    - backprop matches numerical gradient
    - backprop doesn't mutate grad
    """
    args = data.draw(
        populates_ufunc(
            ufunc, arg_index_to_elements=domains, tensor_only=True, min_side=1
        ),
        label="ufunc call sig",
    )
    args.make_arrays_read_only()  # guards against mutation

    mygrad_out = ufunc(*args.args, **args.kwargs)

    # draw upstream gradient to be backpropped
    grad = data.draw(
        hnp.arrays(dtype=float, shape=mygrad_out.shape, elements=st.floats(-1e1, 1e1)),
        label="grad",
    )

    grad.flags["WRITEABLE"] = False  # will raise if backprop mutates grad

    mygrad_out.backward(grad)
    kwargs = args.kwargs.copy()

    # numerical grad needs to write complex-valued outputs
    kwargs["out"] = np.zeros_like(mygrad_out, dtype=complex)
    numpy_ufunc = partial(getattr(np, ufunc.__name__), **kwargs)

    grads = numerical_gradient(numpy_ufunc, *args.args_as_no_mygrad(), back_grad=grad)

    # check that gradients match numerical derivatives
    for n in range(ufunc.nin):
        desired = np.nan_to_num(grads[n])
        actual = args.args[n].grad
        assert_allclose(
            desired=desired,
            actual=actual,
            err_msg=f"the grad of tensor-{n} did not match the "
            f"numerically-computed gradient",
        )


# logaddexp can't be evaluated with complex values
@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=mg.logaddexp,
    true_func=np.logaddexp,
    num_arrays=2,
    index_to_bnds=(-10, 10),
    use_finite_difference=True,
    h=1e-8,
    atol=1e-4,
    rtol=1e-4,
)
def test_logaddexp_bkwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=mg.logaddexp2,
    true_func=np.logaddexp2,
    num_arrays=2,
    atol=1e-4,
    rtol=1e-4,
    use_finite_difference=True,
    h=1e-8,
    index_to_bnds=(-100, 100),
)
def test_logaddexp2_bkwd():
    pass
