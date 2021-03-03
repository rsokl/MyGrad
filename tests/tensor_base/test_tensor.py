from typing import Optional

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, settings
from numpy.testing import assert_array_equal, assert_equal
from pytest import raises

import mygrad as mg
from mygrad import Tensor
from mygrad.errors import InvalidBackprop
from mygrad.linalg.ops import MatMul
from mygrad.math.arithmetic.ops import Add, Divide, Multiply, Negative, Power, Subtract
from mygrad.operation_base import Operation
from tests.custom_strategies import everything_except, tensors, valid_constant_arg
from tests.utils.errors import does_not_raise


def test_simple_default_constant_behavior():
    assert Tensor(1).constant is True
    assert Tensor(1.0).constant is False


@pytest.mark.parametrize(
    "data",
    [
        None,
        np.array(None, dtype="O"),
        np.array([[0], [0, 0]], dtype="O"),
        np.array(1, dtype="O"),
        0j,
        np.array(1, dtype=complex),
    ],
)
@given(constant=st.booleans(), creator=st.none() | st.just(MatMul()))
def test_input_type_checking(data, constant, creator):
    with raises(TypeError):
        Tensor(data, constant=constant, _creator=creator)


@given(constant=everything_except((bool, type(None))))
def test_input_constant_checking(constant):
    with raises(TypeError):
        Tensor(1.0, constant=constant)


@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0, min_side=0), dtype=hnp.floating_dtypes()
    ),
    dtype=hnp.floating_dtypes(),
    copy=st.booleans(),
    ndmin=st.integers(0, 10),
)
def test_ndmin(x: np.ndarray, copy: bool, dtype, ndmin: int):
    """Ensure Tensor(..., ndmin=<val>) mirrors numpy behavior and
    produces appropriate view behavior"""
    arr = np.array(x, copy=copy, ndmin=ndmin, dtype=dtype)
    tensor = Tensor(x, copy=copy, ndmin=ndmin, dtype=dtype)
    assert_equal(tensor, arr)

    # Specifying array(... , ndmin=val) can create an internal
    # base that the array points to. We want to be sure that
    # this behavior doesnt manifest in tensor, which should
    # bot behave as if it is a view

    # check base behavior
    assert tensor.data.flags.writeable
    assert tensor.base is None
    assert tensor[...].base is tensor

    # check mem-lock behavior
    z = +tensor
    assert not tensor.data.flags.writeable
    del z
    assert tensor.data.flags.writeable


@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(min_dims=0, min_side=0), dtype=hnp.floating_dtypes()
    ),
    constant=st.booleans(),
    data=st.data(),
)
def test_basic_backward(x: np.ndarray, constant: bool, data: st.DataObject):
    """Ensure Tensor.backward() sets the expected gradient for general array-shape/dtype"""
    grad = data.draw(hnp.arrays(shape=x.shape, dtype=x.dtype) | st.none(), label="grad")
    tensor = Tensor(x, constant=constant)
    tensor.backward(grad)
    if tensor.constant:
        assert tensor.grad is None
    else:
        assert_array_equal(
            tensor.grad, np.ones_like(tensor.data) if grad is None else grad
        )

    if tensor.grad is not None:
        assert tensor.dtype == tensor.grad.dtype


@given(
    data=hnp.arrays(shape=hnp.array_shapes(), dtype=hnp.floating_dtypes()),
    constant=st.booleans(),
    set_constant=st.booleans() | st.none(),
    invoke_backward=st.booleans(),
)
def test_copy(data, constant, set_constant, invoke_backward):
    x = Tensor(data, constant=constant)
    y = +x
    if invoke_backward:
        y.backward()
    y_copy = y.copy(constant=set_constant)

    if not invoke_backward:
        assert y.creator is not None
    assert y.dtype == y_copy.dtype
    assert y_copy.constant is (constant if set_constant is None else set_constant)
    if y.grad is None:
        assert y_copy.grad is None
    else:
        assert_array_equal(y.grad, y_copy.grad)
    assert_array_equal(y.data, y_copy.data)


@pytest.mark.parametrize("constant", [True, False])
def test_cant_set_constant(constant):
    tensor = Tensor([1.0], constant=constant)
    with pytest.raises(AttributeError):
        tensor.constant = constant


def test_to_scalar():
    nd_tensor = Tensor([1, 2])
    with raises(TypeError):
        float(nd_tensor)

    with raises(TypeError):
        int(nd_tensor)

    with raises(ValueError):
        nd_tensor.item()

    for size1_tensor in (Tensor(1), Tensor([[1]])):
        assert float(size1_tensor) == 1.0
        assert int(size1_tensor) == 1
        assert size1_tensor.item() == 1.0


@pytest.mark.parametrize(
    ("tensor", "repr_"),
    [
        (Tensor(1), "Tensor(1)"),
        (Tensor([1]), "Tensor([1])"),
        (Tensor([1, 2]), "Tensor([1, 2])"),
        (
            mg.arange(9).reshape((3, 3)),
            "Tensor([[0, 1, 2],\n        [3, 4, 5],\n        [6, 7, 8]])",
        ),
    ],
)
def test_repr(tensor, repr_):
    assert repr(tensor) == repr_


@pytest.mark.parametrize("bad_grads", ["bad", 1.0j])
@given(constant=st.booleans())
def test_invalid_gradient_raises(constant: bool, bad_grads):
    x = Tensor(3.0, constant=constant) * 2
    with (pytest.raises((TypeError, ValueError)) if not constant else does_not_raise()):
        x.backward(bad_grads)


@pytest.mark.parametrize("element", (0, [0, 1, 2]))
def test_contains(element):
    t = Tensor([[0, 1, 2], [3, 4, 5]])
    assert (element in t) is (element in t.data)


@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=3, max_dims=5),
        dtype=float,
        elements=st.floats(-100, 100),
    ),
    constant=st.booleans(),
    creator=st.booleans(),
)
def test_properties(a, constant, creator):
    array = np.asarray(a)
    if creator:
        ref = Add()
        tensor = Tensor(a, constant=constant, _creator=ref)
    else:
        ref = None
        tensor = Tensor(a, constant=constant)

    assert tensor.ndim == array.ndim
    assert tensor.shape == array.shape
    assert tensor.size == array.size
    assert len(tensor) == len(array)
    assert tensor.dtype == array.dtype
    assert_equal(actual=tensor.data, desired=a)
    assert (not creator) or tensor.creator is ref


def test_init_data():
    for data in [0, [], (0, 0), ((0, 0), (0, 0)), np.random.rand(3, 4, 2)]:
        assert_equal(
            actual=Tensor(data).data,
            desired=np.asarray(data),
            err_msg="Initialization with non-tensor failed",
        )
        assert_equal(
            actual=Tensor(Tensor(data)).data,
            desired=np.asarray(data),
            err_msg="Initialization with tensor failed",
        )


@given(
    arr=hnp.arrays(
        dtype=hnp.floating_dtypes(), shape=hnp.array_shapes(min_side=1, min_dims=1)
    ),
    as_tensor=st.booleans(),
    constant=st.booleans(),
)
def test_tensor_copy_on_init_mirrors_array(
    arr: np.ndarray, as_tensor: bool, constant: bool
):
    x = Tensor(arr, copy=False) if as_tensor else arr
    tensor = Tensor(x, constant=constant)
    array = np.array(arr)
    assert np.shares_memory(tensor, arr) is np.shares_memory(array, arr)


@given(
    x=hnp.arrays(
        dtype=hnp.floating_dtypes(), shape=hnp.array_shapes(min_side=0, min_dims=0)
    )
)
def test_init_data_rand(x: np.ndarray):
    assert_equal(actual=Tensor(x).data, desired=x)


@given(
    x=hnp.arrays(
        dtype=float,
        shape=hnp.array_shapes(min_dims=0, min_side=0),
        elements=st.floats(allow_infinity=False, allow_nan=False),
    )
    | st.floats(allow_infinity=False, allow_nan=False)
    | st.integers(-100, 100)
)
def test_items(x):
    """ verify that tensor.item() mirrors array.item()"""
    tensor = Tensor(x)
    try:
        value = np.asarray(x).item()
        assert_array_equal(value, tensor.item())
    except ValueError:
        with raises(ValueError):
            tensor.item()


op = Add()
dtype_strat = st.sampled_from(
    (
        None,
        int,
        float,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.float16,
        np.float32,
        np.float64,
    )
)
dtype_strat_numpy = st.sampled_from(
    (np.int8, np.int16, np.int32, np.int64, np.float16, np.float32, np.float64)
)


@given(
    data=st.data(),
    creator=st.sampled_from((None, op)),
    dtype=dtype_strat,
    numpy_dtype=dtype_strat_numpy,
    ndmin=st.integers(0, 10),
    copy=st.none() | st.booleans(),
)
def test_init_params(
    data,
    creator,
    dtype,
    numpy_dtype,
    ndmin: int,
    copy: Optional[bool],
):
    """Check for bad combinations of init parameters leading to unexpected behavior"""
    elements = (
        (lambda x, y: st.floats(x, y, width=8 * np.dtype(numpy_dtype).itemsize))
        if np.issubdtype(numpy_dtype, np.floating)
        else st.integers
    )
    array_strat_args = dict(
        shape=hnp.array_shapes(max_side=3, max_dims=5),
        dtype=numpy_dtype,
        elements=elements(-100, 100),
    )
    a = data.draw(
        hnp.arrays(**array_strat_args) | tensors(**array_strat_args),
        label="a",
    )

    arr = np.array(a, dtype=dtype, ndmin=ndmin)

    constant = data.draw(valid_constant_arg(arr.dtype), label="constant")

    tensor = Tensor(
        a,
        _creator=creator,
        constant=constant,
        dtype=dtype,
        ndmin=ndmin,
        copy=copy,
    )

    if constant is None:
        constant = issubclass(tensor.dtype.type, np.integer)

    assert tensor.creator is creator
    assert tensor.constant is constant
    assert tensor.dtype is arr.dtype
    assert_equal(tensor.data, arr)
    assert tensor.grad is None
    assert tensor.base is None
    assert tensor.ndim >= ndmin


@pytest.mark.parametrize(
    ("op_name", "op"),
    [
        ("add", Add),
        ("sub", Subtract),
        ("mul", Multiply),
        ("truediv", Divide),
        ("pow", Power),
        ("matmul", MatMul),
    ],
)
@pytest.mark.parametrize("right_op", [True, False])
@given(constant_x=st.booleans(), constant_y=st.booleans())
def test_special_methods(
    op_name: str, op: Operation, constant_x: bool, constant_y: bool, right_op: bool
):
    if right_op:
        op_name = "r" + op_name
    op_name = "__" + op_name + "__"
    x = Tensor([2.0, 8.0, 5.0], constant=constant_x)
    y = Tensor([1.0, 3.0, 2.0], constant=constant_y)

    constant = constant_x and constant_y
    assert hasattr(Tensor, op_name)
    tensor_out = getattr(Tensor, op_name)(x, y)
    numpy_out = getattr(np.ndarray, op_name)(x.data, y.data)
    assert isinstance(tensor_out, Tensor)
    assert tensor_out.constant is constant
    assert_equal(tensor_out.data, numpy_out)
    assert isinstance(tensor_out.creator, op)

    if not right_op:
        assert tensor_out.creator.variables[0] is x
        assert tensor_out.creator.variables[1] is y
    else:
        assert tensor_out.creator.variables[0] is y
        assert tensor_out.creator.variables[1] is x


@given(
    x=hnp.arrays(shape=hnp.array_shapes(), dtype=hnp.floating_dtypes()),
    constant=st.booleans(),
)
def test_pos(x: np.ndarray, constant: bool):
    assume(np.all(np.isfinite(x)))
    x = Tensor(x, constant=constant)
    y = +x
    assert y.creator.variables[0] is x
    assert_array_equal(y.data, x.data)
    assert y.constant is x.constant


@given(x=hnp.arrays(shape=hnp.array_shapes(), dtype=hnp.floating_dtypes()))
def test_neg(x):
    assume(np.all(np.isfinite(x)))
    x = Tensor(x)
    op_name = "__neg__"
    assert hasattr(Tensor, op_name)
    tensor_out = getattr(Tensor, "__neg__")(x)
    numpy_out = getattr(np.ndarray, "__neg__")(x.data)
    assert isinstance(tensor_out, Tensor)
    assert_equal(tensor_out.data, numpy_out)
    assert isinstance(tensor_out.creator, Negative)
    assert tensor_out.creator.variables[0] is x


@pytest.mark.parametrize(
    "op", ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__")
)
@given(
    x=hnp.arrays(
        shape=hnp.array_shapes(),
        dtype=hnp.floating_dtypes(),
        elements=st.floats(-10, 10, width=16),
    ),
    x_constant=st.booleans(),
    y_constant=st.booleans(),
    data=st.data(),
)
def test_comparison_ops(
    op: str, x: np.ndarray, x_constant: bool, y_constant: bool, data: st.SearchStrategy
):
    y = data.draw(
        hnp.arrays(shape=x.shape, dtype=x.dtype, elements=st.floats(-10, 10, width=16))
    )
    x = Tensor(x, constant=x_constant)
    y = Tensor(y, constant=y_constant)
    assert hasattr(Tensor, op), "`Tensor` is missing the attribute {}".format(op)
    tensor_out = getattr(Tensor, op)(x, y)
    array_out = getattr(np.ndarray, op)(x.data, y.data)
    assert_equal(actual=tensor_out, desired=array_out)


@pytest.mark.parametrize(
    "attr",
    (
        "sum",
        "prod",
        "cumprod",
        "cumsum",
        "mean",
        "std",
        "var",
        "max",
        "min",
        "transpose",
        "squeeze",
        "ravel",
    ),
)
@given(constant=st.booleans())
def test_math_methods(attr: str, constant: bool):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], constant=constant)

    assert hasattr(x, attr)
    method_out = getattr(x, attr).__call__()
    function_out = getattr(mg, attr).__call__(x)
    assert_equal(method_out.data, function_out.data)
    assert method_out.constant is constant
    assert type(method_out.creator) is type(function_out.creator)


@pytest.mark.parametrize(
    "attr",
    (
        "argmax",
        "argmin",
        "any",
    ),
)
def test_numpy_math_methods(attr: str):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert hasattr(x, attr)
    method_out = getattr(x, attr).__call__()
    function_out = getattr(np, attr).__call__(x.data)
    assert_equal(method_out, function_out)
    assert_equal(method_out, function_out)


@pytest.mark.parametrize("op", ("moveaxis", "swapaxes"))
@given(constant=st.booleans())
def test_axis_interchange_methods(op: str, constant: bool):
    x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], constant=constant)
    method_out = getattr(x, op)(0, -1)
    function_out = getattr(mg, op)(x, 0, -1)
    assert_equal(method_out.data, function_out.data)
    assert method_out.constant is constant
    assert type(method_out.creator) is type(function_out.creator)


@given(x=tensors(include_grad=st.booleans()), clear_graph=st.booleans())
def test_null_gradients(x: Tensor, clear_graph: bool):
    with pytest.warns(FutureWarning):
        x.null_gradients(clear_graph=clear_graph)


@given(x=tensors(include_grad=st.booleans()))
def test_null_grad(x: Tensor):
    y = x.null_grad()
    assert x.grad is None
    assert y is x


@settings(deadline=None)
@given(
    x=st.floats(min_value=-1e-6, max_value=1e6),
    y=st.floats(min_value=-1e-6, max_value=1e6),
    z=st.floats(min_value=-1e-6, max_value=1e6),
)
def test_clear_graph(x, y, z):
    x = Tensor(x)
    y = Tensor(y)
    z = Tensor(z)

    f = x * y + z
    g = x + z * f * f

    # check side effects
    unused = 2 * g - f
    w = 1 * f
    assert unused is not None

    w.clear_graph()

    assert len(g._ops) > 0
    assert g.creator is not None
    assert len(x._ops) == 0
    assert len(y._ops) == 0
    assert len(z._ops) == 0
    assert len(f._ops) == 0
    assert x.creator is None
    assert y.creator is None
    assert z.creator is None
    assert f.creator is None

    with raises(InvalidBackprop):
        g.backward()


# Tensor has its `__eq__` but not its `__hash__` overridden which leads to subtle
# problems if it ends up being used in a hashable context. See
# https://hynek.me/articles/hashes-and-equality/
# for more details. This checks to make sure that anyone who does so will get an
# error. See also https://github.com/rsokl/MyGrad/pull/276
def test_no_hash():
    try:
        {Tensor(3): "this should not work"}
    except TypeError as e:
        assert str(e) == "unhashable type: 'Tensor'"
