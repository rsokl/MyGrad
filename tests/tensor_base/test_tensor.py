from mygrad import Tensor
from mygrad.operation_base import Operation
import mygrad as mg

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from pytest import raises
import numpy as np
from numpy.testing import assert_allclose, assert_equal


def test_to_scalar():
    nd_tensor = Tensor([1, 2])
    with raises(TypeError):
        float(nd_tensor)

    with raises(TypeError):
        int(nd_tensor)

    with raises(ValueError):
        nd_tensor.item()

    for size1_tensor in (Tensor(1), Tensor([[1]])):
        assert float(size1_tensor) == 1.
        assert int(size1_tensor) == 1
        assert size1_tensor.item() == 1.


def test_repr():
    assert repr(Tensor(1)) == 'Tensor(1)'
    assert repr(Tensor([1])) == 'Tensor([1])'
    assert repr(Tensor([1, 2])) == 'Tensor([1, 2])'
    tmp_rep = 'Tensor([[0, 1, 2],\n        [3, 4, 5],\n        [6, 7, 8]])'
    assert repr(mg.arange(9).reshape((3, 3))) == tmp_rep


def test_contains():
    t = Tensor([[0, 1, 2], [3, 4, 5]])
    assert 0 in t and 0 in t.data
    assert [0, 1, 2] in t and [0, 1, 2] in t.data
    assert [0, 3] in t and [0, 3] in t.data
    assert -1 not in t and -1 not in t.data


@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       constant=st.booleans(),
       scalar=st.booleans(),
       creator=st.booleans())
def test_properties(a, constant, scalar, creator):
    array = np.asarray(a)
    if creator:
        ref = Operation()
        tensor = Tensor(a, constant=constant, _creator=ref, _scalar_only=scalar)
    else:
        tensor = Tensor(a, constant=constant, _scalar_only=scalar)

    assert tensor.ndim == array.ndim
    assert tensor.shape == array.shape
    assert tensor.size == array.size
    assert len(tensor) == len(array)
    assert tensor.dtype == array.dtype
    assert assert_equal(actual=tensor.data, desired=a)
    assert (not creator) or tensor.creator is ref


def test_init_data():
    for data in [0, [], (0, 0), ((0, 0), (0, 0)), np.random.rand(3, 4, 2)]:
        assert_equal(actual=Tensor(data).data, desired=np.asarray(data),
                     err_msg="Initialization with non-tensor failed")
        assert_equal(actual=Tensor(Tensor(data)).data, desired=np.asarray(data),
                     err_msg="Initialization with tensor failed")


@given(x=hnp.arrays(dtype=float, shape=hnp.array_shapes(min_dims=1, max_dims=4)))
def test_init_data_rand(x):
    assert_equal(actual=Tensor(x).data, desired=x)


@given(x=hnp.arrays(dtype=float, shape=hnp.array_shapes()))
def test_items(x):
    """ verify that tensor.item() mirrors array.item()"""
    tensor = Tensor(x)
    try:
        value = x.item()
        assert_allclose(value, tensor.item())
    except ValueError:
        with raises(ValueError):
            tensor.item()


op = Operation()
@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       creator=st.sampled_from((None, op)),
       constant=st.booleans(),
       scalar_only=st.booleans(),
       dtype=st.sampled_from((None,)))
def test_init_params(a, creator, constant, scalar_only, dtype):
    kwargs = dict(_creator=creator, constant=constant, _scalar_only=scalar_only)
    if dtype is not None:
        kwargs["dtype"] = dtype

    if dtype is str or dtype == "bad":
        with raises(TypeError):
            Tensor(a, **kwargs)
        return None
    else:
        tensor = Tensor(a, **kwargs)

    if dtype is None:
        dtype = a.dtype

    assert tensor.creator is creator
    assert tensor.constant is constant
    assert tensor.scalar_only is scalar_only
    assert tensor.dtype == a.astype(dtype).dtype
    assert tensor.grad is None


def test_special_methods():
    from mygrad.math.arithmetic.ops import Add, Subtract, Multiply, Divide, Power
    from mygrad.math.arithmetic.ops import Negative

    x = Tensor([2, 8, 5])
    y = Tensor([1, 3, 2])

    for op_name, op in zip(("__add__", "__sub__", "__mul__", "__truediv__", "__pow__"),
                           (Add, Subtract, Multiply, Divide, Power)):
        tensor_out = getattr(Tensor, op_name)(x, y)
        numpy_out = getattr(np.ndarray, op_name)(x.data, y.data)
        assert isinstance(tensor_out, Tensor)
        assert not tensor_out.constant
        assert_equal(tensor_out.data, numpy_out)
        assert isinstance(tensor_out.creator, op)
        assert tensor_out.creator.variables[0] is x
        assert tensor_out.creator.variables[1] is y

    for op_name, op in zip(("__radd__", "__rsub__", "__rmul__", "__rtruediv__", "__rpow__"),
                           (Add, Subtract, Multiply, Divide, Power)):
        tensor_out = getattr(Tensor, op_name)(x, y)
        numpy_out = getattr(np.ndarray, op_name)(x.data, y.data)
        assert isinstance(tensor_out, Tensor)
        assert not tensor_out.constant
        assert_equal(tensor_out.data, numpy_out)
        assert isinstance(tensor_out.creator, op)
        assert tensor_out.creator.variables[0] is y
        assert tensor_out.creator.variables[1] is x

    tensor_out = getattr(Tensor, "__neg__")(x)
    numpy_out = getattr(np.ndarray, "__neg__")(x.data)
    assert isinstance(tensor_out, Tensor)
    assert not tensor_out.constant
    assert_equal(tensor_out.data, numpy_out)
    assert isinstance(tensor_out.creator, Negative)
    assert tensor_out.creator.variables[0] is x


def test_comparison_ops():
    x = Tensor([1, 3, 5])
    y = Tensor([1, 4, 2])
    for op in ("__lt__", "__le__", "__gt__", "__ge__", "__eq__", "__ne__"):
        tensor_out = getattr(Tensor, op)(x, y)
        array_out = getattr(np.ndarray, op)(x.data, y.data)
        assert_equal(actual=tensor_out, desired=array_out)


@given(x=st.floats(min_value=-1E-6, max_value=1E6),
       y=st.floats(min_value=-1E-6, max_value=1E6),
       z=st.floats(min_value=-1E-6, max_value=1E6))
def test_null_gradients(x, y, z):
    x = Tensor(x)
    y = Tensor(y)
    z = Tensor(z)

    f = x*y + z
    g = x + z*f*f

    # check side effects
    unused = 2*g - f
    w = 1*f

    g.backward()
    assert x.grad is not None
    assert y.grad is not None
    assert z.grad is not None
    assert f.grad is not None
    assert g.grad is not None
    assert len(x._ops) > 0
    assert len(y._ops) > 0
    assert len(z._ops) > 0
    assert len(f._ops) > 0
    assert len(g._ops) > 0
    assert w.grad is None

    g.null_gradients()
    assert x.grad is None
    assert y.grad is None
    assert z.grad is None
    assert f.grad is None
    assert g.grad is None
    assert len(x._ops) == 0
    assert len(y._ops) == 0
    assert len(z._ops) == 0
    assert len(f._ops) == 0
    assert len(g._ops) == 0
