from mygrad.tensor_base import Tensor
from mygrad.operation_base import Operation

from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from pytest import raises
import numpy as np
from  numpy.testing import assert_allclose


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
    assert np.all(tensor.data == a)
    assert (not creator) or tensor.creator is ref


def test_init_data():
    for data in [0, [], (0, 0), ((0, 0), (0, 0)), np.random.rand(3, 4, 2)]:
        assert np.all(Tensor(data).data == np.asarray(data)), "Initialization with non-tensor failed"
        assert np.all(Tensor(Tensor(data)).data == np.asarray(data)), "Initialization with tensor failed"


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
