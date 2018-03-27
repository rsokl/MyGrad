from mygrad.tensor_base import Tensor
from mygrad.operations.multivar_operations import Operation
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
from pytest import raises
import numpy as np


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
