from mygrad.tensor_base import Tensor
from ..custom_strategies import valid_axes
import numpy as np
import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp


@given(keepdims=st.booleans(),
       d=st.data()
       )
def test_min_fwd(keepdims, d):
    a = d.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                          dtype=float,
                          elements=st.floats(-100, 100)))
    axes = d.draw(valid_axes(a.ndim))
    a = Tensor(a)

    np_out = a.data.min(axis=axes, keepdims=keepdims)
    pygrad_out = a.min(axis=axes, keepdims=keepdims).data
    if pygrad_out.ndim == 0:
        pygrad_out = np.asscalar(pygrad_out)

    assert np.allclose(np_out, pygrad_out)


@given(fill_val=st.floats(min_value=-100, max_value=100),
       shape=hnp.array_shapes(max_side=3, max_dims=5),
       keepdims=st.booleans(),
       d=st.data()
       )
def test_degenerate_min_back(fill_val, shape, keepdims, d):
    """ test min backprop for degenerate-valued tensors"""
    a = Tensor(np.full(shape=shape, fill_value=fill_val, dtype=float))
    axes = d.draw(valid_axes(a.ndim))

    out = a.min(axis=axes, keepdims=keepdims)
    out2 = out * np.arange(1, 1 + out.data.size).reshape(out.shape)

    out2.backward()

    grad = np.zeros_like(a.data)

    if a.ndim == 0:
        assert a.grad == 1.
        return None

    if out.ndim == 0:
        grad[tuple(0 for i in range(a.ndim))] = 1
        assert np.allclose(grad, a.grad)
    else:
        index = [slice(None) for i in range(a.ndim)]
        if axes is None:
            index = [0 for i in range(len(index))]
        else:
            for i in axes:
                index[i] = 0
        index = tuple(index)
        shape = a.data.min(axis=axes).shape
        grad[index] = np.arange(1, 1 + out.data.size).reshape(shape)
        assert np.allclose(grad, a.grad)




@given(keepdims=st.booleans(),
       d=st.data()
       )
def test_min_back(keepdims, d):
    a = d.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                          dtype=float,
                          elements=st.floats(-100, 100)))
    axes = d.draw(valid_axes(a.ndim, pos_only=True))
    a = Tensor(a)

    # single global minimum
    if axes is None or tuple(sorted(axes)) == tuple(range(a.ndim)):
        index = tuple(np.random.choice(i.flat) for i in np.indices(a.shape))
        a[index] = a.min() - 1

        grad = np.zeros_like(a.data)
        grad[index] = 1

        a = Tensor(a)
        out = a.min(axis=axes, keepdims=keepdims)
        out.backward()
        assert np.allclose(grad, a.grad)
        return None

    # explicitly place maxima within tensor
    static_axes = tuple(sorted(set(range(a.ndim)) - set(axes)))
    static_shape = tuple(a.shape[i] for i in static_axes)
    red_shapes = tuple(a.shape[i] for i in axes)
    sorter = np.argsort(static_axes + axes)

    # generate indices to span static axes
    static_indices = tuple(i for i in np.indices(static_shape))

    # generate random index-runs along reduction axes
    choose_indices = tuple(np.random.choice(range(i), size=static_indices[0].shape) for i in red_shapes)

    # create index tuple the selects random runs along reduction axes
    static_indices += choose_indices
    indices = []
    for i in sorter:
        indices.append(static_indices[i])
    indices = tuple(indices)

    # place extrema
    a[indices] = a.min() - np.random.rand(*indices[0].shape)
    a = Tensor(a)
    out = a.min(axis=axes)

    # break degeneracy amongst grad values
    tmp = np.arange(1, out.data.size+1).reshape(out.shape)
    out2 = out * tmp
    out2.backward()

    grad = np.zeros_like(a.data)
    grad[indices] = np.arange(1, out.data.size+1).reshape(out.shape)

    assert np.allclose(grad, a.grad)
