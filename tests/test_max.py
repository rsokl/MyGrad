from mygrad.tensor_base import Tensor

import numpy as np
import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp


@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                    dtype=float,
                    elements=st.floats(-100, 100)),
       num_axes=st.integers(min_value=0, max_value=5),
       keepdims=st.booleans()
       )
def test_max_fwd(a, num_axes, keepdims):
    a = Tensor(a)
    if num_axes == 0:
        axes = None
    else:
        axes = tuple(np.random.choice(range(0, a.ndim), size=min(num_axes, a.ndim), replace=False))

    np_out = a.data.max(axis=axes, keepdims=keepdims)
    pygrad_out = a.max(axis=axes, keepdims=keepdims).data
    if pygrad_out.ndim == 0:
        pygrad_out = np.asscalar(pygrad_out)

    assert np.allclose(np_out, pygrad_out)

    if num_axes:
        neg_axes = tuple(np.random.choice(range(-a.ndim, 0), size=min(num_axes, a.ndim), replace=False))
        np_out = a.data.max(axis=neg_axes, keepdims=keepdims)
        pygrad_out = a.max(axis=neg_axes, keepdims=keepdims).data

        if pygrad_out.ndim == 0:
            pygrad_out = np.asscalar(pygrad_out)

        assert np.allclose(np_out, pygrad_out)


@given(fill_val=st.floats(min_value=-100, max_value=100),
       shape=hnp.array_shapes(max_side=3, max_dims=5),
       num_axes=st.integers(min_value=0, max_value=5),
       keepdims=st.booleans()
       )
def test_degenerate_max_back(fill_val, shape, num_axes, keepdims):
    """ test max backprop for degenerate-valued tensors"""
    a = Tensor(np.full(shape=shape, fill_value=fill_val, dtype=float))

    if num_axes == 0:
        axes = None
    else:
        axes = tuple(np.random.choice(range(0, a.ndim), size=min(num_axes, a.ndim), replace=False))

    out = a.max(axis=axes, keepdims=keepdims)
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
        shape = a.data.max(axis=axes).shape
        grad[index] = np.arange(1, 1 + out.data.size).reshape(shape)
        assert np.allclose(grad, a.grad)


@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                    dtype=float,
                    elements=st.floats(-100, -100)),
       num_axes=st.integers(min_value=0, max_value=5),
       keepdims=st.booleans()
       )
def test_max_back(a, num_axes, keepdims):
    """ Test Tensor.max for arbitrary data, axis, and keepdim"""
    if num_axes == 0:
        axes = None
    else:
        axes = np.random.choice(range(0, a.ndim), size=min(num_axes, a.ndim), replace=False)
        axes = tuple(sorted(axes))

    # single global maximum
    if axes is None or axes == tuple(range(a.ndim)):
        index = tuple(np.random.choice(i.flat) for i in np.indices(a.shape))
        a[index] = a.max() + 1

        grad = np.zeros_like(a)
        grad[index] = 1

        a = Tensor(a)
        out = a.max(axis=axes, keepdims=keepdims)
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
    a[indices] = a.max() + np.random.rand(*indices[0].shape)
    a = Tensor(a)
    out = a.max(axis=axes)

    # break degeneracy amongst grad values
    tmp = np.arange(1, out.data.size+1).reshape(out.shape)
    out2 = out * tmp
    out2.backward()

    grad = np.zeros_like(a.data)
    grad[indices] = np.arange(1, out.data.size+1).reshape(out.shape)

    assert np.allclose(grad, a.grad)
