from ..utils.numerical_gradient import numerical_gradient_full
from ..custom_strategies import broadcastable_shape

from mygrad import Tensor
from hypothesis import given
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp
import numpy as np

from itertools import chain
from mygrad.linalg.einsum import einsum


def compare_einsum(*operands):
    mygrad_out = einsum(*operands)
    assert isinstance(mygrad_out, Tensor)
    operands = tuple(i.data if isinstance(i, Tensor) else i for i in operands)
    assert np.allclose(np.einsum(*operands), einsum(*operands).data)


def compare_backprop(*operands, atol=1e-5, rtol=1e-5):
    """ Compare back-propagation through mygrad-einsum, and compare
        against numerical derivative"""
    if isinstance(operands[0], str):
        # operands form: "ijk, ijk", x, y
        script = operands[0]
        vars = operands[1:]
        vars = tuple(np.asarray(i).astype(float) for i in vars)
        tensors = tuple(Tensor(i) for i in vars)

        def f(*args): return np.einsum(script, *args)
        out = einsum(script, *tensors)
    else:
        # operands form: op0, sublist0, op1, sublist1, ..., [sublistout]
        end = -1 if len(operands) % 2 else None  # -1 if sublistout is included
        vars = tuple(np.asarray(i).astype(float) for i in operands[:end:2])
        tensors = tuple(Tensor(i) for i in vars)

        def f(*args):
            x = tuple(chain.from_iterable(zip(args, operands[1::2])))
            if end is not None:
                x += (operands[-1],)
            return np.einsum(*x)

        x = tuple(chain.from_iterable(zip(tensors, operands[1::2])))
        if end is not None:
            x += (operands[-1],)
        out = einsum(*x)

    grad = np.random.rand(*out.shape)
    #    grad = np.ones(out.shape)
    out.backward(grad)

    numerical_derivs = backprop_linalg(f, *vars, back_grad=grad)

    for dnum, tensor in zip(numerical_derivs, tensors):
        assert dnum.shape == tensor.grad.shape
        assert np.allclose(dnum, tensor.grad, atol=atol, rtol=rtol)


def backprop_linalg(f, *args, back_grad):
    grads = []
    args = tuple(i for i in args)
    for n in range(len(args)):
        tmp_f = lambda var: f(*args[:n], var, *args[n+1:])
        grads.append(numerical_gradient_full(tmp_f, x=args[n], back_grad=back_grad))
    return grads


def test_unique_from_end():
    from mygrad.linalg.einsum import _unique_from_end
    assert _unique_from_end("") == ""
    assert _unique_from_end("a") == "a"
    assert _unique_from_end("aaaa") == "a"
    assert _unique_from_end("aba") == "ba"
    assert _unique_from_end("abccbac") == "bac"


def test_merge_mappings():
    from mygrad.linalg.einsum import _merge_max_mappings
    a = dict(a=0, b=100, c=3)
    b = dict(a=10, b=10)
    c = dict(c=50)
    d = dict(d=70)
    assert _merge_max_mappings(a, b, c, d) == dict(a=10, b=100, c=50, d=70)


def test_einsum_static_fwd():
    """ Check all einsum examples from numpy doc"""
    a = Tensor(np.arange(25).reshape(5, 5))
    b = Tensor(np.arange(5))
    c = Tensor(np.arange(6).reshape(2, 3))

    compare_einsum('ii', a)
    compare_einsum(a, [0, 0])

    compare_einsum('ii->i', a)
    compare_einsum(a, [0, 0], [0])

    compare_einsum('ij,j', a, b)
    compare_einsum(a, [0, 1], b, [1])

    compare_einsum('...j,j', a, b)
    compare_einsum(a, [Ellipsis, 0], b, [Ellipsis, 0])

    compare_einsum('ji', c)
    compare_einsum(c, [1,0])

    compare_einsum('..., ...', 3, c)
    compare_einsum(3, [Ellipsis], c, [Ellipsis])

    compare_einsum('i,i', b, b)
    compare_einsum(b, [0], b, [0])

    compare_einsum('i,j', np.arange(2) + 1, b)
    compare_einsum('i...->...', a)

    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    compare_einsum('ijk,jil->kl', a, b)
    compare_einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3])

    a = np.arange(6).reshape((3, 2))
    b = np.arange(12).reshape((4, 3))
    compare_einsum('ki,jk->ij', a, b)
    compare_einsum(a, [0, 1], b, [2, 0], [1, 2])

    compare_einsum('ki,...k->i...', a, b)
    compare_einsum(a, [0, 1], b, [Ellipsis, 0], [1, Ellipsis])

    compare_einsum('k...,jk', a, b)
    compare_einsum(a, [0, Ellipsis], b, [2, 0])


def test_einsum_static_bkwd():
    """ Check all einsum examples from numpy doc"""
    a = np.arange(25).reshape(5, 5)
    b = np.arange(5)
    c = np.arange(6).reshape(2, 3)

    compare_backprop('ii', a)
    compare_backprop(a, [0, 0])

    compare_backprop('ii->i', a)
    compare_backprop(a, [0, 0], [0])

    compare_backprop('ij->', a)
    compare_backprop(a, [0, 0], [0])

    compare_backprop('ij,j', a, b)
    compare_backprop(a, [0, 1], b, [1])

    compare_backprop('...j,j', a, b)
    compare_backprop(a, [Ellipsis, 0], b, [Ellipsis, 0])
    
    compare_backprop('ji', c)
    compare_backprop(c, [1, 0])
    
    compare_backprop('..., ...', 3, c)
    compare_backprop(3, [Ellipsis], c, [Ellipsis])
    
    compare_backprop('i,i', b, b)
    compare_backprop(b, [0], b, [0])
    
    compare_backprop('i,j', np.arange(2) + 1, b)

    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    compare_backprop('ijk,jil->kl', a, b, atol=1e-3, rtol=1e-3)
    compare_backprop(a, [0, 1, 2], b, [1, 0, 3], [2, 3], atol=1e-3, rtol=1e-3)
    
    a = np.arange(6).reshape((3, 2))
    b = np.arange(12).reshape((4, 3))
    compare_backprop('ki,jk->ij', a, b)
    compare_backprop(a, [0, 1], b, [2, 0], [1, 2])
    
    compare_backprop('ki,...k->i...', a, b)
    compare_backprop(a, [0, 1], b, [Ellipsis, 0], [1, Ellipsis])
    
    compare_backprop('k...,jk', a, b)
    compare_backprop(a, [0, Ellipsis], b, [2, 0])


def test_traces_bkwd():
    a = np.random.rand(5, 2, 2, 5)
    b = np.random.rand(3, 2, 1)
    c = np.random.rand(2, 2)
    compare_backprop('ijji -> i', a)
    compare_backprop(a, [0, 1, 1, 0], [0])

    compare_backprop('ijji -> j', a)
    compare_backprop('ijji -> ij', a)
    compare_backprop('ijji -> ji', a)
    compare_backprop('ijji -> ', a)
    compare_backprop('ijji,kji -> ', a, b)
    compare_backprop('ijji,kji -> kj', a, b)
    compare_backprop('ijji,kji,jj-> kj', a, b, c)
    compare_backprop('ijji,kji,jj-> ijk', a, b, c)
    compare_backprop('ijji,kji,jj-> jk', a, b, c)


@given(num=st.integers(1, 10),
       data=st.data())
def test_einsum_bkwd1(num, data):
    x = Tensor(np.random.rand(num))
    y_shape = data.draw(broadcastable_shape(x.shape, min_dim=1, max_dim=1))
    y = Tensor(np.random.rand(*y_shape))

    grad = data.draw(st.floats(-100, 100))
    o = einsum("i, i", x, y)
    o.backward(grad)

    def f(x, y): return np.einsum("i, i", x, y)

    dx, dy = backprop_linalg(f, x.data, y.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-5, rtol=1e-5)
    assert np.allclose(y.grad, dy, atol=1e-5, rtol=1e-5)

    o.null_gradients()
    assert x.grad is None
    assert y.grad is None

    # test broadcasting in reverse direction
    o = einsum("i, i", y, x)
    o.backward(grad)

    dy, dx = backprop_linalg(f, y.data, x.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-5, rtol=1e-5)
    assert np.allclose(y.grad, dy, atol=1e-5, rtol=1e-5)

    o.null_gradients()


@given(num=st.integers(1, 10),
       data=st.data())
def test_einsum_bkwd2(num, data):
    y = Tensor(np.random.rand(num))

    # flip so that leading dim of x is broadcastable with y
    x_shape = data.draw(broadcastable_shape(y.shape, min_dim=2, max_dim=2))[::-1]
    x = Tensor(np.random.rand(*x_shape))
    grad = np.random.rand(x.shape[-1])

    o = einsum("ia, i -> a", x, y)
    o.backward(grad)

    def f(x, y): return np.einsum("ia, i -> a", x, y)

    dx, dy = backprop_linalg(f, x.data, y.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-6)
    assert np.allclose(y.grad, dy, atol=1e-6)


@given(shape=hnp.array_shapes(min_dims=2, max_dims=2),
       data=st.data())
def test_einsum_bkwd3(shape, data):
    script = "ia, ia, i -> a"
    x = Tensor(np.random.rand(*shape))

    y_shape = data.draw(broadcastable_shape(shape, min_dim=2, max_dim=2))
    y = Tensor(np.random.rand(*y_shape))

    z_shape = data.draw(broadcastable_shape(x.shape[:1], min_dim=1, max_dim=1))
    z = Tensor(np.random.rand(*z_shape ))

    grad = np.random.rand(x.shape[1])

    o = einsum(script, x, y, z)
    o.backward(grad)

    def f(x, y, z): return np.einsum(script, x, y, z)

    dx, dy, dz = backprop_linalg(f, x.data, y.data, z.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-6)
    assert np.allclose(y.grad, dy, atol=1e-6)
    assert np.allclose(z.grad, dz, atol=1e-6)


@given(shape=hnp.array_shapes(min_dims=2, max_dims=2),
       data=st.data())
def test_einsum_bkwd4(shape, data):
    script = "ia, i -> "

    x = Tensor(np.random.rand(*shape))

    y_shape = data.draw(broadcastable_shape(x.shape[:1], min_dim=1, max_dim=1))
    y = Tensor(np.random.rand(*y_shape))

    grad = np.random.rand(1).item()

    o = einsum(script, x, y)
    o.backward(grad)

    def f(x, y): return np.einsum(script, x, y)

    dx, dy = backprop_linalg(f, x.data, y.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-6)
    assert np.allclose(y.grad, dy, atol=1e-6)


def test_einsum_bkwd5():
    x = Tensor(np.random.rand(5, 3, 4, 6))
    y = Tensor(np.random.rand(1, 5, 6, 2))
    grad = np.random.rand(1, 3, 4, 2)

    def f(x, y): return np.einsum("iBCj, aijd -> aBCd", x, y)

    o = einsum("iBCj, aijd -> aBCd", x, y)
    o.backward(grad)

    dx, dy = backprop_linalg(f, x.data, y.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-6)
    assert np.allclose(y.grad, dy, atol=1e-6)


@given(shape=hnp.array_shapes(min_dims=3, max_dims=3),
       data=st.data())
def test_einsum_bkwd6(shape, data):
    sig = "ijk, -> j"
    x = Tensor(np.random.rand(*shape))
    y = Tensor(np.random.rand(1).item())
    grad = np.random.rand(x.shape[1])

    o = einsum(sig, x, y)
    o.backward(grad)

    def f(x, y): return np.einsum(sig, x, y)

    dx, dy = backprop_linalg(f, x.data, y.data, back_grad=grad)

    assert np.allclose(x.grad, dx, atol=1e-6)
    assert np.allclose(y.grad, dy, atol=1e-6)