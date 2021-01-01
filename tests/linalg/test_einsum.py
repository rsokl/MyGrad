from copy import copy
from itertools import chain

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import assume, given, settings
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import Tensor
from mygrad.linalg.funcs import einsum

from ..custom_strategies import broadcastable_shapes, tensors
from ..utils.numerical_gradient import numerical_gradient_full


def bool_strat():
    """ einsum's optimize=True option has bugs prior to version 1.14.5
        (caught by these very unit tests!), thus we only test `optimize=True`
        for more recent versions."""
    return st.booleans() if np.__version__ >= "1.14.5" else st.just(False)


def compare_einsum(*operands, optimize=False):
    mygrad_out = einsum(*operands)
    assert isinstance(mygrad_out, Tensor)
    operands = tuple(i.data if isinstance(i, Tensor) else i for i in operands)
    assert_allclose(np.einsum(*operands), einsum(*operands, optimize=optimize).data)


def compare_backprop(*operands, atol=1e-5, rtol=1e-5, optimize=False):
    """ Compare back-propagation through mygrad-einsum, and compare
        against numerical derivative"""
    if isinstance(operands[0], str):
        # operands form: "ijk, ijk", x, y
        script = operands[0]
        vars = operands[1:]
        vars = tuple(np.asarray(i).astype(float) for i in vars)
        tensors = tuple(Tensor(i) for i in vars)

        def f(*args):
            return np.einsum(script, *args)

        out = einsum(script, *tensors, optimize=optimize)
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
        out = einsum(*x, optimize=optimize)

    grad = np.random.rand(*out.shape)
    #    grad = np.ones(out.shape)
    out.backward(grad)

    numerical_derivs = numerical_gradient_full(f, *vars, back_grad=grad)

    for n, (dnum, tensor) in enumerate(zip(numerical_derivs, tensors)):
        assert dnum.shape == tensor.grad.shape
        assert_allclose(
            dnum,
            tensor.grad,
            atol=atol,
            rtol=rtol,
            err_msg="The numerical and mygrad derivatives disagree for "
            "variable index {}".format(n),
        )


@pytest.mark.parametrize(
    ("full_string", "end"),
    [("", ""), ("a", "a"), ("aaaa", "a"), ("aba", "ba"), ("abccbac", "bac")],
)
def test_unique_from_end(full_string, end):
    from mygrad.linalg.ops import _unique_from_end

    assert _unique_from_end(full_string) == end


def test_merge_mappings():
    from mygrad.linalg.ops import _merge_max_mappings

    a = dict(a=0, b=100, c=3)
    b = dict(a=10, b=10)
    c = dict(c=50)
    d = dict(d=70)
    e = dict()
    assert _merge_max_mappings(a, b, c, d, e) == dict(a=10, b=100, c=50, d=70)


@given(optimize=bool_strat())
def test_einsum_static_fwd(optimize):
    """ Check all einsum examples from numpy doc"""
    a = mg.arange(25).reshape(5, 5)
    b = mg.arange(5)
    c = mg.arange(6).reshape(2, 3)

    compare_einsum("ii", a, optimize=optimize)
    compare_einsum(a, [0, 0], optimize=optimize)

    compare_einsum("ii->i", a, optimize=optimize)
    compare_einsum(a, [0, 0], [0], optimize=optimize)

    compare_einsum("ij,j", a, b, optimize=optimize)
    compare_einsum(a, [0, 1], b, [1], optimize=optimize)

    compare_einsum("...j,j", a, b, optimize=optimize)
    compare_einsum(a, [Ellipsis, 0], b, [Ellipsis, 0], optimize=optimize)

    compare_einsum("ji", c, optimize=optimize)
    compare_einsum(c, [1, 0], optimize=optimize)

    compare_einsum("..., ...", 3, c, optimize=optimize)
    compare_einsum(3, [Ellipsis], c, [Ellipsis], optimize=optimize)

    compare_einsum("i,i", b, b, optimize=optimize)
    compare_einsum(b, [0], b, [0], optimize=optimize)

    compare_einsum("i,j", np.arange(2) + 1, b, optimize=optimize)
    compare_einsum("i...->...", a, optimize=optimize)

    a = np.arange(60.0).reshape(3, 4, 5)
    b = np.arange(24.0).reshape(4, 3, 2)
    compare_einsum("ijk,jil->kl", a, b, optimize=optimize)
    compare_einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3], optimize=optimize)

    a = np.arange(6).reshape((3, 2))
    b = np.arange(12).reshape((4, 3))
    compare_einsum("ki,jk->ij", a, b, optimize=optimize)
    compare_einsum(a, [0, 1], b, [2, 0], [1, 2], optimize=optimize)

    compare_einsum("ki,...k->i...", a, b, optimize=optimize)
    compare_einsum(a, [0, 1], b, [Ellipsis, 0], [1, Ellipsis], optimize=optimize)

    compare_einsum("k...,jk", a, b, optimize=optimize)
    compare_einsum(a, [0, Ellipsis], b, [2, 0], optimize=optimize)


@given(optimize=bool_strat())
def test_einsum_static_bkwd(optimize):
    """ Check all einsum examples from numpy doc"""
    a = np.arange(25).reshape(5, 5)
    b = np.arange(5)
    c = np.arange(6).reshape(2, 3)

    compare_backprop("ii", a, optimize=optimize)
    compare_backprop(a, [0, 0], optimize=optimize)

    compare_backprop("ii->i", a, optimize=optimize)
    compare_backprop(a, [0, 0], [0], optimize=optimize)

    compare_backprop("ij->", a, optimize=optimize)
    compare_backprop(a, [0, 1], optimize=optimize)

    compare_backprop("ij,j", a, b, optimize=optimize)
    compare_backprop(a, [0, 1], b, [1], optimize=optimize)

    compare_backprop("...j,j", a, b, optimize=optimize)
    compare_backprop(a, [Ellipsis, 0], b, [Ellipsis, 0], optimize=optimize)

    compare_backprop("ji", c, optimize=optimize)
    compare_backprop(c, [1, 0], optimize=optimize)

    compare_backprop("..., ...", 3, c, optimize=optimize)
    compare_backprop(3, [Ellipsis], c, [Ellipsis], optimize=optimize)

    compare_backprop("i,i", b, b, optimize=optimize)
    compare_backprop(b, [0], b, [0], optimize=optimize)

    compare_backprop("i,j", np.arange(2) + 1, b, optimize=optimize)

    a = np.arange(60.0).reshape(3, 4, 5)
    b = np.arange(24.0).reshape(4, 3, 2)
    compare_backprop("ijk,jil->kl", a, b, atol=1e-3, rtol=1e-3, optimize=optimize)
    compare_backprop(
        a, [0, 1, 2], b, [1, 0, 3], [2, 3], atol=1e-3, rtol=1e-3, optimize=optimize
    )

    a = np.arange(6).reshape((3, 2))
    b = np.arange(4).reshape((4, 1))
    compare_backprop("ki,jk->ij", a, b, optimize=optimize)
    compare_backprop(a, [0, 1], b, [2, 0], [1, 2], optimize=optimize)

    compare_backprop("ki,...k->i...", a, b, optimize=optimize)
    compare_backprop(a, [0, 1], b, [Ellipsis, 0], [1, Ellipsis], optimize=optimize)

    compare_backprop("k...,jk", a, b, optimize=optimize)
    compare_backprop(a, [0, Ellipsis], b, [2, 0], optimize=optimize)


@settings(deadline=1000)
@given(optimize=bool_strat())
def test_traces_bkwd(optimize):
    a = np.random.rand(5, 2, 2, 5)
    b = np.random.rand(3, 2, 1)
    c = np.random.rand(1, 1)
    d = np.random.rand(5, 5, 5)
    compare_backprop("ijji -> i", a, optimize=optimize)
    compare_backprop(a, [0, 1, 1, 0], [0], optimize=optimize)

    compare_backprop("iii -> i", d, optimize=optimize)
    compare_backprop("ijji -> j", a, optimize=optimize)
    compare_backprop("ijji -> ij", a, optimize=optimize)
    compare_backprop("ijji -> ji", a, optimize=optimize)
    compare_backprop("ijji -> ", a, optimize=optimize)
    compare_backprop("ijji,kji -> ", a, b, optimize=optimize)
    compare_backprop("ijji,kji -> kj", a, b, optimize=optimize)
    compare_backprop("ijji,kji,jj-> kj", a, b, c, optimize=optimize)
    compare_backprop("ijji,kji,jj-> ijk", a, b, c, optimize=optimize)
    compare_backprop("ijji,kji,jj-> jk", a, b, c, optimize=optimize)


def test_redundant_args():
    """
    Test behavior for when einsum receives redundant inputs. An optimization
    was added such that einsum will only compute the gradient for such an entry
    once and scale it accordingly.
    """
    a = mg.arange(4).reshape(2, 2)
    a_copy = copy(a)

    # check standard summation
    o = einsum("ij,ij", a, a)
    assert len(o.creator.cache) == 1
    o.sum().backward()

    o = einsum("ij,ij", a_copy, a_copy * 1)
    assert len(o.creator.cache) == 2
    o.sum().backward()
    assert_allclose(a.grad, a_copy.grad)

    a = Tensor(np.arange(4).reshape(2, 2))
    a_copy = copy(a)

    # check standard summation using alt signature
    o = einsum(a, [0, 1], a, [0, 1])
    assert len(o.creator.cache) == 1
    o.sum().backward()

    o = einsum(a_copy, [0, 1], a_copy * 1, [0, 1])
    assert len(o.creator.cache) == 2
    o.sum().backward()
    assert_allclose(a.grad, a_copy.grad)

    a = Tensor(np.arange(4).reshape(2, 2))
    a_copy = copy(a)

    # check matmul (no redundant indices)
    o = einsum("ij,jk", a, a)
    assert len(o.creator.cache) == 2
    o.sum().backward()

    o = a_copy @ a_copy
    o.sum().backward()
    assert_allclose(a.grad, a_copy.grad)

    a = Tensor(np.arange(4).reshape(2, 2))
    a_copy = copy(a)

    # check traces
    o = einsum("ii,ii", a, a)
    assert len(o.creator.cache) == 1
    o.sum().backward()

    o = einsum("ii,ii", a_copy, a_copy * 1)
    assert len(o.creator.cache) == 2
    o.sum().backward()
    assert_allclose(a.grad, a_copy.grad)

    a = Tensor(np.arange(4).reshape(2, 2))
    a_copy = copy(a)

    b = Tensor(-1 * np.arange(2).reshape(2, 1))
    b_copy = copy(b)

    # check broadcasting and multiply-redundant input tensors
    # with distinct einsum labels
    o = einsum("ii,ii,i...,i...,...i,...i", a, a, b, b, a, a)
    assert len(o.creator.cache) == 3
    o.sum().backward()

    o = einsum(
        "ii,ii,i...,i...,...i,...i",
        a_copy,
        a_copy * 1,
        b_copy,
        b_copy * 1,
        a_copy,
        1 * a_copy,
    )
    assert len(o.creator.cache) == 6
    o.sum().backward()
    assert_allclose(a.grad, a_copy.grad)
    assert_allclose(b.grad, b_copy.grad)


@given(num=st.integers(1, 10), optimize=bool_strat(), data=st.data())
def test_einsum_bkwd1(num, optimize, data):
    x = mg.random.rand(num)
    y_shape = data.draw(broadcastable_shapes(x.shape, min_dims=1, max_dims=1))
    y = Tensor(np.random.rand(*y_shape))

    grad = data.draw(st.floats(-100, 100))
    o = einsum("i, i", x, y, optimize=optimize)
    o.backward(grad)

    def f(x, y):
        return np.einsum("i, i", x, y)

    dx, dy = numerical_gradient_full(f, x.data, y.data, back_grad=grad)

    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)
    assert_allclose(y.grad, dy, atol=1e-5, rtol=1e-5)

    assert not x._ops
    assert not y._ops

    # test broadcasting in reverse direction
    o = einsum("i, i", y, x, optimize=optimize)
    o.backward(grad)

    assert x.grad is not None
    assert y.grad is not None

    dy, dx = numerical_gradient_full(f, y.data, x.data, back_grad=grad)

    assert_allclose(x.grad, dx, atol=1e-5, rtol=1e-5)
    assert_allclose(y.grad, dy, atol=1e-5, rtol=1e-5)


@given(num=st.integers(1, 10), optimize=bool_strat(), data=st.data())
def test_einsum_bkwd2(num, optimize, data):
    y = Tensor(np.random.rand(num))

    # flip so that leading dim of x is broadcastable with y
    x_shape = data.draw(broadcastable_shapes(y.shape, min_dims=2, max_dims=2))[::-1]
    x = mg.random.rand(*x_shape)
    grad = np.random.rand(x.shape[-1])

    o = einsum("ia, i -> a", x, y, optimize=optimize)
    o.backward(grad)

    def f(x, y):
        return np.einsum("ia, i -> a", x, y)

    dx, dy = numerical_gradient_full(f, x.data, y.data, back_grad=grad)

    assert_allclose(x.grad, dx, atol=1e-6)
    assert_allclose(y.grad, dy, atol=1e-6)


@given(
    shape=hnp.array_shapes(min_dims=2, max_dims=2),
    optimize=bool_strat(),
    data=st.data(),
)
def test_einsum_bkwd3(shape, optimize, data):
    script = "ia, ia, i -> a"
    x = mg.random.rand(*shape)

    y_shape = data.draw(
        broadcastable_shapes(shape, min_dims=2, max_dims=2), label="y_shape"
    )
    y = mg.random.rand(*y_shape)

    z_shape = data.draw(
        broadcastable_shapes(x.shape[:1], min_dims=1, max_dims=1), label="z_shape"
    )
    z = mg.random.rand(*z_shape)

    try:
        o = einsum(script, x, y, z, optimize=optimize)
    except ValueError:
        assume(False)  # skip over invalid einsum shapes
        return

    grad = np.random.rand(*o.shape)
    o.backward(grad)

    def f(x, y, z):
        return np.einsum(script, x, y, z)

    dx, dy, dz = numerical_gradient_full(f, x.data, y.data, z.data, back_grad=grad)

    assert_allclose(x.grad, dx, atol=1e-6)
    assert_allclose(y.grad, dy, atol=1e-6)
    assert_allclose(z.grad, dz, atol=1e-6)


@given(
    shape=hnp.array_shapes(min_dims=2, max_dims=2),
    optimize=bool_strat(),
    data=st.data(),
)
def test_einsum_bkwd4(shape, optimize, data):
    script = "ia, i -> "

    x = mg.random.rand(*shape)

    y_shape = data.draw(broadcastable_shapes(x.shape[:1], min_dims=1, max_dims=1))
    y = mg.random.rand(*y_shape)

    grad = np.random.rand(1).item()

    o = einsum(script, x, y, optimize=optimize)
    o.backward(grad)

    def f(x, y):
        return np.einsum(script, x, y)

    dx, dy = numerical_gradient_full(f, x.data, y.data, back_grad=grad)

    assert_allclose(x.grad, dx, atol=1e-6)
    assert_allclose(y.grad, dy, atol=1e-6)


@settings(deadline=2000)
@given(optimize=bool_strat())
def test_einsum_bkwd5(optimize):
    x = mg.random.rand(5, 3, 4, 6)
    y = mg.random.rand(1, 5, 6, 2)
    grad = np.random.rand(1, 3, 4, 2)

    def f(x, y):
        return np.einsum("iBCj, aijd -> aBCd", x, y)

    o = einsum("iBCj, aijd -> aBCd", x, y, optimize=optimize)
    o.backward(grad)

    dx, dy = numerical_gradient_full(f, x.data, y.data, back_grad=grad)

    assert_allclose(x.grad, dx, atol=1e-6)
    assert_allclose(y.grad, dy, atol=1e-6)


@settings(deadline=2000)
@given(shape=hnp.array_shapes(min_dims=3, max_dims=3), optimize=bool_strat())
def test_einsum_bkwd6(shape, optimize):
    sig = "ijk, -> j"
    x = mg.random.rand(*shape)
    y = mg.random.rand(1)[0]
    grad = np.random.rand(x.shape[1])

    o = einsum(sig, x, y, optimize=optimize)
    o.backward(grad)

    def f(x, y):
        return np.einsum(sig, x, y)

    dx, dy = numerical_gradient_full(f, x.data, y.data, back_grad=grad)

    assert_allclose(x.grad, dx, atol=1e-6)
    assert_allclose(y.grad, dy, atol=1e-6)


@given(
    ndim=st.integers(1, 4),
    side_length=st.integers(1, 4),
    optimize=st.booleans(),
    data=st.data(),
)
def test_einsum_can_produce_diag_view_for_nd_tensor(
    ndim: int, side_length: int, optimize: bool, data: st.DataObject
):
    x = data.draw(tensors(shape=(side_length,) * ndim), label="x")

    diag = mg.einsum(f"{ndim*'i'} -> i", x, optimize=optimize)
    assert diag.base is x


@given(x=tensors().filter(lambda x: x.ndim > 0), optimize=st.booleans())
def test_einsum_can_produce_full_view(x: Tensor, optimize: bool):
    view = mg.einsum(f"... -> ...", x, optimize=optimize)
    assert view.base is x
    assert_array_equal(x, view)


@pytest.mark.parametrize("downstream_of_view", [True, False])
@given(
    orig_x=tensors(shape=(3, 3, 3), elements=st.floats(-100, 100), constant=False),
    optimize=st.booleans(),
    scalar=tensors(shape=tuple(), elements=st.floats(-100, 100), constant=False),
)
def test_backprop_through_inplace_op(
    orig_x: Tensor, scalar: Tensor, optimize: bool, downstream_of_view: bool
):
    x = +orig_x

    if downstream_of_view:
        x = x[...]
    diag = mg.einsum("iii -> i", x, optimize=optimize)
    diag[...] = scalar

    if not downstream_of_view:
        assert diag.base is x

    x.sum().backward()
    assert scalar.grad == 3.0
    assert_allclose(x.grad, np.ones_like(x.data))

    orig_x_grad = np.ones_like(orig_x)
    np.einsum("iii->i", orig_x_grad)[...] = 0.0
    assert_allclose(actual=orig_x.grad, desired=orig_x_grad)
