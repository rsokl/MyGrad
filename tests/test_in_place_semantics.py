from typing import Callable

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose, assert_array_equal

import mygrad as mg
from mygrad import Tensor
from tests.custom_strategies import tensors


# Make sure we actually test the correctness of the
# in-place syntaxes, e.g. `x += y`, and not just
# `x.__iadd__(y)`
#
# Also, make sure that augmented updates on tensors
# match behavior of numpy
def test_iadd_mirrors_numpy():
    an = np.array([3.0, 4.0])
    at = mg.Tensor(an)

    bn = an
    vn = an[...]
    an += 2.0

    bt = at
    vt = at[...]
    at += 2.0

    assert_array_equal(an, at)
    assert_array_equal(bn, bt)
    assert_array_equal(vn, vt)


def test_isub_mirrors_numpy():
    an = np.array([3.0, 4.0])
    at = mg.Tensor(an)

    bn = an
    vn = an[...]
    an -= 2.0

    bt = at
    vt = at[...]
    at -= 2.0

    assert_array_equal(an, at)
    assert_array_equal(bn, bt)
    assert_array_equal(vn, vt)


def test_imul_mirrors_numpy():
    an = np.array([3.0, 4.0])
    at = mg.Tensor(an)

    bn = an
    vn = an[...]
    an *= 2.0

    bt = at
    vt = at[...]
    at *= 2.0

    assert_array_equal(an, at)
    assert_array_equal(bn, bt)
    assert_array_equal(vn, vt)


def test_idiv_mirrors_numpy():
    an = np.array([3.0, 4.0])
    at = mg.Tensor(an)

    bn = an
    vn = an[...]
    an /= 2.0

    bt = at
    vt = at[...]
    at /= 2.0

    assert_array_equal(an, at)
    assert_array_equal(bn, bt)
    assert_array_equal(vn, vt)


def test_ipow_mirrors_numpy():
    an = np.array([3.0, 4.0])
    at = mg.Tensor(an)

    bn = an
    vn = an[...]
    an **= 2.1

    bt = at
    vt = at[...]
    at **= 2.1

    assert_array_equal(an, at)
    assert_array_equal(bn, bt)
    assert_array_equal(vn, vt)


def test_isqr_mirrors_numpy():
    an = np.array([3.0, 4.0])
    at = mg.Tensor(an)

    bn = an
    vn = an[...]
    an **= 2

    bt = at
    vt = at[...]
    at **= 2

    assert_array_equal(an, at)
    assert_array_equal(bn, bt)
    assert_array_equal(vn, vt)


def test_ipow_1_mirrors_numpy():
    an = np.array([3.0, 4.0])
    at = mg.Tensor(an)

    bn = an
    vn = an[...]
    an **= 1

    bt = at
    vt = at[...]
    at **= 1

    assert_array_equal(an, at)
    assert_array_equal(bn, bt)
    assert_array_equal(vn, vt)


@pytest.mark.parametrize("inplace_on_view", [True, False])
def test_raising_during_in_place_op_doesnt_corrupt_graph(inplace_on_view: bool):
    x = mg.arange(1.0, 5.0)
    y_base = 2 * x

    y = y_base[...] if inplace_on_view else y_base

    w = y[...]
    with pytest.raises(ValueError):
        y[:2] = y  # shape mismatch

    (2 * w).backward()
    assert (y.base is y_base) if inplace_on_view else (y.base is None)
    assert w.base is y_base
    assert np.shares_memory(w, y)
    assert_allclose(w.grad, 2 * np.ones_like(y))
    assert_allclose(y_base.grad, 2 * np.ones_like(y_base))
    assert_allclose(y.grad, 2 * np.ones_like(y))
    assert_allclose(x.grad, 4 * np.ones_like(y))


@pytest.mark.parametrize("inplace_on_view", [False, True])
@pytest.mark.parametrize("x_constant", [False, True])
@pytest.mark.parametrize("y_constant", [False, True])
def test_inplace_update_propagates_constant_info(
    inplace_on_view: bool, x_constant: bool, y_constant: bool
):
    x = mg.arange(1.0, 5.0, constant=x_constant)
    y = mg.zeros_like(x, constant=y_constant)

    if inplace_on_view:
        x = x[...]

    dangling_view = x[:2]
    assert x.constant is x_constant
    assert dangling_view.constant is x_constant

    x[...] = y

    assert x.constant is (x_constant and y_constant)
    assert dangling_view.constant is x.constant


@pytest.mark.parametrize("inplace_on_view", [True, False])
@pytest.mark.parametrize("constant", [True, False])
def test_in_place_op_propagates_to_views(constant: bool, inplace_on_view: bool):
    x = mg.arange(1.0, 5.0, constant=constant)
    y_base = +x
    y = y_base[...] if inplace_on_view else y_base

    view1 = y[...]
    view2 = view1[...]  # view of view
    y[:2] = -1  # should mutate all views
    assert y_base.base is None
    if inplace_on_view:
        assert y.base is y_base
    assert view1.base is y_base
    assert view2.base is y_base
    assert_array_equal(x, mg.arange(1.0, 5.0))

    assert_array_equal(y_base, [-1.0, -1.0, 3.0, 4.0])
    assert_array_equal(y_base, y)
    assert_array_equal(y_base, view1)
    assert_array_equal(y_base, view2)


@given(tensors(shape=(4,), constant=False))
@pytest.mark.parametrize("inplace_on_view", [True, False])
def test_simple_backprop_from_view_post_upstream_mutation(
    inplace_on_view: bool, x: Tensor
):
    y_base = +x
    y = y_base[...] if inplace_on_view else y_base
    z = y[...]
    y[:2] = 0  # base is mutated
    # downstream view should carry appropriate info
    # for backprop post-mutation
    w = mg.ones_like(z)
    (w * z).backward()

    assert_array_equal(y, y_base)
    assert_array_equal(z, y_base)
    assert_array_equal(w.grad, [0.0, 0.0, *y_base.data[-2:]])
    assert_array_equal(z.grad, np.ones_like(y_base))
    assert_array_equal(y_base.grad, np.ones_like(y_base))
    assert_array_equal(y.grad, np.ones_like(y_base))
    assert_array_equal(x.grad, [0.0, 0.0, 1.0, 1.0])


@given(tensors(shape=(4,), elements=st.floats(-10, 10), constant=False))
@pytest.mark.parametrize("inplace_on_view", [True, False])
def test_mutation_doesnt_corrupt_upstream_op(inplace_on_view: bool, x: Tensor):
    y_base = +x
    y = y_base[...] if inplace_on_view else y_base
    view = y[...]

    # z = x**4
    z = mg.multiply_sequence(x, y, view, view[...])

    y[:2] = 0  # shouldn't change backprop through z

    z.backward()  # dz/dx = 6 * x ** 2

    assert_allclose(z, x.data ** 4)
    assert_array_equal(view, y)
    assert_allclose(z.grad, np.ones_like(y))
    assert_allclose(x.grad, 4 * np.asarray(x) ** 3)

    assert y_base.grad is None
    assert y.grad is None
    assert view.grad is None


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("inplace_on_view", [True, False])
def test_sets_and_restores_writeability(inplace_on_view: bool, constant: bool):
    x = mg.arange(1.0, 5.0, constant=constant)
    y_base = +x
    y = y_base[...] if inplace_on_view else y_base
    y[...] = 0
    assert x.data.flags.writeable is False
    assert y_base.data.flags.writeable is False
    assert y.data.flags.writeable is False
    y.backward()
    assert x.data.flags.writeable is True
    assert y_base.data.flags.writeable is True
    assert y.data.flags.writeable is True


@pytest.mark.parametrize("inplace_on_view", [True, False])
@given(x=tensors(read_only=True))
def test_respects_original_writeability(x: Tensor, inplace_on_view: bool):
    assert x.data.flags.writeable is False
    if inplace_on_view:
        x = x[...]

    with pytest.raises(ValueError):
        x[...] = 0


@pytest.mark.parametrize("constant", [True, False])
@pytest.mark.parametrize("inplace_on_view", [True, False])
def test_respects_disabled_memguard(constant: bool, inplace_on_view: bool):
    x = mg.arange(1.0, 5.0, constant=constant)
    y_base = +x
    y = y_base[...] if inplace_on_view else y_base

    with mg.mem_guard_off:
        y[...] = 0
    assert x.data.flags.writeable is False
    assert y_base.data.flags.writeable is True
    assert y.data.flags.writeable is True
    y.backward()
    assert x.data.flags.writeable is True
    assert y_base.data.flags.writeable is True
    assert y.data.flags.writeable is True


@pytest.mark.parametrize("inplace_on_view", [True, False])
@pytest.mark.parametrize(
    "target_op",
    [
        lambda x: x,  # backprop directly post-setitem var
        lambda x: +x,  # backprop from downstream node
        lambda x: x[...],  # backprop from downstream view
    ],
)
@pytest.mark.parametrize(
    "source_op",
    [
        lambda x: x,  # direct view
        lambda x: +x,  # downstream of view
        lambda x: x[...],  # view of view
    ],
)
@given(num_in_place_updates=st.integers(1, 3))
def test_writing_a_view_with_a_view(
    target_op: Callable[[Tensor], Tensor],
    source_op: Callable[[Tensor], Tensor],
    inplace_on_view: bool,
    num_in_place_updates: int,
):
    x = mg.arange(1.0, 5.0)
    y_base = +x
    y = y_base[...] if inplace_on_view else y_base
    dangling_view = y[...]

    for _ in range(num_in_place_updates):
        # after the first in-place update, any additional
        # should have no further effect
        y[:2] = source_op(y[-2:])  # y = [3, 4, 3, 4]

    proxy_y = target_op(y)

    assert_array_equal(y, [3.0, 4.0, 3.0, 4.0])
    assert_array_equal(proxy_y, [3.0, 4.0, 3.0, 4.0])
    assert_array_equal(y, dangling_view)

    # output: -1 x2 + 2 x3 + -3 x2 + 4 x3 -> -4 x2 + 6 x3
    ([-1, 2, -3, 4] * proxy_y).sum().backward()

    assert_array_equal(proxy_y.grad, [-1.0, 2.0, -3.0, 4.0])
    assert_array_equal(y_base.grad, [-1.0, 2.0, -3.0, 4.0])
    assert_array_equal(y.grad, [-1.0, 2.0, -3.0, 4.0])
    assert_array_equal(dangling_view.grad, y_base.grad)
    assert_array_equal(x.grad, [0.0, 0.0, -4.0, 6.0])

    assert dangling_view.base is y_base

    dangling_view.clear_graph()  # release memory

    assert x.data.flags.writeable
    assert y_base.data.flags.writeable
    assert y.data.flags.writeable
    assert dangling_view.data.flags.writeable


@pytest.mark.parametrize("include_upstream_view", [True, False])
@pytest.mark.parametrize("include_downstream_view", [True, False])
def test_set_item_with_broadcasting(
    include_upstream_view: bool, include_downstream_view: bool
):
    xo = mg.arange(27.0).reshape(3, 3, 3).copy()
    x = +xo

    if include_upstream_view:
        x = x[...]

    y = x[np.newaxis]

    if include_downstream_view:
        x = x[...]

    x[:2] = xo[:1]

    # fmt: off
    x_expected = \
        Tensor([[[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]],

                [[0., 1., 2.],
                 [3., 4., 5.],
                 [6., 7., 8.]],

                [[18., 19., 20.],
                 [21., 22., 23.],
                 [24., 25., 26.]]])
    # fmt: on
    assert_array_equal(x, x_expected)
    assert_array_equal(y, x.data[np.newaxis])

    # equivalent to x ** 2
    (y * x).sum().backward()

    xo_grad = np.zeros_like(xo)
    xo_grad[0] += 4 * xo.data[0]
    xo_grad[2] += 2 * xo.data[2]
    assert_array_equal(xo.grad, xo_grad)

    x_grad = 2 * x.data
    assert_array_equal(x.grad, x_grad)
    assert_array_equal(y.grad, x.grad[np.newaxis])


@pytest.mark.parametrize("include_extraneous_ops", [True, False])
def test_complicated_inplace_pattern(include_extraneous_ops: bool):
    static_x = mg.arange(9.0).reshape(3, 3).copy()
    x = +static_x
    y = mg.arange(4.0)
    static_row = +x[...][0]  # [0. 1. 2.]
    view_row = x[...][0]  # view of row-0

    # This is just a complicated way of transposing x in-place
    mg.add(0, x, out=x.T[...])  # set diag of x to be y

    if include_extraneous_ops:
        # These line shouldn't have any effect
        x[::-1] = x[::-1] + 0 * static_x + (x - x).sum() + (0 * y).sum()
        _ = x.T.sum() + y.sum() + static_x[...].sum()

    # static_x:  (we'll call it xs)
    # Tensor([[0., 1., 2.],
    #         [3., 4., 5.],
    #         [6., 7., 8.]])
    #
    # x:  (transpose of static_x)
    # Tensor([[0., 3., 6.],
    #         [1., 4., 7.],
    #         [2., 5., 8.]])
    #
    # view_row:
    # Tensor([[0., 3., 6.])
    #
    # y:
    # Tensor([0., 1., 2., 3.])
    #
    #
    # ℒ =   (static_x00 + ... + static_x22)
    #     + -3 * (xs00 + xs01 + xs02)
    #     +  4 * ( x00 +  x01 +  x02) <- (xij -> xsji)
    #     +  y0 * x11 + y1 * x12 + y2 * x21 + y3 * x22
    #
    # dℒ/dy = [x11, x12, x21, x22]
    #       = [xs11, xs21, xs12, xs22]
    #
    # dℒ/dx = [[4. 4. 4.]
    #          [0. y0 y1]
    #          [0. y2 y3]
    #
    # dℒ/dxs = [[2. -2. -2.]
    #           [5.  y0  y2]
    #           [5.  y1  y3]
    out = (
        static_x.sum()
        + -3 * static_row.sum()
        + 4 * view_row[...].sum()
        + (x[1:, 1:].ravel() * y).sum()
    )  # sum of squared diag

    if include_extraneous_ops:
        out += 0 * (x[...] + static_x[...]).sum()

    out.backward()

    assert_array_equal(
        static_x, Tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    )
    xs = static_x.data
    assert_array_equal(x, xs.T)
    assert_array_equal(static_row, xs[0])
    assert_array_equal(view_row, xs[:, 0])
    assert_array_equal(y, mg.arange(4.0))

    y_grad = static_x.data[np.array([1, 2, 1, 2]), np.array([1, 1, 2, 2])]
    assert_array_equal(y.grad, y_grad)

    x_grad = np.zeros_like(x)
    x_grad[0] += 4.0
    x_grad[1:, 1:] += y.data.reshape(2, 2)
    assert_array_equal(x.grad, x_grad)

    xs_grad = np.ones_like(x)
    xs_grad[0] -= 3.0
    xs_grad[:, 0] += 4.0
    xs_grad[1:, 1:] += y.data.reshape(2, 2).T
    assert_array_equal(static_x.grad, xs_grad)

    if include_extraneous_ops:
        del _


@given(
    x=tensors(shape=(3, 3), elements=st.floats(-1e6, 1e6), constant=False),
    y=tensors(
        shape=(3,),
        elements=st.floats(-1e6, 1e6),
        constant=False,
        read_only=st.booleans(),
    ),
)
def test_complicated_inplace_pattern2(x: Tensor, y: Tensor):
    # this should ultimately leave view_of_x identical to x
    vert_flipped_x = x[::-1]
    vert_horiz_flipped_x = vert_flipped_x[:, ::-1]
    horiz_flipped_x = vert_horiz_flipped_x[::-1]
    view_of_x = horiz_flipped_x[:, ::-1].T.T

    diag_x = mg.einsum("ii->i", view_of_x)  # view of diag of x
    mg.add(y, 0, out=diag_x[...])  # set diag of x to be y

    # check that update propagated to views of `x`
    assert_allclose(vert_flipped_x[::-1], x)
    assert_allclose(vert_horiz_flipped_x[::-1][:, ::-1], x)
    assert_allclose(horiz_flipped_x[:, ::-1], x)
    assert_allclose(view_of_x, x)

    assert vert_flipped_x.base is x
    assert vert_horiz_flipped_x.base is x
    assert horiz_flipped_x.base is x
    assert view_of_x.base is x
    assert x.base is None

    assert not x.data.flags.writeable
    assert not diag_x.data.flags.writeable

    # y0**2 + x01*y1 + x02*y2
    (diag_x * x[0]).sum().backward()  # diag times top-row

    assert x.data.flags.writeable
    assert diag_x.data.flags.writeable

    grad_x = np.zeros_like(x)

    # dl/dx =
    # [2y0, y1, y2]
    # [0., x01, 0 ]
    # [0., 0., x02]
    np.einsum("ii->i", grad_x)[:] = x.data[0]
    grad_x[0, 0] = 2 * y.data[0]
    grad_x[0, 1:] = y.data[1:]

    assert_allclose(grad_x, x.grad)

    if not y.constant:
        # dl/dy = 2y0, x01, x02
        grad_y = x.data[0]
        grad_y[0] = grad_y[0] * 2
        assert_allclose(grad_y, y.grad)
    else:
        assert y.grad is None


def test_unview_backprop_through_multiple_view_funcs():
    # caught bug in unview where multiple distinct sequences
    # of views weren't being exercised through var-1
    x = mg.arange(9.0).reshape(3, 3).copy()
    xx = +x
    y = xx[:, 2]

    # [x12 x02]
    y2 = y[:-1][::-1]

    y2 *= (2, 3)
    # [[2x00 2x01 7x02]
    #  [2x10 2x11 5x12]
    #  [2x20 2x21 2x22]]
    coeff = np.array([[2.0, 2.0, 7.0], [2.0, 2.0, 5.0], [2.0, 2.0, 2.0]])
    out = x.sum() + xx.sum() + y2.sum()
    assert_allclose(out, (x * coeff).sum())
    out.backward()

    assert_array_equal(x.grad, coeff)


def test_op_wrapper_supports_inplace_target():
    from mygrad.math.arithmetic.ops import Multiply

    x_orig = mg.arange(3.0)
    x = +x_orig
    Tensor._op(Multiply, x, x, out=x)
    assert_allclose(x, mg.arange(3) ** 2)
    x.backward()
    assert_allclose(x_orig.grad, 2 * np.arange(3.0))
