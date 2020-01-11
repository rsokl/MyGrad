from typing import Tuple

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

from mygrad import matmul
from mygrad.nnet.activations import sigmoid, tanh
from mygrad.nnet.layers import gru
from mygrad.tensor_base import Tensor
from tests.utils import does_not_raise


@settings(deadline=None)
@given(
    s0=st.none()
    | hnp.arrays(shape=(1, 2), dtype=float, elements=st.just(0))
    | hnp.arrays(shape=(1, 2), dtype=float, elements=st.just(0)).map(
        lambda x: Tensor(x)
    )
    | hnp.arrays(shape=(1, 2), dtype=float, elements=st.just(0)).map(
        lambda x: Tensor(x, constant=True)
    ),
    dropout=st.floats(0, 1),
    out_constant=st.booleans(),
)
def test_nonconstant_s0_raises(s0, dropout: float, out_constant: bool):
    T, N, C, D = 5, 1, 3, 2
    X = Tensor(np.random.rand(T, N, C))
    Wz, Wr, Wh = Tensor(np.random.rand(3, D, D))
    Uz, Ur, Uh = Tensor(np.random.rand(3, C, D))
    bz, br, bh = Tensor(np.random.rand(3, D))

    with does_not_raise() if (
        out_constant or s0 is None or isinstance(s0, np.ndarray) or s0.constant
    ) else pytest.raises(ValueError):
        gru(
            X,
            Uz,
            Wz,
            bz,
            Ur,
            Wr,
            br,
            Uh,
            Wh,
            bh,
            s0=s0,
            dropout=dropout,
            constant=out_constant,
        )


@settings(deadline=None)
@given(out_constant=st.booleans())
def test_all_constant(out_constant: bool):
    T, N, C, D = 5, 1, 3, 2
    X = Tensor(np.random.rand(T, N, C), constant=True)
    Wz, Wr, Wh = Tensor(np.random.rand(3, D, D), constant=True)
    Uz, Ur, Uh = Tensor(np.random.rand(3, C, D), constant=True)
    bz, br, bh = Tensor(np.random.rand(3, D), constant=True)

    gru(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, constant=out_constant).backward()

    assert X.grad is None

    assert Wz.grad is None
    assert Wr.grad is None
    assert Wh.grad is None

    assert Uz.grad is None
    assert Ur.grad is None
    assert Uh.grad is None

    assert bz.grad is None
    assert br.grad is None
    assert bh.grad is None


@settings(deadline=None)
@given(
    X=hnp.arrays(
        shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
        dtype=float,
        elements=st.floats(-10, 10),
    ),
    D=st.sampled_from(list(range(1, 5))),
    dropout=st.sampled_from([0, 0.45]),
    data=st.data(),
)
def test_gru_fwd(X, D, dropout, data: st.DataObject):
    T, N, C = X.shape

    Wz, Wr, Wh = data.draw(
        hnp.arrays(shape=(3, D, D), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="Wz, Wr, Wh",
    )

    Uz, Ur, Uh = data.draw(
        hnp.arrays(shape=(3, C, D), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="Uz, Ur, Uh",
    )

    bz, br, bh = data.draw(
        hnp.arrays(shape=(3, D), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="bz, br, bh",
    )

    V = data.draw(
        hnp.arrays(shape=(D, C), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="V",
    )

    s0 = np.zeros((N, D), dtype=float)

    X = Tensor(X)
    X2 = X.__copy__()

    Wz = Tensor(Wz)
    Wz2 = Wz.__copy__()

    Uz = Tensor(Uz)
    Uz2 = Uz.__copy__()

    bz = Tensor(bz)
    bz2 = bz.__copy__()

    Wr = Tensor(Wr)
    Wr2 = Wr.__copy__()

    Ur = Tensor(Ur)
    Ur2 = Ur.__copy__()

    br = Tensor(br)
    br2 = br.__copy__()

    Wh = Tensor(Wh)
    Wh2 = Wh.__copy__()

    Uh = Tensor(Uh)
    Uh2 = Uh.__copy__()

    bh = Tensor(bh)
    bh2 = bh.__copy__()

    V = Tensor(V)
    V2 = V.__copy__()

    s0 = Tensor(s0)
    s2 = s0.__copy__()

    s = gru(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, dropout=dropout, constant=True)
    o = matmul(s[1:], V)
    ls = o.sum()

    assert s.constant is True

    if dropout:
        for d in [
            s.creator._dropUr,
            s.creator._dropUz,
            s.creator._dropUh,
            s.creator._dropWr,
            s.creator._dropWz,
            s.creator._dropWh,
        ]:
            assert np.all(np.logical_or(d == 1 / (1 - dropout), d == 0))

    stt = s2
    all_s = [s0.data]
    ls2 = 0
    if dropout:
        Wz2d = s.creator._dropWz * Wz2
        Wr2d = s.creator._dropWr * Wr2
        Wh2d = s.creator._dropWh * Wh2
    else:
        Wz2d = Wz2
        Wr2d = Wr2
        Wh2d = Wh2
    for n, x in enumerate(X2):
        if not dropout:
            z = sigmoid(matmul(x, Uz2) + matmul(stt, Wz2d) + bz2)
            r = sigmoid(matmul(x, Ur2) + matmul(stt, Wr2d) + br2)
            h = tanh(matmul(x, Uh2) + matmul((r * stt), Wh2d) + bh2)
        else:
            z = sigmoid(
                (s.creator._dropUz[0] * matmul(x, Uz2)) + matmul(stt, Wz2d) + bz2
            )
            r = sigmoid(
                (s.creator._dropUr[0] * matmul(x, Ur2)) + matmul(stt, Wr2d) + br2
            )
            h = tanh(
                (s.creator._dropUh[0] * matmul(x, Uh2)) + matmul((r * stt), Wh2d) + bh2
            )

        stt = (1 - z) * h + z * stt
        all_s.append(stt)
        o = matmul(stt, V2)
        ls2 += o.sum()

    tolerances = dict(atol=1e-5, rtol=1e-5)
    rec_s_dat = np.stack([i.data for i in all_s])

    assert_allclose(ls.data, ls2.data, **tolerances)

    assert_allclose(rec_s_dat, s.data, **tolerances)

    assert_allclose(Wz.data, Wz2.data, **tolerances)
    assert_allclose(Wr.data, Wr2.data, **tolerances)
    assert_allclose(Wh.data, Wh2.data, **tolerances)

    assert_allclose(Uz.data, Uz2.data, **tolerances)
    assert_allclose(Ur.data, Ur2.data, **tolerances)
    assert_allclose(Uh.data, Uh2.data, **tolerances)

    assert_allclose(bz.data, bz2.data, **tolerances)
    assert_allclose(br.data, br2.data, **tolerances)
    assert_allclose(bh.data, bh2.data, **tolerances)

    assert_allclose(V.data, V2.data, **tolerances)

    assert_allclose(X.data, X2.data, **tolerances)

    ls.null_gradients()
    for x in [s, Wz, Wr, Wh, bz, br, bh, X, Uz, Ur, Uh, V]:
        assert x.grad is None


# There is an occasional overflow in the oracle sigmoid
# that is acceptable - reducing the input domain to ameliorate this
# would potentially mask real numerical issues
@settings(deadline=None)
@given(
    data=st.data(),
    X=hnp.arrays(
        shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
        dtype=float,
        elements=st.floats(-10, 10),
    ),
    D=st.sampled_from(list(range(1, 5))),
    bp_lim=st.booleans(),
    dropout=st.sampled_from([0, 0.45]),
    U_constants=st.tuples(*[st.booleans()] * 3),
    W_constants=st.tuples(*[st.booleans()] * 3),
    b_constants=st.tuples(*[st.booleans()] * 3),
    X_constant=st.booleans(),
    V_constant=st.booleans(),
)
@pytest.mark.filterwarnings("ignore: overflow encountered in exp")
def test_gru_backward(
    data: st.DataObject,
    X: np.ndarray,
    D: int,
    bp_lim: bool,
    dropout: bool,
    U_constants: Tuple[bool, bool, bool],
    W_constants: Tuple[bool, bool, bool],
    b_constants: Tuple[bool, bool, bool],
    X_constant: bool,
    V_constant: bool,
):
    tolerances = dict(atol=1e-5, rtol=1e-5)
    T, N, C = X.shape

    Wz, Wr, Wh = data.draw(
        hnp.arrays(shape=(3, D, D), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="Wz, Wr, Wh",
    )

    Uz, Ur, Uh = data.draw(
        hnp.arrays(shape=(3, C, D), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="Uz, Ur, Uh",
    )

    bz, br, bh = data.draw(
        hnp.arrays(shape=(3, D), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="bz, br, bh",
    )

    V = data.draw(
        hnp.arrays(shape=(D, C), dtype=float, elements=st.floats(-10.0, 10.0)),
        label="V",
    )

    s0 = np.zeros((N, D), dtype=float)

    X = Tensor(X, constant=X_constant)
    X2 = X.__copy__()

    Wz = Tensor(Wz, constant=W_constants[0])
    Wz2 = Wz.__copy__()

    Uz = Tensor(Uz, constant=U_constants[0])
    Uz2 = Uz.__copy__()

    bz = Tensor(bz, constant=b_constants[0])
    bz2 = bz.__copy__()

    Wr = Tensor(Wr, constant=W_constants[1])
    Wr2 = Wr.__copy__()

    Ur = Tensor(Ur, constant=U_constants[1])
    Ur2 = Ur.__copy__()

    br = Tensor(br, constant=b_constants[1])
    br2 = br.__copy__()

    Wh = Tensor(Wh, constant=W_constants[2])
    Wh2 = Wh.__copy__()

    Uh = Tensor(Uh, constant=U_constants[2])
    Uh2 = Uh.__copy__()

    bh = Tensor(bh, constant=b_constants[2])
    bh2 = bh.__copy__()

    V = Tensor(V, constant=V_constant)
    V2 = V.__copy__()

    s0 = Tensor(s0)
    s2 = s0.__copy__()

    # bp_lim = len(X) - 1 should behave the same as no bp-lim
    s = gru(
        X,
        Uz,
        Wz,
        bz,
        Ur,
        Wr,
        br,
        Uh,
        Wh,
        bh,
        dropout=dropout,
        constant=False,
        bp_lim=len(X) - 1 if bp_lim else None,
    )
    o = matmul(s[1:], V)
    ls = o.sum()
    ls.backward()

    stt = s2
    all_s = [s0.data]
    ls2 = 0
    if dropout:
        Wz2d = s.creator._dropWz * Wz2
        Wr2d = s.creator._dropWr * Wr2
        Wh2d = s.creator._dropWh * Wh2
    else:
        Wz2d = Wz2
        Wr2d = Wr2
        Wh2d = Wh2
    for n, x in enumerate(X2):
        if not dropout:
            z = sigmoid(matmul(x, Uz2) + matmul(stt, Wz2d) + bz2)
            r = sigmoid(matmul(x, Ur2) + matmul(stt, Wr2d) + br2)
            h = tanh(matmul(x, Uh2) + matmul((r * stt), Wh2d) + bh2)
        else:
            z = sigmoid(
                (s.creator._dropUz[0] * matmul(x, Uz2)) + matmul(stt, Wz2d) + bz2
            )
            r = sigmoid(
                (s.creator._dropUr[0] * matmul(x, Ur2)) + matmul(stt, Wr2d) + br2
            )
            h = tanh(
                (s.creator._dropUh[0] * matmul(x, Uh2)) + matmul((r * stt), Wh2d) + bh2
            )
        stt = (1 - z) * h + z * stt
        all_s.append(stt)
        o = matmul(stt, V2)
        ls2 += o.sum()
    ls2.backward()

    rec_s_grad = np.stack([i.grad for i in all_s[1:]])

    if not s.constant:
        assert_allclose(rec_s_grad, s.grad, **tolerances)
    else:
        assert s.grad is None

    if not Wz.constant:
        assert_allclose(Wz.grad, Wz2.grad, **tolerances)
    else:
        assert Wz.grad is None

    if not Wr.constant:
        assert_allclose(Wr.grad, Wr2.grad, **tolerances)
    else:
        assert Wr.grad is None

    if not Wh.constant:
        assert_allclose(Wh.grad, Wh2.grad, **tolerances)
    else:
        assert Wh.grad is None

    if not Uz.constant:
        assert_allclose(Uz.grad, Uz2.grad, **tolerances)
    else:
        assert Uz.grad is None

    if not Ur.constant:
        assert_allclose(Ur.grad, Ur2.grad, **tolerances)
    else:
        assert Ur.grad is None

    if not Uh.constant:
        assert_allclose(Uh.grad, Uh2.grad, **tolerances)
    else:
        assert Uh.grad is None

    if not bz.constant:
        assert_allclose(bz.grad, bz2.grad, **tolerances)
    else:
        assert bz.grad is None

    if not br.constant:
        assert_allclose(br.grad, br2.grad, **tolerances)
    else:
        assert br.grad is None

    if not bh.constant:
        assert_allclose(bh.grad, bh2.grad, **tolerances)
    else:
        assert bh.grad is None

    if not V.constant:
        assert_allclose(V.grad, V2.grad, **tolerances)
    else:
        assert V.grad is None

    if not X.constant:
        assert_allclose(X.grad, X2.grad, **tolerances)
    else:
        assert X.grad is None

    ls.null_gradients()
    ls2.null_gradients()

    for x in [s, Wz, Wr, Wh, bz, br, bh, X, Uz, Ur, Uh, V]:
        assert x.grad is None
