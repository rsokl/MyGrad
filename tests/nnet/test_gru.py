from mygrad.tensor_base import Tensor
from mygrad.nnet.layers import dense, gru
from mygrad.nnet.activations import tanh, sigmoid


import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np
from numpy.testing import assert_allclose


@given(st.data())
def test_gru_fwd(data):
    X = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, 10)),
                  label="X")
    T, N, C = X.shape
    D = data.draw(st.sampled_from(list(range(1, 5))), label="D")
    dropout = data.draw(st.sampled_from([0, .45]), label="dropout")


    Wz = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Wz")

    Uz = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Uz")

    bz = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="bz")

    Wr = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Wr")

    Ur = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Ur")

    br = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="br")

    Wh = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Wh")

    Uh = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Uh")

    bh = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="bh")

    V = data.draw(hnp.arrays(shape=(D, C),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)),
                  label="V")


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
    o = dense(s[1:], V)
    ls = o.sum()

    assert s.constant is True

    if dropout:
        for d in [s.creator._dropUr, s.creator._dropUz, s.creator._dropUh,
                  s.creator._dropWr, s.creator._dropWz, s.creator._dropWh]:
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
            z = sigmoid(dense(x, Uz2) + dense(stt, Wz2d) + bz2)
            r = sigmoid(dense(x, Ur2) + dense(stt, Wr2d) + br2)
            h = tanh(dense(x, Uh2) + dense((r * stt), Wh2d) + bh2)
        else:
            z = sigmoid((s.creator._dropUz[0] * dense(x, Uz2)) + dense(stt, Wz2d) + bz2)
            r = sigmoid((s.creator._dropUr[0] * dense(x, Ur2)) + dense(stt, Wr2d) + br2)
            h =    tanh((s.creator._dropUh[0] * dense(x, Uh2)) + dense((r * stt), Wh2d) + bh2)

        stt = (1 - z) * h + z * stt
        all_s.append(stt)
        o = dense(stt, V2)
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


@given(st.data())
def test_gru_backward(data):
    tolerances = dict(atol=1e-5, rtol=1e-5)
    X = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, 10)),
                  label="X")
    T, N, C = X.shape
    D = data.draw(st.sampled_from(list(range(1, 5))), label="D")
    dropout = data.draw(st.sampled_from([0, 0.45]), label="dropout")  # TODO: RESTORE DROPOUT


    Wz = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Wz")

    Uz = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Uz")

    bz = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="bz")

    Wr = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Wr")

    Ur = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Ur")

    br = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="br")

    Wh = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Wh")

    Uh = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="Uh")

    bh = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)),
                   label="bh")

    V = data.draw(hnp.arrays(shape=(D, C),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)),
                  label="V")

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

    s = gru(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, dropout=dropout, constant=False)
    o = dense(s[1:], V)
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
            z = sigmoid(dense(x, Uz2) + dense(stt, Wz2d) + bz2)
            r = sigmoid(dense(x, Ur2) + dense(stt, Wr2d) + br2)
            h = tanh(dense(x, Uh2) + dense((r * stt), Wh2d) + bh2)
        else:
            z = sigmoid((s.creator._dropUz[0] * dense(x, Uz2)) + dense(stt, Wz2d) + bz2)
            r = sigmoid((s.creator._dropUr[0] * dense(x, Ur2)) + dense(stt, Wr2d) + br2)
            h =    tanh((s.creator._dropUh[0] * dense(x, Uh2)) + dense((r * stt), Wh2d) + bh2)
        stt = (1 - z) * h + z * stt
        all_s.append(stt)
        o = dense(stt, V2)
        ls2 += o.sum()
    ls2.backward()

    rec_s_grad = np.stack([i.grad for i in all_s[1:]])

    assert_allclose(rec_s_grad, s.grad, **tolerances)

    assert_allclose(Wz.grad, Wz2.grad, **tolerances)
    assert_allclose(Wr.grad, Wr2.grad, **tolerances)
    assert_allclose(Wh.grad, Wh2.grad, **tolerances)

    assert_allclose(Uz.grad, Uz2.grad, **tolerances)
    assert_allclose(Ur.grad, Ur2.grad, **tolerances)
    assert_allclose(Uh.grad, Uh2.grad, **tolerances)

    assert_allclose(bz.grad, bz2.grad, **tolerances)
    assert_allclose(br.grad, br2.grad, **tolerances)
    assert_allclose(bh.grad, bh2.grad, **tolerances)

    assert_allclose(V.grad, V2.grad, **tolerances)

    assert_allclose(X.grad, X2.grad, **tolerances)

    ls.null_gradients()
    ls2.null_gradients()
    for x in [s, Wz, Wr, Wh, bz, br, bh, X, Uz, Ur, Uh, V]:
        assert x.grad is None
