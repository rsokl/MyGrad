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
                             elements=st.floats(-10, 10)))
    T, N, C = X.shape
    D = data.draw(st.sampled_from(list(range(1, 5))))
    dropout = data.draw(st.sampled_from([0, .33]))


    Wz = data.draw(hnp.arrays(shape=(D, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Uz = data.draw(hnp.arrays(shape=(C, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    bz = data.draw(hnp.arrays(shape=(D,),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Wr = data.draw(hnp.arrays(shape=(D, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Ur = data.draw(hnp.arrays(shape=(C, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    br = data.draw(hnp.arrays(shape=(D,),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Wh = data.draw(hnp.arrays(shape=(D, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Uh = data.draw(hnp.arrays(shape=(C, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    bh = data.draw(hnp.arrays(shape=(D,),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    V = data.draw(hnp.arrays(shape=(D, C),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))


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

    s = gru(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, dropout=dropout)
    o = dense(s[1:], V)
    ls = o.sum()

    if dropout:
        for d in [s.creator._dropr, s.creator._dropz, s.creator._droph]:
            assert np.all(np.logical_or(d == 1 / (1 - dropout), d == 0))

    stt = s2
    all_s = [s0.data]
    ls2 = 0
    for n, x in enumerate(X2):
        z = sigmoid(dense(x, Uz2) + dense(stt, Wz2) + bz2)
        if dropout:
            z *= s.creator._dropz[n]
        r = sigmoid(dense(x, Ur2) + dense(stt, Wr2) + br2)
        if dropout:
            r *= s.creator._dropr[n]
        h = tanh(dense(x, Uh2) + dense((r * stt), Wh2) + bh2)
        if dropout:
            h *= s.creator._droph[n]
        stt = (1 - z) * h + z * stt
        all_s.append(stt)
        o = dense(stt, V2)
        ls2 += o.sum()

    rec_s_dat = np.stack([i.data for i in all_s])

    assert_allclose(ls.data, ls2.data)

    assert_allclose(rec_s_dat, s.data)

    assert_allclose(Wz.data, Wz2.data)
    assert_allclose(Wr.data, Wr2.data)
    assert_allclose(Wh.data, Wh2.data)

    assert_allclose(Uz.data, Uz2.data)
    assert_allclose(Ur.data, Ur2.data)
    assert_allclose(Uh.data, Uh2.data)

    assert_allclose(bz.data, bz2.data)
    assert_allclose(br.data, br2.data)
    assert_allclose(bh.data, bh2.data)

    assert_allclose(V.data, V2.data)

    assert_allclose(X.data, X2.data)

    ls.null_gradients()
    for x in [s, Wz, Wr, Wh, bz, br, bh, X, Uz, Ur, Uh, V]:
        assert x.grad is None


@given(st.data())
def test_gru_backward(data):
    tolerances = dict(atol=1e-5, rtol=1e-5)
    X = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, 10)))
    T, N, C = X.shape
    D = data.draw(st.sampled_from(list(range(1, 5))))
    dropout = data.draw(st.sampled_from([0, .33]))


    Wz = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    Uz = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    bz = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    Wr = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    Ur = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    br = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    Wh = data.draw(hnp.arrays(shape=(D, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    Uh = data.draw(hnp.arrays(shape=(C, D),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    bh = data.draw(hnp.arrays(shape=(D,),
                              dtype=float,
                              elements=st.floats(-10.0, 10.0)))

    V = data.draw(hnp.arrays(shape=(D, C),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

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

    s = gru(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh, dropout=dropout)
    o = dense(s[1:], V)
    ls = o.sum()
    ls.backward()

    stt = s2
    all_s = [s0.data]
    ls2 = 0
    for n, x in enumerate(X2):
        z = sigmoid(dense(x, Uz2) + dense(stt, Wz2) + bz2)
        if dropout:
            z *= s.creator._dropz[n]
        r = sigmoid(dense(x, Ur2) + dense(stt, Wr2) + br2)
        if dropout:
            r *= s.creator._dropr[n]
        h = tanh(dense(x, Uh2) + dense((r * stt), Wh2) + bh2)
        if dropout:
            h *= s.creator._droph[n]
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
