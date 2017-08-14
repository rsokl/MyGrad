from mygrad.tensor_base import Tensor
from mygrad.nnet.layers import dense, GRU
from mygrad.nnet.activations import tanh, sigmoid
from mygrad.math import add_sequence

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data(), st.choices())
def test_gru(data, choice):
    X = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, 10)))
    T, N, C = X.shape
    D = choice(list(range(1, 5)))


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

    s = GRU(X, Uz, Wz, bz, Ur, Wr, br, Uh, Wh, bh)
    o = dense(s[1:], V)
    ls = o.sum()
    ls.backward()

    # rec = OldRecurrentUnit(U, W, V, T)
    #
    # if X.shape[0] > 1:
    #     s = rec(X)
    #     o = [dense(i, V).sum() for i in s]
    #     ls = add_sequence(*o[1:])
    # else:
    #     s = rec(X)
    #     ls = dense(s[1], V).sum()
    # ls.backward()

    stt = s2
    all_s = [s0.data]
    ls2 = 0
    for n, x in enumerate(X2):
        z = sigmoid(dense(x, Uz2) + dense(stt, Wz2) + bz2)
        r = sigmoid(dense(x, Ur2) + dense(stt, Wr2) + br2)
        h = tanh(dense(x, Uh2) + dense((r * stt), Wh2) + bh2)
        stt = (1 - z) * h + z * stt
        all_s.append(stt)
        o = dense(stt, V2)
        ls2 += o.sum()
    ls2.backward()

    rec_s_dat = np.stack([i.data for i in all_s])
    rec_s_grad = np.stack([i.grad for i in all_s[1:]])

    assert np.allclose(ls.data, ls2.data)

    assert np.allclose(rec_s_dat, s.data)
    assert np.allclose(rec_s_grad, s.grad)

    assert np.allclose(Wz.data, Wz2.data)
    assert np.allclose(Wr.data, Wr2.data)
    assert np.allclose(Wh.data, Wh2.data)

    assert np.allclose(Wz.grad, Wz2.grad)
    assert np.allclose(Wr.grad, Wr2.grad)
    assert np.allclose(Wh.grad, Wh2.grad)

    assert np.allclose(Uz.data, Uz2.data)
    assert np.allclose(Ur.data, Ur2.data)
    assert np.allclose(Uh.data, Uh2.data)

    assert np.allclose(Uz.grad, Uz2.grad)
    assert np.allclose(Ur.grad, Ur2.grad)
    assert np.allclose(Uh.grad, Uh2.grad)

    assert np.allclose(bz.data, bz2.data)
    assert np.allclose(br.data, br2.data)
    assert np.allclose(bh.data, bh2.data)

    assert np.allclose(bz.grad, bz2.grad)
    assert np.allclose(br.grad, br2.grad)
    assert np.allclose(bh.grad, bh2.grad)

    assert np.allclose(V.data, V2.data)
    assert np.allclose(V.grad, V2.grad)

    assert np.allclose(X.data, X2.data)
    assert np.allclose(X.grad, X2.grad)
