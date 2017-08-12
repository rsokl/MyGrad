from mygrad.tensor_base import Tensor
from mygrad.nnet.layers import dense
from mygrad.nnet.layers.recurrent import GRUnit
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

    Wr = data.draw(hnp.arrays(shape=(D, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Ur = data.draw(hnp.arrays(shape=(C, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Wh = data.draw(hnp.arrays(shape=(D, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    Uh = data.draw(hnp.arrays(shape=(C, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    V = data.draw(hnp.arrays(shape=(D, C),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))


    s0 = np.zeros((N, D), dtype=float)

    Wz = Tensor(Wz)
    Wz2 = Wz.__copy__()

    Uz = Tensor(Uz)
    Uz2 = Uz.__copy__()

    Wr = Tensor(Wr)
    Wr2 = Wr.__copy__()

    Ur = Tensor(Ur)
    Ur2 = Ur.__copy__()

    Wh = Tensor(Wh)
    Wh2 = Wh.__copy__()

    Uh = Tensor(Uh)
    Uh2 = Uh.__copy__()

    V = Tensor(V)
    V2 = V.__copy__()

    s0 = Tensor(s0)
    s2 = s0.__copy__()

    gru = GRUnit(Uz, Wz, Ur, Wr, Uh, Wh, V, T)
    s = gru(X)
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
    for n, x in enumerate(X):
        z = sigmoid(dense(x, Uz2) + dense(stt, Wz2))
        r = sigmoid(dense(x, Ur2) + dense(stt, Wr2))
        h = tanh(dense(x, Uh2) + dense((r * stt), Wh2))
        stt = (1 - z) * h + z * stt
        all_s.append(stt.data)
        o = dense(stt, V2)
        ls2 += o.sum()
    ls2.backward()

    all_s = np.stack(all_s)
    z = np.stack(z.grad)
    r = np.stack(r.grad)
    h = np.stack(h.grad)
    s = np.stack([i.data for i in s])

    assert np.allclose(all_s, s)
    assert np.allclose(ls.data, ls2.data)

    #assert np.allclose(gru._z.grad, z)
    #assert np.allclose(gru._r.grad, r)
    #assert np.allclose(gru._h.grad, h)

    assert np.allclose(Wz.data, Wz2.data)
    assert np.allclose(Wr.data, Wr2.data)
    assert np.allclose(Wh.data, Wh2.data)

    assert np.allclose(Wz.grad, Wz2.grad)
    assert np.allclose(Wr.grad, Wr2.grad)
    assert np.allclose(Wh.grad, Wh2.grad)

    assert np.allclose(Uz.data, Uz2.data)
    assert np.allclose(Ur.data, Ur2.data)
    assert np.allclose(Uh.data, Uh2.data)

    # assert np.allclose(Uz.grad, Uz2.grad)
    # assert np.allclose(Ur.grad, Ur2.grad)
    # assert np.allclose(Uh.grad, Uh2.grad)

    assert np.allclose(V.data, V2.data)
    assert np.allclose(V.grad, V2.grad)
