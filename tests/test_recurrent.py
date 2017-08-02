from mygrad.tensor_base import Tensor
from mygrad.nnet.layers import RecurrentUnit, dense
from mygrad.nnet.layers.recurrent import OldRecurrentUnit
from mygrad.nnet.activations import tanh
from mygrad.math import add_sequence

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data(), st.choices())
def test_recurrent(data, choice):
    X = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, 10)))
    T, N, C = X.shape
    D = choice(list(range(1, 5)))

    s0 = data.draw(hnp.arrays(shape=(N, D),
                              dtype=float,
                              elements=st.floats(0.0, 0.0)))
    
    W = data.draw(hnp.arrays(shape=(D, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    U = data.draw(hnp.arrays(shape=(C, D),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    V = data.draw(hnp.arrays(shape=(D, C),
                             dtype=float,
                             elements=st.floats(-10.0, 10.0)))

    X = Tensor(X)
    X2 = X.__copy__()

    W = Tensor(W)
    W2 = W.__copy__()

    U = Tensor(U)
    U2 = U.__copy__()

    V = Tensor(V)
    V2 = V.__copy__()

    s0 = Tensor(s0)
    s2 = s0.__copy__()

    rec = RecurrentUnit(U, W, T, constant=False)
    s = rec(X)
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
        stt = tanh(dense(x, U2) + dense(stt, W2))
        all_s.append(stt)
        o = dense(stt, V2)
        ls2 += o.sum()
    ls2.backward()

    rec_s_dat = np.stack([i.data for i in all_s])
    rec_s_grad = np.stack([i.grad for i in all_s[1:]])
    assert np.allclose(rec_s_dat, s.data)
    assert np.allclose(rec_s_grad, s.grad[1:])
    assert np.allclose(ls.data, ls2.data)
    assert np.allclose(W.data, W2.data)
    assert np.allclose(W.grad, W2.grad)
    assert np.allclose(U.data, U2.data)
    assert np.allclose(U.grad, U2.grad)
    assert np.allclose(V.data, V2.data)
    assert np.allclose(V.grad, V2.grad)
    # assert np.allclose(X.data, X2.data)
    # assert np.allclose(X.grad, X2.grad)

