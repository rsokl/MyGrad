from mygrad.tensor_base import Tensor
from mygrad.nnet.layers import RecurrentUnit, dense
from mygrad.nnet.activations import tanh
from mygrad.math import add_sequence

import hypothesis.strategies as st
from hypothesis import given
import hypothesis.extra.numpy as hnp

import numpy as np


@given(st.data(), st.choices())
def test_recurrent(data, choice):
    X = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=3, min_dims=3, max_dims=3),
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


    W = Tensor(W)
    W2 = W.__copy__()

    U = Tensor(U)
    U2 = U.__copy__()

    V = Tensor(V)
    V2 = V.__copy__()

    s0 = Tensor(s0)
    s2 = s0.__copy__()

    rec = RecurrentUnit(U, W, V, T)

    if X.shape[0] > 1:
        ls = add_sequence(*(dense(i, V).sum() for i in rec(X)[1:]))
    else:
        ls = dense(rec(X)[1], V).sum()
    ls.backward()

    s = s2
    ls2 = 0
    for n, x in enumerate(X):
        s = tanh(dense(x, U2) + dense(s, W2))
        o = dense(s, V2)
        ls2 += o.sum()
    ls2.backward()

    assert np.allclose(W.data, W2.data, atol=1E-3)
    assert np.allclose(W.grad, W2.grad, atol=1E-3)
    assert np.allclose(U.data, U2.data, atol=1E-3)
    assert np.allclose(U.grad, U2.grad, atol=1E-3)
    assert np.allclose(V.data, V2.data, atol=1E-3)
    assert np.allclose(V.grad, V2.grad, atol=1E-3)
