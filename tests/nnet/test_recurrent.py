import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose

from mygrad import matmul
from mygrad.nnet.activations import tanh
from mygrad.nnet.layers import simple_RNN
from mygrad.tensor_base import Tensor


@settings(deadline=None)
@given(st.data())
def test_recurrent(data):
    X = data.draw(hnp.arrays(shape=hnp.array_shapes(max_side=5, min_dims=3, max_dims=3),
                             dtype=float,
                             elements=st.floats(-10, 10)))
    T, N, C = X.shape
    D = data.draw(st.sampled_from(list(range(1, 5))))

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

    s = simple_RNN(X, U, W)
    o = matmul(s[1:], V)
    ls = o.sum()
    ls.backward()

    stt = s2
    all_s = [s0.data]
    ls2 = 0
    for n, x in enumerate(X2):
        stt = tanh(matmul(x, U2) + matmul(stt, W2))
        all_s.append(stt)
        o = matmul(stt, V2)
        ls2 += o.sum()
    ls2.backward()

    rec_s_dat = np.stack([i.data for i in all_s])
    rec_s_grad = np.stack([i.grad for i in all_s[1:]])
    assert_allclose(rec_s_dat, s.data, atol=1e-5, rtol=1e-5)
    assert_allclose(rec_s_grad, s.grad[1:], atol=1e-5, rtol=1e-5)
    assert_allclose(ls.data, ls2.data, atol=1e-5, rtol=1e-5)
    assert_allclose(W.data, W2.data, atol=1e-5, rtol=1e-5)
    assert_allclose(W.grad, W2.grad, atol=1e-5, rtol=1e-5)
    assert_allclose(U.data, U2.data, atol=1e-5, rtol=1e-5)
    assert_allclose(U.grad, U2.grad, atol=1e-5, rtol=1e-5)
    assert_allclose(V.data, V2.data, atol=1e-5, rtol=1e-5)
    assert_allclose(V.grad, V2.grad, atol=1e-5, rtol=1e-5)
    assert_allclose(X.data, X2.data, atol=1e-5, rtol=1e-5)
    assert_allclose(X.grad, X2.grad, atol=1e-5, rtol=1e-5)

    ls.null_gradients()
    for x in [s, W, X, U, V]:
        assert x.grad is None
