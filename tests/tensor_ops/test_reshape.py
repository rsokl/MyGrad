import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from numpy.testing import assert_allclose
from hypothesis import given

from mygrad.tensor_base import Tensor


def gen_shape(size):
    """ Given an array's size, generate a compatible random shape

        Parameters
        ----------
        size : Integral

        Returns
        -------
        Tuple[int, ...] """

    def _factors(n):
        """ Returns the divisors of n

            >>> _factors(4)
            {1, 2, 4}"""
        from functools import reduce
        return set(reduce(list.__add__,
                          ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    assert size > 0
    if size == 1:
        return (1,)

    shape = []
    rem = int(size / np.prod(shape))
    while rem > 1:
        if len(shape) > 6:
            shape.append(rem)
            break

        shape.append(np.random.choice(list(_factors(rem))))
        rem = int(size / np.prod(shape))

    return tuple(int(i) for i in shape)


@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                    dtype=float,
                    elements=st.floats(-100, 100))
       )
def test_reshape_fwd(a):
    new_shape = gen_shape(a.size)

    x = Tensor(a).reshape(new_shape)
    a = a.reshape(new_shape)

    assert x.shape == a.shape, "Tensor.reshape failed"
    assert_allclose(a, x.data), "Tensor.reshape failed"


@given(a=hnp.arrays(shape=hnp.array_shapes(max_side=3, max_dims=5),
                    dtype=float,
                    elements=st.floats(-100, 100))
       )
def test_reshape_backward(a):
    new_shape = gen_shape(a.size)
    grad = np.arange(a.size).reshape(a.shape)

    x = Tensor(a)
    o = x.reshape(new_shape)
    o.backward(grad.reshape(new_shape))

    assert x.grad.shape == grad.shape
    assert_allclose(x.grad, grad)
