import functools
import mygrad as mg

from hypothesis import given, note
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp

from numpy.testing import assert_allclose


def multi_matmul_slow(arrays): return functools.reduce(mg.matmul, arrays)


@given(num_arrays=st.integers(2, 5),
       left_1d=st.booleans(),
       right_1d=st.booleans(),
       output_is_constant=st.booleans(),
       data=st.data())
def test_multi_matmul(num_arrays, left_1d, right_1d, output_is_constant, data):
    """
    Ensures that ``multi_matmul`` behaves identically to:

        functools.reduce(mg.matmul, arrays)

    Includes edge cases in which the 1st and last tensors in the sequence are 1D
    """
    shape_endpoints = data.draw(st.tuples(*[st.integers(1, 10) for i in range(num_arrays + 1)]), label="endpoints")
    shapes = [shape_endpoints[i:i+2] for i in range(num_arrays)]

    if left_1d:
        shapes[0] = shapes[0][:0:-1]

    if right_1d:
        shapes[-1] = shapes[-1][:1]

    constants = data.draw(st.tuples(*[st.booleans() for i in range(num_arrays)]), label="constants")
    output_is_constant = output_is_constant or all(constants)

    arrs = [data.draw(hnp.arrays(dtype=float, shape=shapes[i], elements=st.floats(0, 1e6),),
                      label="arr-{}".format(i))
            for i in range(num_arrays)]
    note("tensor shapes: {}".format([i.shape for i in arrs]))

    arrs1 = [mg.Tensor(x, constant=const) for x, const in zip(arrs, constants)]
    arrs2 = [x.__copy__() for x in arrs1]

    actual = mg.multi_matmul(arrs1, constant=output_is_constant)
    desired = multi_matmul_slow(arrs2)
    assert_allclose(actual.data, desired.data, atol=1e-6, rtol=1e-6,
                    err_msg="`multi_matmul` does not produce the same result as "
                            "`functools.reduce(mg.matmul, arrays)`")

    assert actual.constant is output_is_constant, "`multi_matmul` does not carry constant info properly"

    if output_is_constant:
        return

    grad = data.draw(hnp.arrays(shape=desired.shape, dtype=float, elements=st.floats(0, 1e6)))

    (desired * grad).sum().backward()
    (actual * grad).sum().backward()

    for n, (const, arr1, arr2) in enumerate(zip(constants, arrs1, arrs2)):
        assert const is arr1.constant is arr2.constant, "tensor-{}-constant was not set properly".format(n)

        if const:
            assert arr2.grad is None, "tensor-{} is a constant, but its gradient is not `None`".format(n)
        else:
            assert_allclose(arr1.grad, arr2.grad, atol=1e-6, rtol=1e-6,
                            err_msg="The gradients for tensor-{} for not match".format(n))

    actual.null_gradients()
    for n, arr1 in enumerate(arrs1):
        assert arr1.grad is None, "tensor-{} did not get its gradient nulled".format(n)

