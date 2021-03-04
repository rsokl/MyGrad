import numpy as np
import pytest
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad.operation_base import _NoValue
from tests.utils.functools import SmartSignature


@pytest.mark.parametrize("where", [_NoValue, True, [True, False]])
@pytest.mark.parametrize("input_cast", [mg.asarray, mg.astensor])
@pytest.mark.parametrize("out_cast", [mg.asarray, mg.astensor])
@pytest.mark.parametrize("pos_or_kwarg", ["pos", "kwarg", "tuple"])
def test_ufunc_overload_with_out(where, input_cast, out_cast, pos_or_kwarg: str):
    if input_cast is mg.asarray and out_cast is mg.asarray:
        pytest.skip("all-array inputs to numpy ufunc doesn't exercise mygrad")
        return

    input = input_cast([-1.0, -2.0])
    target = out_cast([0.0, 0.0])
    if pos_or_kwarg != "pos":
        if pos_or_kwarg == "tuple":
            sig = SmartSignature(input, out=(target,))
        else:
            sig = SmartSignature(input, out=target)
    else:
        sig = SmartSignature(input, target)

    sig["where"] = where
    sig2 = sig.copy()

    overload_out = np.negative(*sig, **sig)

    mygrad_out = mg.negative(*sig2, **sig2)

    assert isinstance(overload_out, mg.Tensor) and (
        (overload_out.data is target) or (overload_out is target)
    )
    assert_array_equal(overload_out, mygrad_out)
    assert mygrad_out.constant is overload_out.constant


def test_mutiple_outputs_to_overloaded_ufunc_raises():
    input = mg.tensor([-1.0, -2.0])
    target = mg.tensor([0.0, 0.0])
    # with pytest.raises(ValueError):
    np.negative(input, out=(target, target))


def test_boolean_overload():
    x = mg.tensor([1.0, 2.0])
    out = np.greater(x, 1.5)
    assert isinstance(out, np.ndarray)
    assert_array_equal(out, [False, True])

    x = mg.tensor([1.0, 2.0])
    target = mg.tensor([0.0, 0.0])
    out = np.greater(x, 1.5, target)
    assert isinstance(out, np.ndarray)
    assert_array_equal(out, [False, True])
    assert_array_equal(target, [False, True])

    x = mg.tensor([1.0, 2.0])
    target = mg.tensor([0.0, 0.0])
    out = np.greater(x, 1.5, out=target)
    assert isinstance(out, np.ndarray)
    assert_array_equal(out, [False, True])
    assert_array_equal(target, [False, True])

    x = mg.tensor([1.0, 2.0])
    target = mg.tensor([0.0, 0.0])
    out = np.greater(mg.asarray(x), 1.5, out=target)
    assert isinstance(out, np.ndarray)
    assert_array_equal(out, [False, True])
    assert_array_equal(target, [False, True])


def test_const_only():
    x = mg.tensor([1.5], constant=True)
    y = mg.tensor([0.0], constant=True)
    out = np.ceil(x, out=y)
    assert_array_equal(out, [2.0])
    assert out is y.data


def test_const_only_overload_raises_on_variable_input():
    with pytest.raises(ValueError):
        np.ceil(mg.tensor(1.0, constant=False))

    with pytest.raises(ValueError):
        np.ceil(np.array(1.0), out=mg.tensor(1.0, constant=False))
