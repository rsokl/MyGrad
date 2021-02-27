import mygrad as mg
from mygrad.operation_base import _NoValue

from .functools import SmartSignature


def test_init():
    pargs = SmartSignature(1, None, a_kwarg=1, not_set=_NoValue)
    assert (1, None) == pargs.args
    assert dict(a_kwarg=1) == pargs.kwargs


def test_tensor_only():
    x = mg.tensor([1])
    y = mg.tensor(1)
    x1, y1 = SmartSignature(1, x, [1], y, mg.asarray(x), a_kwarg=None).tensors_only()
    assert x is x1
    assert y is y1


def test_as_no_mygrad():
    x = mg.tensor([1])
    a, b, c = SmartSignature(1, x, None).args_as_no_mygrad()
    assert a == 1
    assert b is x.data
    assert c is None


def test_smart_sig_named_unpack():
    assert dict(**SmartSignature(1, 2, _NoValue, a=-1, b=-2, c=_NoValue)) == dict(
        a=-1, b=-2
    )


def test_iter_unpack():
    assert [*SmartSignature(1, 2, _NoValue, 4, a=-1, b=-2, c=_NoValue)] == [1, 2, 4]


def test_setitem_ignores_no_value():
    sig = SmartSignature()
    sig["a"] = _NoValue
    sig["b"] = 1
    assert dict(**sig) == {"b": 1}
