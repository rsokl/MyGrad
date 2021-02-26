import mygrad as mg
from mygrad.operation_base import _NoValue

from .functools import populate_args


def test_populate_args():
    pargs = populate_args(1, None, a_kwarg=1, not_set=_NoValue)
    assert (1, None) == pargs.args
    assert dict(a_kwarg=1) == pargs.kwargs


def test_populate_args_tensor_only():
    x = mg.tensor([1])
    y = mg.tensor(1)
    x1, y1 = populate_args(1, x, [1], y, mg.asarray(x), a_kwarg=None).tensors_only()
    assert x is x1
    assert y is y1


def test_populate_args_as_no_mygrad():
    x = mg.tensor([1])
    a, b, c = populate_args(1, x, None).args_as_no_mygrad()
    assert a == 1
    assert b is x.data
    assert c is None
