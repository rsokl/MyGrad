import inspect
from typing import Type

import numpy as np
import pytest

from mygrad.operation_base import Ufunc


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        (s for c in cls.__subclasses__() for s in all_subclasses(c))
    )


all_concrete_ufuncs = [
    ufunc for ufunc in all_subclasses(Ufunc) if not inspect.isabstract(ufunc)
]


@pytest.mark.parametrize("ufunc", all_concrete_ufuncs)
@pytest.mark.parametrize(
    "attribute",
    [
        "nin",
        "nout",
        "nargs",
        "ntypes",
        "types",
        "identity",
        "signature",
    ],
)
def test_ufunc_attributes_match_numpy_counterpart(ufunc: Type[Ufunc], attribute: str):
    ufunc = ufunc()
    assert getattr(ufunc, attribute) == getattr(ufunc.numpy_ufunc, attribute)


@pytest.mark.parametrize("ufunc", all_concrete_ufuncs)
def test_numpy_ufunc_is_actually_a_ufunc(ufunc: Type[Ufunc]):
    assert isinstance(ufunc.numpy_ufunc, np.ufunc)
