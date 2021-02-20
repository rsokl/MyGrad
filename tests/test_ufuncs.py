import inspect
from typing import Type

import numpy as np
import pytest

from mygrad.operation_base import Ufunc


def all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        (s for c in cls.__subclasses__() for s in all_subclasses(c))
    )


# names need to be sorted so parallel testing doesn't break
all_concrete_ufuncs = sorted(
    (ufunc for ufunc in all_subclasses(Ufunc) if (not inspect.isabstract(ufunc))),
    key=lambda x: x.__name__,
)


@pytest.mark.parametrize("ufunc", all_concrete_ufuncs)
def test_numpy_ufunc_is_actually_a_ufunc(ufunc: Type[Ufunc]):
    assert isinstance(ufunc.numpy_ufunc, np.ufunc)
