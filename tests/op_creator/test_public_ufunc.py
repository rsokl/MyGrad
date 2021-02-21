from typing import get_type_hints

import numpy as np
import pytest

import mygrad
from mygrad._utils.op_creators import MetaBinaryUfunc, MetaUnaryUfunc

public_names = []
ufuncs = []

for name in sorted(
    public_name for public_name in dir(mygrad) if not public_name.startswith("_")
):
    attr = getattr(mygrad, name)
    if isinstance(attr, (MetaUnaryUfunc, MetaBinaryUfunc)):
        ufuncs.append(attr)
        public_names.append(name)


numpy_ufunc_property_names = sorted(
    name
    for name in dir(np.multiply)
    if not name.startswith("_") and not callable(getattr(np.multiply, name))
)


@pytest.mark.parametrize("ufunc", ufuncs)
def test_ufunc_annotations_available(ufunc):
    hints = get_type_hints(ufunc)
    assert hints
    assert hints["return"] is mygrad.Tensor


@pytest.mark.parametrize("ufunc", ufuncs)
def test_ufunc_docs_available(ufunc):
    assert "Parameters" in ufunc.__doc__ and "Returns" in ufunc.__doc__


@pytest.mark.parametrize("public_name, ufunc", list(zip(public_names, ufuncs)))
def test_ufunc_name_matches_public_name(public_name, ufunc):
    assert ufunc.__name__ == public_name or "divide" in public_name
