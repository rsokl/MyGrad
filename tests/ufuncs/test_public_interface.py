from inspect import signature
from typing import get_type_hints

import numpy as np
import pytest

import mygrad
from tests.utils.ufuncs import public_ufunc_names, ufuncs


@pytest.mark.parametrize("ufunc", ufuncs)
def test_bound_numpy_ufunc_is_actually_a_ufunc(ufunc):
    assert isinstance(ufunc._wrapped_op.numpy_ufunc, np.ufunc)


numpy_ufunc_property_names = sorted(
    name
    for name in dir(np.multiply)
    if not name.startswith("_") and not callable(getattr(np.multiply, name))
)

numpy_ufunc_method_names = sorted(
    name
    for name in dir(np.multiply)
    if not name.startswith("_") and callable(getattr(np.multiply, name))
)

mygrad_type_codes = sorted(set(code[0] for ufunc in ufuncs for code in ufunc.types))


@pytest.mark.parametrize("ufunc", ufuncs)
def test_ufunc_annotations_available(ufunc):
    hints = get_type_hints(ufunc)
    assert hints
    assert hints["return"] is mygrad.Tensor


@pytest.mark.parametrize("ufunc", ufuncs)
def test_ufunc_docs_available(ufunc):
    assert "Parameters" in ufunc.__doc__ and "Returns" in ufunc.__doc__


@pytest.mark.parametrize("public_name, ufunc", list(zip(public_ufunc_names, ufuncs)))
def test_ufunc_name_matches_public_name(public_name, ufunc):
    assert ufunc.__name__ == public_name or "divide" in public_name


@pytest.mark.parametrize("ufunc", ufuncs)
@pytest.mark.parametrize("method_name", numpy_ufunc_method_names)
def test_ufunc_methods_exist(ufunc, method_name):
    assert hasattr(ufunc, method_name)


@pytest.mark.parametrize("public_name, ufunc", list(zip(public_ufunc_names, ufuncs)))
@pytest.mark.parametrize("property_name", numpy_ufunc_property_names)
def test_ufunc_attribute_matches(public_name, ufunc, property_name):
    if "type" in property_name:
        # mygrad ufuncs have restricted type support
        assert hasattr(ufunc, property_name)
    else:
        numpy_ufunc = getattr(np, public_name)
        expected = getattr(numpy_ufunc, property_name)
        actual = getattr(ufunc, property_name)
        assert expected == actual


@pytest.mark.parametrize("code", mygrad_type_codes)
def test_type_codes_are_supported_by_tensor(code: str):
    x = mygrad.tensor([1, 0], dtype=np.dtype(code))
    assert issubclass(x.dtype.type, (np.floating, np.integer, np.bool_))
    if issubclass(x.dtype.type, np.floating):
        assert x.constant is False
    elif issubclass(x.dtype.type, (np.integer, np.bool_)):
        assert x.constant is True
    else:
        raise AssertionError()


@pytest.mark.parametrize("public_name, ufunc", list(zip(public_ufunc_names, ufuncs)))
def test_ufunc_repr(public_name: str, ufunc):
    if public_name == "divide":
        return
    assert repr(ufunc) == f"<mygrad-ufunc '{public_name}'>"


@pytest.mark.parametrize("ufunc", ufuncs)
@pytest.mark.parametrize("expected_param", ["out", "where", "dtype", "constant"])
def test_ufunc_signature_is_defined(ufunc, expected_param: str):
    params = signature(ufunc).parameters
    assert expected_param in params
