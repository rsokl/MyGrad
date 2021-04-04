from inspect import signature
from typing import get_type_hints

import numpy as np
import pytest

import mygrad
from mygrad.linalg.ops import EinSum
from mygrad.math.arithmetic.ops import Positive
from mygrad.ufuncs._ufunc_creators import MyGradBinaryUfuncNoMask, ufunc_creator
from tests.utils.ufuncs import public_ufunc_names, ufuncs

ALIAS_UFUNC_NAMES = {"divide", "abs"}


@pytest.mark.parametrize("ufunc", ufuncs)
def test_known_mygrad_ufuncs_mirror_numpy_ufuncs(ufunc):
    assert issubclass(ufunc, mygrad.ufunc) is (
        type(getattr(np, ufunc.__name__)) is np.ufunc
    )


def test_mygrad_ufunc_type_not_inheritable():
    with pytest.raises(TypeError):

        class A(mygrad.ufunc):
            pass


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
    assert ufunc.__name__ == public_name or public_name in ALIAS_UFUNC_NAMES


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
    assert (
        repr(ufunc) == f"<mygrad-ufunc '{public_name}'>"
        or public_name in ALIAS_UFUNC_NAMES
    )


@pytest.mark.parametrize("ufunc", ufuncs)
@pytest.mark.parametrize("expected_param", ["out", "where", "dtype", "constant"])
def test_ufunc_signature_is_defined(ufunc, expected_param: str):
    params = signature(ufunc).parameters
    if type(ufunc) is MyGradBinaryUfuncNoMask:
        pytest.mark.skip("ufunc does not support where")
        return
    assert expected_param in params


@pytest.mark.parametrize("not_a_ufunc", [EinSum, object])
def test_ufunc_creator_raises_on_non_ufunc_op(not_a_ufunc):
    with pytest.raises(TypeError):

        @ufunc_creator(not_a_ufunc)
        def f(*args, **kwargs):
            return


def test_ufunc_method_not_supported():
    with pytest.raises(NotImplementedError):

        @ufunc_creator(Positive, at_op=Positive)
        def f(*args, **kwargs):
            return
