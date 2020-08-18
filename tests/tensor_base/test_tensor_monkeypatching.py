import pytest

attrs = [
    "sum",
    "prod",
    "cumprod",
    "cumsum",
    "mean",
    "std",
    "var",
    "max",
    "min",
    "argmax",
    "argmin",
    "swapaxes",
    "transpose",
    "moveaxis",
    "squeeze",
    "ravel",
    "matmul",
    "any",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
]


@pytest.mark.parametrize("attribute", attrs)
def test_Tensor_from_tensorbase_has_makeypatched_attributes(attribute: str):
    from mygrad.tensor_base import Tensor

    assert hasattr(Tensor, attribute)


def test_toplevel_Tensor_and_base_Tensor_are_identical():
    from mygrad.tensor_base import Tensor as Tensor_from_base
    from mygrad import Tensor as Tensor_from_toplevel

    assert Tensor_from_toplevel.__dict__ == Tensor_from_base.__dict__
