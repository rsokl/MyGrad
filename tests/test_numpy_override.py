import pytest

from mygrad.tensor_base import (
    _REGISTERED_DIFFERENTIABLE_NUMPY_FUNCS,
    _REGISTERED_NO_DIFF_NUMPY_FUNCS,
    implements_numpy_override,
)


def test_override_on_non_numpy_function():
    def not_a_numpy_function():
        pass

    with pytest.raises(AttributeError):
        implements_numpy_override()(not_a_numpy_function)


@pytest.mark.parametrize(
    "numpy_func",
    sorted(_REGISTERED_DIFFERENTIABLE_NUMPY_FUNCS, key=lambda x: x.__name__),
)
def test_registered_numpy_overrides_are_callables(numpy_func):
    assert callable(numpy_func)
    assert callable(_REGISTERED_DIFFERENTIABLE_NUMPY_FUNCS[numpy_func])


@pytest.mark.parametrize(
    "numpy_func", sorted(_REGISTERED_NO_DIFF_NUMPY_FUNCS, key=lambda x: x.__name__)
)
def test_registered_non_diff_numpy_overrides_are_callables(numpy_func):
    assert callable(numpy_func)
