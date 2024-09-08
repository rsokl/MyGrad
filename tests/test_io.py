from pathlib import Path
from string import ascii_lowercase
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings
from numpy.testing import assert_array_equal

from mygrad import Tensor, load, save
from tests.custom_strategies import everything_except, tensors

filenames = st.text(ascii_lowercase, min_size=1).map(lambda x: x + ".npz")


@given(
    fname=filenames,
    as_path=st.booleans(),
    tensor=tensors(
        include_grad=st.booleans(), dtype=st.sampled_from(["float32", "float64"])
    ),
)
@pytest.mark.usefixtures("cleandir")
def test_save_load_roundtrip(fname: str, as_path: bool, tensor: Tensor):
    if as_path:
        fname = Path(fname)

    save(fname, tensor)
    loaded = load(fname)
    assert_array_equal(tensor, loaded)

    if tensor.grad is None:
        assert loaded.grad is None
    else:
        assert_array_equal(tensor.grad, loaded.grad)


@settings(max_examples=10, suppress_health_check=(HealthCheck.too_slow,))
@given(
    fname=filenames,
    as_path=st.booleans(),
    not_tensor=everything_except(Tensor),
)
@pytest.mark.usefixtures("cleandir")
def test_validation(fname: str, as_path: bool, not_tensor: Any):
    if as_path:
        fname = Path(fname)

    with pytest.raises(TypeError):
        save(fname, not_tensor)
