import pytest

import mygrad as mg
from mygrad._dtype_mirrors import __all__ as all_mirrored_dtyped


@pytest.mark.parametrize("dtype_str", all_mirrored_dtyped)
def test_mirrored_dtype_is_valid(dtype_str):
    mg.tensor(1, dtype=getattr(mg, dtype_str))
