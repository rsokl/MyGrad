from typing import Union

import numpy as np
import pytest

import mygrad as mg


@pytest.mark.parametrize(
    ("mygrad_constant", "numpy_constant"),
    [
        (mg.Infinity, np.Infinity),
        (mg.NAN, np.NAN),
        (mg.NINF, np.NINF),
        (mg.NZERO, np.NZERO),
        (mg.NaN, np.NaN),
        (mg.PINF, np.PINF),
        (mg.PZERO, np.PZERO),
        (mg.e, np.e),
        (mg.euler_gamma, np.euler_gamma),
        (mg.inf, np.inf),
        (mg.infty, np.infty),
        (mg.nan, np.nan),
        (mg.newaxis, np.newaxis),
    ],
)
def test_constants(
    mygrad_constant: Union[None, float], numpy_constant: Union[None, float]
):
    assert mygrad_constant is numpy_constant
