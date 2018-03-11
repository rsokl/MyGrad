from .binary_func import numerical_gradient
import numpy as np

from hypothesis import given
import hypothesis.searchstrategy as st
import hypothesis.extra.numpy as hnp


def test_numerical_gradient():
    def f(x, y): return x * y**2

    # no broadcast