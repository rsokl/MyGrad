from decimal import Decimal
import numpy as np

from hypothesis import given, assume
import hypothesis.strategies as st
import hypothesis.extra.numpy as hnp


def to_decimal_array(arr):
   return np.array(tuple(Decimal(float(i)) for i in arr.flat), dtype=Decimal).reshape(arr.shape)


@st.composite
def numerical_gradient(draw, f, *, x, y, vary_ind=0, h=1e-8):
    h = Decimal(h)
    x = to_decimal_array(x)
    y = to_decimal_array(y)

    if vary_ind == 0:
        dx = (f(x + h, y) - f(x - h, y)) / (Decimal(2) * h)
    else:
        dx = (f(x, y + h) - f(x, y - h)) / (Decimal(2) * h)
    return x.astype(float), y.astype(float), dx.astype(float)

