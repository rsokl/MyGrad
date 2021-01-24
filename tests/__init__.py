from typing import Union

import numpy as np

import mygrad as mg


def is_float_arr(arr: Union[np.ndarray, mg.Tensor]) -> bool:
    return issubclass(arr.dtype.type, np.integer)
