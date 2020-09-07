from contextlib import contextmanager
from typing import Dict, Union

import numpy as np

from mygrad import Tensor


@contextmanager
def does_not_raise():
    """An 'empty' constext manager that yields ``None``. This is
    to be used in conjunction with ``pytest.raises`` in scenarios
    where the tested function ought not raise any exception.

    Examples
    --------
    >>> import pytest
    >>> x = "hello"
    >>> with (pytest.raises(AttributeError) if not isinstance(x, str) else does_not_raise()):
        ... x.lower()
    """
    yield


array_flag_fields = (
    "ALIGNED",
    "BEHAVED",
    "C_CONTIGUOUS",
    "CARRAY",
    "CONTIGUOUS",
    "F_CONTIGUOUS",
    "FARRAY",
    "FNC",
    "FORC",
    "FORTRAN",
    "OWNDATA",
    "WRITEABLE",
    "WRITEBACKIFCOPY",
)


def flags_to_dict(x: Union[Tensor, np.ndarray]) -> Dict[str, bool]:
    arr = np.asarray(x)
    return {k: arr.flags[k] for k in array_flag_fields}
