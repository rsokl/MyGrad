"""
Provides user interface for suspending computational graph tracking and back-propagation
"""
from functools import wraps
from typing import Callable

import numpy as np

from mygrad._utils import ContextTracker

__all__ = ["no_autodiff"]


# If `False`, suspends all computational graph tracking and backprop
TRACK_GRAPH = True  # type: bool


class _NoAutoDiff(ContextTracker):
    """ Serves as a context manager and decorator for suspending
    all computational graph tracking.

    Note that memory guarding does not occur in the `no_autodiff` context,
    so there is no need to nest this context with `mem_guard_off`.
    """

    _enter_set_value = False

    @property
    def state(self):
        return TRACK_GRAPH

    @state.setter
    def state(self, value: bool):
        if not isinstance(value, bool):  # pragma: no cover
            raise TypeError(
                f"TRACK_GRAPH must be set to a boolean value, got {value} (type={type(value)})"
            )

        global TRACK_GRAPH
        TRACK_GRAPH = value

    def __call__(self, func: Callable, to_numpy: bool = False) -> Callable:
        """Decorates a function so that it will have graph-tracking suspended
        during its execution.

        Parameters
        ----------
        func : Callable
            The function to be decorated

        to_numpy : bool, optional (default=False)
            If true, the output is assumed to be array-like and
            will be cast to a numpy array

        Returns
        -------
        decorated_func : Callable"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                out = func(*args, **kwargs)
            return out if not to_numpy else np.asarray(out)

        return wrapper


no_autodiff = _NoAutoDiff()
