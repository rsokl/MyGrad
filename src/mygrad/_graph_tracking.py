"""
Provides user interface for suspending computational graph tracking and back-propagation
"""

from functools import wraps
from typing import Callable

__all__ = ["no_autodiff"]


# If `False`, suspends all computational graph tracking and backprop
TRACK_GRAPH = True  # type: bool


class NoAutoDiff:
    """ Serves as a context manager and decorator for suspending
    all computational graph tracking."""

    def __enter__(self):
        """Suspends graph-tracking"""
        global TRACK_GRAPH
        TRACK_GRAPH = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restores graph-tracking"""
        global TRACK_GRAPH
        TRACK_GRAPH = True

    def __call__(self, func: Callable) -> Callable:
        """A decorated function will have graph-tracking suspended
        during its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper


no_autodiff = NoAutoDiff()
