from contextlib import contextmanager


class InternalTestError(Exception):
    """Marks errors that are caused by bad test configurations"""


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