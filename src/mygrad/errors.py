class MyGradException(Exception):
    """Generic parent class for exceptions thrown by MyGrad."""


class InvalidGradient(MyGradException):
    """An invalid gradient (i.e. a non-numeric or non-array-like object)
    was produced by an operation or was supplied by a user."""


class InvalidBackprop(MyGradException):
    """Backpropagation was invoked through a partially-cleared graph
    or from a non-scalar for a scalar-only graph"""


class DisconnectedView(MyGradException):
    custom_msg = (
        "An inplace operation was invoked on a tensor-view that "
        "is not connected to its base tensor through a computational "
        "graph."
        "It is likely that this tensor was involved during backpropagation;"
        "consider recreating this view before proceeding."
    )
