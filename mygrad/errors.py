class MyGradException(Exception):
    """Generic parent class for exceptions thrown by MyGrad."""


class InvalidGradient(MyGradException):
    """An invalid gradient (i.e. a non-numeric or non-array-like object)
    was produced by an operation or was supplied by a user."""


class InvalidBackprop(MyGradException):
    """Backpropagation was invoked through a partially-cleared graph
    or from a non-scalar for a scalar-only graph"""