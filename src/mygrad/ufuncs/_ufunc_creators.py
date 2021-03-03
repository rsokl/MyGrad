import sys
from abc import ABC, ABCMeta, abstractmethod
from inspect import signature
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_type_hints,
    overload,
)

import numpy as np

from mygrad import Tensor
from mygrad.operation_base import BinaryUfunc, Operation, Ufunc, UnaryUfunc, _NoValue
from mygrad.typing import ArrayLike, DTypeLikeReals, Index, Mask, Real

__all__ = ["ufunc_creator"]


if sys.version_info >= (3, 8):  # pragma: no cover
    from typing import Literal

    HAS_LITERAL = True
else:  # pragma: no cover
    try:
        from typing_extensions import Literal
    except ImportError:
        HAS_LITERAL = False
    else:
        HAS_LITERAL = True

if TYPE_CHECKING and HAS_LITERAL:  # pragma: no cover
    One = Literal[1]
    Two = Literal[2]
else:  # pragma: no cover
    One = int
    Two = int


T = TypeVar("T")


def _permitted_type_str(x: str) -> bool:
    for code in ["F", "D", "G", "M", "m", "O"]:
        # filter non bool/int/float types
        if code in x:
            return False
    return True


class MyGradUfunc(type):
    _wrapped_op: Union[Type[UnaryUfunc], Type[BinaryUfunc]]
    _decorated_func: Callable
    nin: int
    nout: int
    nargs: int
    ntypes: int
    types: List[str]
    identity: int
    signature: str

    def __call__(cls, *args, **kwargs) -> Tensor:  # pragma: no cover
        raise NotImplementedError()

    def __repr__(cls) -> str:
        return f"<mygrad-ufunc '{cls._decorated_func.__name__}'>"


class Final(type):
    def __new__(mcs, name, bases, classdict):
        for b in bases:
            if isinstance(b, Final):
                raise TypeError("mygrad.ufunc cannot be subclassed")
        return type.__new__(mcs, name, bases, dict(classdict))


class FinalABC(ABCMeta, Final):
    pass


class ufunc(ABC, metaclass=FinalABC):
    """A un-inheritable, abstract superclass of all MyGrad ufuncs.

    Examples
    --------
    >>> import mygrad as mg
    >>> issubclass(mg.add, mg.ufunc)
    True
    """

    nin: int
    nout: int
    nargs: int
    ntypes: int
    types: List[str]
    identity: int
    signature: str

    def __call__(
        self,
        *args: ArrayLike,
        dtype: DTypeLikeReals,
        constant: Optional[bool] = None,
        **kwargs,
    ) -> Tensor:  # pragma: no cover
        pass

    def at(
        self,
        a: ArrayLike,
        indices: Union[ArrayLike, Index, Tuple[ArrayLike, Index]],
        b: Optional[ArrayLike] = None,
        *,
        constant: Optional[bool] = None,
    ) -> Tensor:  # pragma: no cover
        """Not implemented"""
        raise NotImplementedError()

    def accumulate(
        self,
        array: ArrayLike,
        axis: int = 0,
        dtype: DTypeLikeReals = None,
        out: Optional[Union[Tensor, np.ndarray]] = None,
        *,
        constant: Optional[bool] = None,
    ) -> Tensor:  # pragma: no cover
        """Not implemented"""
        raise NotImplementedError()

    def outer(
        self,
        a: ArrayLike,
        b: ArrayLike,
        *,
        dtype: DTypeLikeReals,
        out: Optional[Union[Tensor, np.ndarray]],
    ) -> Tensor:  # pragma: no cover
        """Not Implemented"""
        raise NotImplementedError()

    def reduce(
        self,
        a: ArrayLike,
        axis: Optional[Union[int, Tuple[int, ...]]] = 0,
        dtype: DTypeLikeReals = None,
        out: Optional[Union[Tensor, np.ndarray]] = None,
        keepdims: bool = False,
        initial: Real = _NoValue,
        where: Mask = True,
    ) -> Tensor:  # pragma: no cover
        """Not Implemented"""
        raise NotImplementedError()

    @abstractmethod
    def reduceat(
        self,
        a: ArrayLike,
        indices: ArrayLike,
        axis: Optional[Union[int, Tuple[int, ...]]] = 0,
        dtype: DTypeLikeReals = None,
        out: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tensor:  # pragma: no cover
        """Not Implemented"""
        raise NotImplementedError()


class MyGradUnaryUfunc(MyGradUfunc):
    _wrapped_op: Type[UnaryUfunc]
    nin: One

    def __call__(
        cls,
        x: ArrayLike,
        out: Optional[Union[Tensor, np.ndarray]] = None,
        *,
        where: Mask = True,
        dtype: DTypeLikeReals = None,
        constant: Optional[bool] = None,
    ) -> Tensor:
        # it is fastest to check if out is None, which is likely the
        # most common scenario, and this is a very "hot path" in the
        # code
        if out is not None and isinstance(out, Tensor):
            out._in_place_op(
                cls._wrapped_op,
                x,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
            )
            return out
        else:
            return Tensor._op(
                cls._wrapped_op,
                x,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
                out=out,
            )


class MyGradBinaryUfunc(MyGradUfunc):
    _wrapped_op: Type[BinaryUfunc]
    nin: Two

    def __call__(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        out: Optional[Union[Tensor, np.ndarray]] = None,
        *,
        where: Mask = True,
        dtype: DTypeLikeReals = None,
        constant: Optional[bool] = None,
    ) -> Tensor:
        # it is fastest to check if out is None, which is likely the
        # most common scenario, and this is a very "hot path" in the
        # code
        if out is not None and isinstance(out, Tensor):
            out._in_place_op(
                cls._wrapped_op,
                x,
                y,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
            )
            return out
        else:
            return Tensor._op(
                cls._wrapped_op,
                x,
                y,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
                out=out,
            )


class MyGradBinaryUfuncNoMask(MyGradUfunc):
    _wrapped_op: Type[BinaryUfunc]
    nin: Two

    def __call__(
        cls,
        x: ArrayLike,
        y: ArrayLike,
        out: Optional[Union[Tensor, np.ndarray]] = None,
        *,
        dtype: DTypeLikeReals = None,
        constant: Optional[bool] = None,
    ) -> Tensor:
        # it is fastest to check if out is None, which is likely the
        # most common scenario, and this is a very "hot path" in the
        # code
        if out is not None and isinstance(out, Tensor):
            out._in_place_op(
                cls._wrapped_op,
                x,
                y,
                op_kwargs={"dtype": dtype},
                constant=constant,
            )
            return out
        else:
            return Tensor._op(
                cls._wrapped_op,
                x,
                y,
                op_kwargs={"dtype": dtype},
                constant=constant,
                out=out,
            )


@overload
def _create_ufunc(
    op: Type[UnaryUfunc],
    decorated_func: Callable,
    *,
    at_op: Optional[Type[Operation]] = None,
    accumulate_op: Optional[Type[Operation]] = None,
    outer_op: Optional[Type[Operation]] = None,
    reduce_op: Optional[Type[Operation]] = None,
    reduceat_op: Optional[Type[Operation]] = None,
) -> MyGradUnaryUfunc:  # pragma: no cover
    ...


@overload
def _create_ufunc(
    op: Type[BinaryUfunc],
    decorated_func: Callable,
    *,
    at_op: Optional[Type[Operation]] = None,
    accumulate_op: Optional[Type[Operation]] = None,
    outer_op: Optional[Type[Operation]] = None,
    reduce_op: Optional[Type[Operation]] = None,
    reduceat_op: Optional[Type[Operation]] = None,
) -> MyGradBinaryUfunc:  # pragma: no cover
    ...


# noinspection PyPep8Naming
def _create_ufunc(
    op: Type[Ufunc],
    decorated_func,
    *,
    at_op=None,
    accumulate_op=None,
    outer_op=None,
    reduce_op=None,
    reduceat_op=None,
):
    def at(
        a: ArrayLike,
        indices: Union[ArrayLike, Index, Tuple[ArrayLike, Index]],
        b: Optional[ArrayLike] = None,
        *,
        constant: Optional[bool] = None,
    ) -> Tensor:  # pragma: no cover
        """Not implemented"""
        raise NotImplementedError()

    def accumulate(
        array: ArrayLike,
        axis: int = 0,
        dtype: DTypeLikeReals = None,
        out: Optional[Union[Tensor, np.ndarray]] = None,
        *,
        constant: Optional[bool] = None,
    ) -> Tensor:  # pragma: no cover
        """Not implemented"""
        raise NotImplementedError()

    def outer(
        a: ArrayLike,
        b: ArrayLike,
        *,
        dtype: DTypeLikeReals,
        out: Optional[Union[Tensor, np.ndarray]],
    ) -> Tensor:  # pragma: no cover
        """Not Implemented"""
        raise NotImplementedError()

    def reduce(
        a: ArrayLike,
        axis: Optional[Union[int, Tuple[int, ...]]] = 0,
        dtype: DTypeLikeReals = None,
        out: Optional[Union[Tensor, np.ndarray]] = None,
        keepdims: bool = False,
        initial: Real = _NoValue,
        where: Mask = True,
    ) -> Tensor:  # pragma: no cover
        """Not Implemented"""
        raise NotImplementedError()

    def reduceat(
        a: ArrayLike,
        indices: ArrayLike,
        axis: Optional[Union[int, Tuple[int, ...]]] = 0,
        dtype: DTypeLikeReals = None,
        out: Optional[Union[Tensor, np.ndarray]] = None,
    ) -> Tensor:  # pragma: no cover
        """Not Implemented"""
        raise NotImplementedError()

    if op.numpy_ufunc.nin == 1:
        MetaBuilder = MyGradUnaryUfunc
    elif op.numpy_ufunc.nin == 2:
        MetaBuilder = (
            MyGradBinaryUfunc if op._supports_where else MyGradBinaryUfuncNoMask
        )
    else:  # pragma: no cover
        raise NotImplementedError(
            "MyGrad Internal: `mygrad._utils.op_creator` only supports unary and binary ufuncs currently"
        )

    # filter out non-real dtypes
    types = [
        type_code
        for type_code in op.numpy_ufunc.types
        if _permitted_type_str(type_code)
    ]
    out = MetaBuilder(
        decorated_func.__name__,
        (object,),
        (
            dict(
                _wrapped_op=op,
                at=at,
                accumulate=accumulate,
                reduce=reduce,
                reduceat=reduceat,
                outer=outer,
                signature=op.numpy_ufunc.signature,
                identity=op.numpy_ufunc.identity,
                nargs=op.numpy_ufunc.nargs,
                nin=op.numpy_ufunc.nin,
                nout=op.numpy_ufunc.nout,
                ntypes=len(types),
                types=types,
                _decorated_func=decorated_func,
                __name__=decorated_func.__name__,
                __qualname__=decorated_func.__name__,
                __signature__=signature(decorated_func),
                __annotations__=get_type_hints(decorated_func),
                __doc__=decorated_func.__doc__,
            )
        ),
    )
    ufunc.register(out)
    return out


class ufunc_creator:
    """Wraps function stub and returns a mygrad-ufunc with the same name and
    signature.

    Examples
    --------
    >>> from mygrad.ufuncs import ufunc_creator, MyGradUnaryUfunc
    >>> from mygrad.math.trigonometric.ops import Sin
    >>> @ufunc_creator(mygrad_op=Sin)
    ... def sin(x, *, constant=None):
    ...    '''custom docs'''

    ``sin`` is now a mygrad ufunc

    >>> isinstance(sin, MyGradUnaryUfunc)
    True
    >>> sin.nin  # `sin` has all of the attributes of a numpy-ufunc
    1
    >>>
    Tensor(0.)

    It has the docstring and signature that was specified by the above stub.

    >>> sin.__doc__
    custom docs
    >>> import inspect
    >>> inspect.signature(sin)
    <Signature (x, *, constant=None)>
    """

    def __init__(
        self,
        mygrad_op: Type[Ufunc],
        *,
        at_op: Optional[Type[Operation]] = None,
        accumulate_op: Optional[Type[Operation]] = None,
        outer_op: Optional[Type[Operation]] = None,
        reduce_op: Optional[Type[Operation]] = None,
        reduceat_op: Optional[Type[Operation]] = None,
    ):
        """Provide the Operation-based implementation for the ufunc, and, optionally, implementations
        for the various methods of that ufunc (e.g. ufunc.at, ufunc.outer, etc.)

        Parameters
        ----------
        mygrad_op : Type[Ufunc]
            An operation-based implementation of a ufunc that supports back-propagation
            through mygrad tensors.
        """
        if not issubclass(mygrad_op, (UnaryUfunc, BinaryUfunc)):
            raise TypeError(
                "ufunc_creator can only accept `UnaryUfunc` and `BinaryUfunc` operations"
            )
        self.op = mygrad_op
        if any(
            item is not None
            for item in (at_op, accumulate_op, outer_op, reduce_op, reduceat_op)
        ):
            raise NotImplementedError(
                "There isn't support for binding ufunc methods presently."
            )

        self.at_op = at_op
        self.accumulate_op = accumulate_op
        self.outer_op = outer_op
        self.reduce_op = reduce_op
        self.reduceat_op = reduceat_op

    def __call__(self, decorated_func: T) -> T:
        return _create_ufunc(
            self.op,
            decorated_func=decorated_func,
            at_op=self.at_op,
            accumulate_op=self.accumulate_op,
            outer_op=self.outer_op,
            reduce_op=self.reduce_op,
            reduceat_op=self.reduceat_op,
        )
