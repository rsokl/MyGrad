from functools import wraps
from inspect import signature
from typing import (
    Callable,
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

__all__ = ["op_creator"]


def _permitted_type_str(x: str) -> bool:
    for code in ["F", "D", "G", "M", "m", "O"]:
        # filter non bool/int/float types
        if code in x:
            return False
    return True


class MetaUnaryUfunc(type):
    _Op: Type[UnaryUfunc]
    _decorated_func: Callable

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
                cls._Op,
                x,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
            )
            return out
        else:
            return Tensor._op(
                cls._Op,
                x,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
                out=out,
            )

    def __repr__(cls) -> str:
        return f"<mygrad-ufunc '{cls._decorated_func.__name__}'>"


class MetaBinaryUfunc(type):
    _Op: Type[BinaryUfunc]
    _decorated_func: Union[Callable, Ufunc]

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
                cls._Op,
                x,
                y,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
            )
            return out
        else:
            return Tensor._op(
                cls._Op,
                x,
                y,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
                out=out,
            )

    def __repr__(cls) -> str:
        return f"<mygrad-ufunc '{cls._decorated_func.__name__}'>"


T = TypeVar("T")


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
) -> MetaUnaryUfunc:
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
) -> MetaBinaryUfunc:
    ...


# noinspection PyPep8Naming
def _create_ufunc(
    op,
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
        MetaBuilder = MetaUnaryUfunc
    elif op.numpy_ufunc.nin == 2:
        MetaBuilder = MetaBinaryUfunc
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
    return MetaBuilder(
        decorated_func.__name__,
        (object,),
        (
            dict(
                _Op=op,
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


class op_creator:
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
        self.op = mygrad_op
        self.at_op = at_op
        self.accumulate_op = accumulate_op
        self.outer_op = outer_op
        self.reduce_op = reduce_op
        self.reduceat_op = reduceat_op

    def __call__(self, decorated_func: T) -> T:
        if not issubclass(self.op, (UnaryUfunc, BinaryUfunc)):
            for _f in [
                self.at_op,
                self.accumulate_op,
                self.outer_op,
                self.reduce_op,
                self.reduceat_op,
            ]:
                if _f is not None:  # pragma: no cover
                    raise Exception(
                        "MyGrad Internal Error: ufunc method specified for non-ufunc op."
                    )

            @wraps(decorated_func)
            def wrapped(
                *arrs: ArrayLike,
                constant: Optional[bool] = None,
                out: Optional[Union[np.ndarray, Tensor]] = None,
                **kwargs,
            ) -> Tensor:
                # it is fastest to check if out is None, which is likely the
                # most common scenario, and this is a very "hot path" in the
                # code
                if out is not None and isinstance(out, Tensor):
                    out._in_place_op(
                        self.op, *arrs, op_kwargs=kwargs, constant=constant
                    )
                    return out

                else:
                    return Tensor._op(
                        self.op, *arrs, op_kwargs=kwargs, constant=constant, out=out
                    )

            return wrapped

        else:
            return _create_ufunc(
                self.op,
                decorated_func=decorated_func,
                at_op=self.at_op,
                accumulate_op=self.accumulate_op,
                outer_op=self.outer_op,
                reduce_op=self.reduce_op,
                reduceat_op=self.reduceat_op,
            )
