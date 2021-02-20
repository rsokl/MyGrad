from functools import wraps
from inspect import signature
from typing import Callable, Optional, Tuple, Type, TypeVar, Union, get_type_hints

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
        if not isinstance(out, Tensor):
            return Tensor._op(
                cls._Op,
                x,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
                out=out,
            )
        else:
            out._in_place_op(
                cls._Op,
                x,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
            )
            return out

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
        if not isinstance(out, Tensor):
            return Tensor._op(
                cls._Op,
                x,
                y,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
                out=out,
            )
        else:
            out._in_place_op(
                cls._Op,
                x,
                y,
                op_kwargs={"where": where, "dtype": dtype},
                constant=constant,
            )
            return out

    def __repr__(cls) -> str:
        return f"<mygrad-ufunc '{cls._decorated_func.__name__}'>"


T = TypeVar("T")


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
        self.outer = outer_op
        self.reduce = reduce_op
        self.reduceat = reduceat_op

    def __call__(self, decorated_func: T) -> T:
        if not issubclass(self.op, Ufunc):
            for _f in [
                self.at_op,
                self.accumulate_op,
                self.outer,
                self.reduce,
                self.reduceat,
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
                return Tensor._op(
                    self.op, *arrs, op_kwargs=kwargs, constant=constant, out=out
                )

            return wrapped

        _at = self.at_op
        _accumulate = self.accumulate_op

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

        MetaBuilder = (
            MetaUnaryUfunc if self.op.numpy_ufunc.nin == 1 else MetaBinaryUfunc
        )

        # filter out non-real dtypes
        types = [
            type_code
            for type_code in self.op.numpy_ufunc.types
            if _permitted_type_str(type_code)
        ]
        return MetaBuilder(
            decorated_func.__name__,
            (object,),
            (
                dict(
                    _Op=self.op,
                    at=at,
                    accumulate=accumulate,
                    reduce=reduce,
                    reduceat=reduceat,
                    outer=outer,
                    signature=self.op.numpy_ufunc.signature,
                    identity=self.op.numpy_ufunc.identity,
                    nargs=self.op.numpy_ufunc.nargs,
                    nin=self.op.numpy_ufunc.nin,
                    nout=self.op.numpy_ufunc.nout,
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
