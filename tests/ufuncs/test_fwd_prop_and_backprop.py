from collections import defaultdict
from functools import reduce
from itertools import zip_longest
from typing import Mapping, Optional, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.typing import Shape
from mygrad.ufuncs import MyGradBinaryUfunc, MyGradUnaryUfunc
from tests.custom_strategies import array_likes, no_value
from tests.utils.functools import MinimalArgs, populate_args


def _broadcast_two_shapes(shape_a: Shape, shape_b: Shape) -> Shape:
    result = []
    for a, b in zip_longest(reversed(shape_a), reversed(shape_b), fillvalue=1):
        if a != b and (a != 1) and (b != 1):
            raise ValueError(
                f"shapes {shape_a!r} and {shape_b!r} are not broadcast-compatible"
            )
        result.append(a if a != 1 else b)
    return tuple(reversed(result))


def _broadcast_shapes(*shapes):
    """Returns the shape resulting from broadcasting the
    input shapes together.
    Raises ValueError if the shapes are not broadcast-compatible"""
    assert len(shapes)
    return reduce(_broadcast_two_shapes, shapes, ())


@st.composite
def populates_ufunc(
    draw: st.DataObject.draw,
    ufunc: Union[MyGradUnaryUfunc, MyGradBinaryUfunc],
    arg_index_to_elements: Optional[Mapping[int, st.SearchStrategy]] = None,
) -> st.SearchStrategy[MinimalArgs]:
    ind_to_elements = defaultdict(lambda: st.floats(-1e-9, 1e9))

    if arg_index_to_elements is not None:
        for k, v in arg_index_to_elements.items():
            ind_to_elements[k] = v

    shapes: hnp.BroadcastableShapes = draw(
        hnp.mutually_broadcastable_shapes(
            num_shapes=ufunc.nin + 1, min_side=0, max_dims=3, max_side=3
        ),
        label="shapes",
    )

    args = populate_args(
        *(
            draw(array_likes(shape=shape, dtype=float, elements=ind_to_elements[n]))
            for n, shape in enumerate(shapes.input_shapes[:-1])
        ),
        where=draw(
            no_value()
            | st.booleans().map(lambda x: not x)
            | hnp.arrays(
                shape=shapes.input_shapes[-1], dtype=bool, elements=st.booleans()
            )
        ),
    )
    fill_value = draw(st.integers(0, 1))

    where = args.kwargs.get("where", True)
    if where is not True:
        # the predicted results shape can be wrong if the inputs don't `where`
        # broadcast against `where`
        out_shape = (
            shapes.result_shape
            if not isinstance(where, bool)
            else _broadcast_shapes(*(mg.asarray(a).shape for a in args.args))
        )
        args["out"] = np.full(out_shape, fill_value=fill_value, dtype=float)
    return args


not_zero = st.floats(-1e9, 1e9).filter(lambda x: not np.isclose(x, 0, atol=1e-5))


@pytest.mark.parametrize(
    "ufunc, domains",
    [
        (mg.negative, None),
        (mg.positive, None),
        (mg.reciprocal, {0: not_zero}),
        (mg.add, None),
        (mg.multiply, None),
        (
            mg.power,
            {0: st.floats(0.001, 1e9), 1: st.sampled_from([1, 2]) | st.floats(-3, 3)},
        ),
        (mg.square, None),
        (mg.subtract, None),
        (mg.true_divide, {1: not_zero}),
        (mg.divide, {1: not_zero}),
    ],
)
@given(data=st.data())
def test_ufunc_fwd(
    data: st.DataObject,
    ufunc: Union[MyGradUnaryUfunc, MyGradBinaryUfunc],
    domains: Optional[Mapping[int, st.SearchStrategy]],
):
    """
    Checks:
    - mygrad implementation of ufunc against numpy
    - tensor base matches array base
    - constant propagates as expected
    - grad shape consistency
    """
    args = data.draw(
        populates_ufunc(ufunc, arg_index_to_elements=domains), label="ufunc args"
    )

    # Explicitly retrieve numpy's ufunc of the same name.
    # Don't trust that mygrad ufunc is binds the correct numpy ufunc under the hood
    numpy_ufunc = getattr(numpy, ufunc.__name__)

    numpy_out = numpy_ufunc(*args.args_as_no_mygrad(), **args.kwargs)
    mygrad_out = ufunc(*args.args, **args.kwargs)

    # Check that numpy and mygrad implementations agree
    assert_allclose(actual=mygrad_out, desired=numpy_out)
    assert mygrad_out.base is None and numpy_out.base is None

    # Check that constant propagated properly
    expected_constant = all(t.constant is True for t in args.tensors_only())
    assert mygrad_out.constant is expected_constant
    mygrad_out.backward()

    # Check grad shape consistency
    for n, t in enumerate(args.tensors_only()):
        assert t.constant or t.grad.shape == t.shape, f"arg: {n}"
