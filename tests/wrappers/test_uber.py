import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from numpy.testing import assert_allclose

import mygrad as mg

from .uber import backprop_test_factory


class KWARG1:
    passed = False


name_to_pos = dict(a=0, b=1, c=2)


@settings(deadline=None, max_examples=20)
@given(
    args_as_kwargs=st.fixed_dictionaries(
        {},
        optional=dict(
            a=st.just(mg.Tensor(1.0)),
            b=st.just(mg.Tensor(2.0)),
            c=st.just(mg.Tensor(3.0)),
            kwarg1=st.just(KWARG1()),
        ),
    ),
)
def test_arr_from_kwargs(args_as_kwargs):
    expected_a = 1.0 if "a" in args_as_kwargs else -1.0
    expected_b = 2.0 if "b" in args_as_kwargs else -2.0
    expected_c = 3.0 if "c" in args_as_kwargs else -3.0
    KWARG1.passed = False
    arrs_from_kwargs = {name_to_pos[k]: k for k in args_as_kwargs if k != "kwarg1"}

    def sentinel(a, b, c, kwarg1=None):
        assert_allclose(
            np.real(a if isinstance(a, np.ndarray) else a.data),
            expected_a,
            err_msg="Got bad value for `a`: {}".format(a),
        )

        assert_allclose(
            np.real(b if isinstance(b, np.ndarray) else b.data),
            expected_b,
            err_msg="Got bad value for `b`: {}".format(b),
        )
        assert_allclose(
            np.real(c if isinstance(c, np.ndarray) else c.data),
            expected_c,
            err_msg="Got bad value for `c`: {}".format(c),
        )

        if kwarg1 is not None:
            assert isinstance(kwarg1, KWARG1)
            KWARG1.passed = True

        return a * b * c

    @settings(deadline=None, max_examples=5)
    @backprop_test_factory(
        mygrad_func=sentinel,
        true_func=sentinel,
        num_arrays=3,
        index_to_arr_shapes={0: tuple(), 1: tuple(), 2: tuple()},
        index_to_bnds={0: (-1, -1), 1: (-2, -2), 2: (-3, -3)},
        arrs_from_kwargs=arrs_from_kwargs,
        kwargs=args_as_kwargs,
    )
    def factory_func():
        pass

    factory_func()

    if "kwarg1" in args_as_kwargs:
        assert KWARG1.passed
