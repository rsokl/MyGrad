import warnings
from collections import defaultdict
from functools import partial
from typing import Dict, Hashable, Optional, Union

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.ufuncs import MyGradBinaryUfunc, MyGradUnaryUfunc
from tests.custom_strategies import populates_ufunc
from tests.utils.numerical_gradient import numerical_gradient
from tests.wrappers.uber import backprop_test_factory

ufuncs = [
    mg.add,
    mg.arccos,
    mg.arcsin,
    mg.arctan,
    mg.arctan2,
    mg.cos,
    mg.divide,
    mg.exp,
    mg.exp2,
    mg.expm1,
    mg.log,
    mg.log10,
    mg.log1p,
    mg.log2,
    mg.logaddexp,
    mg.logaddexp2,
    mg.multiply,
    mg.negative,
    mg.positive,
    mg.power,
    mg.reciprocal,
    mg.sin,
    mg.square,
    mg.subtract,
    mg.tan,
    mg.true_divide,
]

DOES_NOT_SUPPORT_COMPLEX_DOMAIN = {mg.logaddexp, mg.logaddexp2, mg.arctan2}


not_zero = st.floats(-1e9, 1e9).filter(lambda x: not np.isclose(x, 0, atol=1e-5))

log_largest = np.log(np.finfo(np.float64).max)
log2_largest = np.log2(np.finfo(np.float64).max)
valid_log_inputs = st.floats(min_value=0, allow_infinity=False, exclude_min=True)
valid_exp_inputs = st.floats(max_value=log_largest, allow_infinity=False)
valid_exp2_inputs = st.floats(max_value=log2_largest, allow_infinity=False)

# Specifies the domain from which ufunc numerical inputs are drawn to test fwd-prop.
# These are generally meant to match the true domain for the given function.
FWD_DOMAINS: Dict[Hashable, Optional[Dict[int, st.SearchStrategy]]] = defaultdict(
    lambda: None
)
FWD_DOMAINS[mg.arccos] = {0: st.floats(-1, 1, exclude_min=True, exclude_max=True)}
FWD_DOMAINS[mg.arcsin] = {0: st.floats(-1, 1, exclude_min=True, exclude_max=True)}
FWD_DOMAINS[mg.divide] = {1: not_zero}
FWD_DOMAINS[mg.exp] = {0: valid_exp_inputs}
FWD_DOMAINS[mg.exp2] = {0: valid_exp2_inputs}
FWD_DOMAINS[mg.expm1] = {0: valid_exp_inputs}
FWD_DOMAINS[mg.log] = {0: valid_log_inputs}
FWD_DOMAINS[mg.log2] = {0: valid_log_inputs}
FWD_DOMAINS[mg.log10] = {0: valid_log_inputs}
FWD_DOMAINS[mg.log1p] = {
    0: st.floats(min_value=-1 + np.finfo(np.float64).eps, allow_infinity=False)
}
FWD_DOMAINS[mg.logaddexp] = {
    0: valid_exp_inputs,
    1: valid_exp_inputs,
}
FWD_DOMAINS[mg.power] = {
    0: st.floats(min_value=1e-9, max_value=1e9),
    1: st.sampled_from([1, 2]) | st.floats(-3, 3),
}
FWD_DOMAINS[mg.reciprocal] = {0: not_zero}
FWD_DOMAINS[mg.true_divide] = {1: not_zero}
FWD_DOMAINS[mg.tan] = {
    0: st.floats(-np.pi / 2, np.pi / 2, exclude_min=True, exclude_max=True)
}


@pytest.mark.parametrize("ufunc", ufuncs)
@given(data=st.data())
def test_ufunc_fwd(
    data: st.DataObject,
    ufunc: Union[MyGradUnaryUfunc, MyGradBinaryUfunc],
):
    """
    Checks:
    - mygrad and numpy results agree numerically and in dtype
    - op doesn't mutate inputs
    - tensor base matches array base
    - constant propagates as expected
    - grad shape consistency
    """
    args = data.draw(
        populates_ufunc(
            ufunc, arg_index_to_elements=FWD_DOMAINS[ufunc], min_side=0, min_dims=0
        ),
        label="ufunc call sig",
    )

    args.make_array_based_args_read_only()  # guards against mutation
    # Explicitly retrieve numpy's ufunc of the same name.
    # Don't trust that mygrad ufunc is binds the correct numpy ufunc under the hood
    numpy_ufunc = getattr(np, ufunc.__name__)

    numpy_out = numpy_ufunc(*args.args_as_no_mygrad(), **args)
    mygrad_out = ufunc(*args, **args)

    assert numpy_out.dtype == mygrad_out.dtype

    # Check that numpy and mygrad implementations agree
    assert_allclose(actual=mygrad_out, desired=numpy_out)
    assert mygrad_out.base is None and numpy_out.base is None

    # Check that constant propagated properly
    expected_constant = all(t.constant is True for t in args.tensors_only())
    assert mygrad_out.constant is expected_constant

    with warnings.catch_warnings():
        # It is okay if we backprop on unstable domains here;
        # we are not testing for numerical correctness here, but rather want to
        # ensure that general combinations of inputs never produce errors upon
        # backprop.
        warnings.simplefilter("ignore")
        mygrad_out.backward()

    # Check grad shape consistency
    for n, t in enumerate(args.tensors_only()):
        assert t.constant or t.grad.shape == t.shape, f"arg: {n}"


easy_log_domain = st.floats(min_value=1e-6, max_value=1e10)

# Specifies the domain from which ufunc numerical inputs are drawn to test fwd-prop.
# These are tailored to ensure that the resulting derivatives can be checked against
# numerical approximations in a reliable way
BKWD_DOMAINS: Dict[Hashable, Optional[Dict[int, st.SearchStrategy]]] = defaultdict(
    lambda: None
)
BKWD_DOMAINS[mg.arccos] = {0: st.floats(-1, 1, exclude_min=True, exclude_max=True)}
BKWD_DOMAINS[mg.arcsin] = {0: st.floats(-1, 1, exclude_min=True, exclude_max=True)}
BKWD_DOMAINS[mg.divide] = {1: not_zero}
BKWD_DOMAINS[mg.exp] = {0: valid_exp_inputs}
BKWD_DOMAINS[mg.exp2] = {0: valid_exp2_inputs}
BKWD_DOMAINS[mg.expm1] = {0: valid_exp_inputs}
BKWD_DOMAINS[mg.log] = {0: easy_log_domain}
BKWD_DOMAINS[mg.log2] = {0: easy_log_domain}
BKWD_DOMAINS[mg.log10] = {0: easy_log_domain}
BKWD_DOMAINS[mg.log1p] = {0: st.floats(min_value=-1 + 1e-6, max_value=1e9)}
BKWD_DOMAINS[mg.logaddexp] = {
    0: valid_exp_inputs,
    1: valid_exp_inputs,
}
BKWD_DOMAINS[mg.power] = {
    0: st.floats(0.001, 1e9),
    1: st.sampled_from([1, 2]) | st.floats(-3, 3),
}
BKWD_DOMAINS[mg.reciprocal] = {0: not_zero}
BKWD_DOMAINS[mg.true_divide] = {1: not_zero}
BKWD_DOMAINS[mg.tan] = {
    0: st.floats(-np.pi / 2, np.pi / 2, exclude_min=True, exclude_max=True)
}


@pytest.mark.parametrize(
    "ufunc", [u for u in ufuncs if u not in DOES_NOT_SUPPORT_COMPLEX_DOMAIN]
)
@given(data=st.data())
def test_ufunc_bkwd(
    data: st.DataObject,
    ufunc: Union[MyGradUnaryUfunc, MyGradBinaryUfunc],
):
    """
    Checks:
    - backprop matches numerical gradient
    - backprop doesn't mutate grad
    """
    args = data.draw(
        populates_ufunc(
            ufunc,
            arg_index_to_elements=BKWD_DOMAINS[ufunc],
            tensor_only=True,
            min_side=1,
        ),
        label="ufunc call sig",
    )
    args.make_array_based_args_read_only()  # guards against mutation

    mygrad_out = ufunc(*args.args, **args.kwargs)

    # Draw upstream gradient to be backpropped
    # We limit this to [-1, 1] so that we are less likely to encounter
    # invalid domains
    grad = data.draw(
        hnp.arrays(dtype=float, shape=mygrad_out.shape, elements=st.floats(-1e0, 1e0)),
        label="grad",
    )

    grad.flags["WRITEABLE"] = False  # will raise if backprop mutates grad

    mygrad_out.backward(grad)
    kwargs = args.kwargs.copy()

    # numerical grad needs to write complex-valued outputs
    kwargs["out"] = np.zeros_like(mygrad_out, dtype=complex)
    numpy_ufunc = partial(getattr(np, ufunc.__name__), **kwargs)

    grads = numerical_gradient(numpy_ufunc, *args.args_as_no_mygrad(), back_grad=grad)

    # check that gradients match numerical derivatives
    for n in range(ufunc.nin):
        desired = np.nan_to_num(grads[n])
        actual = args.args[n].grad
        assert_allclose(
            desired=desired,
            actual=actual,
            err_msg=f"the grad of tensor-{n} did not match the "
            f"numerically-computed gradient",
        )


# logaddexp can't be evaluated with complex values
@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=mg.logaddexp,
    true_func=np.logaddexp,
    num_arrays=2,
    index_to_bnds=(-10, 10),
    use_finite_difference=True,
    h=1e-8,
    atol=1e-4,
    rtol=1e-4,
)
def test_logaddexp_bkwd():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=mg.logaddexp2,
    true_func=np.logaddexp2,
    num_arrays=2,
    atol=1e-4,
    rtol=1e-4,
    use_finite_difference=True,
    h=1e-8,
    index_to_bnds=(-100, 100),
)
def test_logaddexp2_bkwd():
    pass


@backprop_test_factory(
    mygrad_func=mg.arctan2,
    true_func=np.arctan2,
    num_arrays=2,
    atol=1e-4,
    rtol=1e-4,
    index_to_bnds={0: (1e-4, 1e9)},
    use_finite_difference=True,
    h=1e-8,
)
def test_arctan2_bkwd_pos_x():
    pass


@backprop_test_factory(
    mygrad_func=mg.arctan2,
    true_func=np.arctan2,
    num_arrays=2,
    atol=1e-4,
    rtol=1e-4,
    index_to_bnds={0: (-1e9, -1e-4)},
    use_finite_difference=True,
    h=1e-8,
)
def test_arctan2_bkwd_neg_x():
    pass
