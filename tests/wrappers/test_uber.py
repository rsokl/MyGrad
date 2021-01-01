import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given, settings
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad import Tensor
from mygrad.operation_base import Operation
from tests.utils import clear_all_mem_locking_state

from .uber import backprop_test_factory, fwdprop_test_factory


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
            err_msg=f"Got bad value for `a`: {a}",
        )

        assert_allclose(
            np.real(b if isinstance(b, np.ndarray) else b.data),
            expected_b,
            err_msg=f"Got bad value for `b`: {b}",
        )
        assert_allclose(
            np.real(c if isinstance(c, np.ndarray) else c.data),
            expected_c,
            err_msg=f"Got bad value for `c`: {c}",
        )

        if kwarg1 is not None:
            assert isinstance(kwarg1, KWARG1)
            KWARG1.passed = True

        return a * b * c

    @settings(deadline=None, max_examples=20)
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


@mg.mem_guard_off
def test_catches_bad_fwd_pass():
    def mul2(x, constant=False):
        return mg.multiply(x, 2, constant=constant)

    def mul3(x):
        return 3 * np.asarray(x)

    @settings(deadline=None, max_examples=20)
    @fwdprop_test_factory(num_arrays=1, mygrad_func=mul2, true_func=mul3)
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


def test_catches_output_not_tensor():
    def mul2(x, constant=False):
        return np.asarray(mg.multiply(x, 2, constant=constant))

    @settings(deadline=None, max_examples=20)
    @fwdprop_test_factory(num_arrays=1, mygrad_func=mul2, true_func=mul2)
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


@mg.mem_guard_off
def test_catches_bad_constant_propagation():
    def mul2(x, constant=False):
        return mg.multiply(x, 2, constant=False)

    @settings(deadline=None, max_examples=20)
    @fwdprop_test_factory(num_arrays=1, mygrad_func=mul2, true_func=mul2)
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


@mg.mem_guard_off
def test_catches_mutation_error():
    class Mul2_Mutate(Operation):
        def __call__(self, tensor: Tensor):
            self.variables = (tensor,)
            tensor.data.flags.writeable = True
            tensor.data *= 2
            tensor.data.flags.writeable = False
            return tensor.data * 2

        def backward_var(self, grad: np.ndarray, index: int, **kwargs) -> np.ndarray:
            raise NotImplementedError()

    def mul2_mutate(x, constant=False):
        return Tensor._op(Mul2_Mutate, x, constant=constant)

    def mul2(x):
        return 2 * np.asarray(x)

    @settings(deadline=None, max_examples=20)
    @fwdprop_test_factory(num_arrays=1, mygrad_func=mul2_mutate, true_func=mul2)
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


def test_catches_numpy_view_mygrad_copy():
    def copy(x, constant):
        return mg.reshape(x, x.shape, constant=constant).copy()

    def view(x):
        return np.reshape(x, x.shape)

    @settings(deadline=None, max_examples=20)
    @fwdprop_test_factory(
        num_arrays=1, mygrad_func=copy, true_func=view, permit_0d_array_as_float=False,
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


@mg.mem_guard_off
def test_catches_numpy_copy_mygrad_view():
    def copy(x, constant):
        return mg.reshape(x, x.shape, constant=constant)

    def view(x):
        return np.reshape(x, x.shape).copy()

    @settings(deadline=None, max_examples=20)
    @fwdprop_test_factory(
        num_arrays=1, mygrad_func=copy, true_func=view, permit_0d_array_as_float=False,
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


@mg.mem_guard_off
def test_bad_constant_propagation():
    def bad_const_prop(x, y, z, constant=False):
        if not constant and any(not i.constant for i in [x, y, z]):
            constant = True
        return mg.multiply_sequence(x, y, z, constant=constant)

    @settings(deadline=None, max_examples=20)
    @fwdprop_test_factory(
        num_arrays=3,
        mygrad_func=bad_const_prop,
        true_func=lambda x, y, z: x * y * z,
        permit_0d_array_as_float=False,
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


def test_catches_incorrect_op_gradient():
    class Mul2(Operation):
        def __call__(self, tensor):
            self.variables = (tensor,)
            return tensor.data * 2

        def backward_var(self, grad, index, **kwargs):
            a = self.variables[index]
            return grad * np.ones_like(a)  # should be: grad * 2 * np.ones_like(a)

    def mul2_wrong_grad(x, constant=False):
        return Tensor._op(Mul2, x, constant=constant)

    @settings(deadline=None, max_examples=20)
    @backprop_test_factory(
        num_arrays=1, mygrad_func=mul2_wrong_grad, true_func=lambda x: 2 * x
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


def test_catches_op_didnt_propagate_grad():
    class Mul2(Operation):
        def __call__(self, tensor):
            self.variables = (tensor,)
            return tensor.data * 2

        def backward_var(self, grad, index, **kwargs):
            a = self.variables[index]
            return 2 * np.ones_like(a)  # should be: grad * 2 * np.ones_like(a)

    def mul2_doesnt_prop_incoming_grad(x, constant=False):
        return Tensor._op(Mul2, x, constant=constant)

    @settings(deadline=None, max_examples=20)
    @backprop_test_factory(
        num_arrays=1,
        mygrad_func=mul2_doesnt_prop_incoming_grad,
        true_func=lambda x: 2 * x,
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


def test_catches_mutated_gradient():
    class Mul2(Operation):
        def __call__(self, tensor):
            self.variables = (tensor,)
            return tensor.data * 2

        def backward_var(self, grad, index, **kwargs):
            a = self.variables[index]
            grad *= 2 * np.ones_like(a)
            return grad

    def mul2_backprop_mutates_grad(x, constant=False):
        return Tensor._op(Mul2, x, constant=constant)

    @settings(deadline=None, max_examples=20)
    @backprop_test_factory(
        num_arrays=1, mygrad_func=mul2_backprop_mutates_grad, true_func=lambda x: 2 * x
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


def test_catches_backprop_mutated_input():
    class Mul2(Operation):
        def __call__(self, tensor):
            self.variables = (tensor,)
            return tensor.data * 2

        def backward_var(self, grad, index, **kwargs):
            a = self.variables[index]  # type: Tensor
            a.data.flags.writeable = True
            a.data *= 3
            a.data.flags.writeable = False
            return grad * 2 * np.ones_like(a)

    def mul2_backprop_mutates_input(x, constant=False):
        return Tensor._op(Mul2, x, constant=constant)

    @settings(deadline=None, max_examples=20)
    @backprop_test_factory(
        num_arrays=1, mygrad_func=mul2_backprop_mutates_input, true_func=lambda x: 2 * x
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()


def tests_catches_input_tensors_memory_not_locked_by_op():
    def mul_releases_input_lock(x, y, constant=False):
        out = mg.multiply(x, y, constant=constant)
        x.data.flags.writeable = True
        return out

    @settings(deadline=None, max_examples=20)
    @backprop_test_factory(
        num_arrays=2, mygrad_func=mul_releases_input_lock, true_func=lambda x, y: x * y
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()

    clear_all_mem_locking_state()


def tests_catches_output_tensors_memory_not_locked_by_op():
    def mul_releases_output_lock(x, y, constant=False):
        out = mg.multiply(x, y, constant=constant)
        out.data.flags.writeable = True
        return out

    @settings(deadline=None, max_examples=20)
    @backprop_test_factory(
        num_arrays=2, mygrad_func=mul_releases_output_lock, true_func=lambda x, y: x * y
    )
    def should_catch_error():
        pass

    with pytest.raises(AssertionError):
        should_catch_error()

    clear_all_mem_locking_state()
