from typing import Dict

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from numpy.testing import assert_allclose

import mygrad as mg
from mygrad.errors import InvalidBackprop
from tests.custom_strategies import tensors
from tests.utils import clear_all_mem_locking_state


@pytest.mark.parametrize("constant", [True, False])
def test_tensor_base_matches_ndarray_base(constant: bool):
    tens = mg.arange(10, constant=constant)
    arr = np.arange(10)

    assert tens.base is None
    assert arr.base is None

    t1 = tens[:5]
    a1 = arr[:5]
    assert t1.base is tens
    assert t1.base.data is tens.data
    assert a1.base is arr

    t2 = t1[:2]
    a2 = a1[:2]

    assert t2.base is tens
    assert t2.data.base is tens.data
    assert a2.base is arr

    t3 = tens + 1
    a3 = arr + 1
    assert t3.base is None
    assert a3.base is None


@pytest.mark.parametrize("constant", [True, False])
def test_views_of_non_arrays_leave_no_base(constant: bool):
    assert mg.reshape(2.0, (1,), constant=constant).base is None
    assert mg.reshape(list(range(9)), (3, 3), constant=constant).base is None


@pytest.mark.parametrize("constant", [True, False])
def test_no_share_memory_view_is_still_view(constant: bool):
    # an empty array can be a view without sharing memory
    array = np.array([])
    array_view = array[tuple()]
    assert array_view.base is array, "expected numpy behavior does not hold"

    array_view_of_view = array_view[tuple()]
    assert (
        array_view_of_view.base is not array_view
    ), "expected numpy behavior does not hold"
    assert array_view_of_view.base is array, "expected numpy behavior does not hold"

    tensor = mg.Tensor([], constant=constant)
    tensor_view = tensor[tuple()]
    assert tensor_view.base is tensor

    tensor_view_of_view = tensor_view[tuple()]
    assert tensor_view_of_view.base is not tensor_view
    assert tensor_view_of_view.base is tensor


def create_view_graph(base_constant: bool = False) -> Dict[str, mg.Tensor]:
    """
    Creates the following graph:

               base -------------------
                |                     |
                |                 leaf_view
                |                     |
            downstream_view      view_of_leaf_view
                |
        view_of_downstream_view
                |
    * Backprop will only occur *
    * along this branch.       *

    """
    base = mg.Tensor([0.0, 1.0, 2.0, 3.0], constant=base_constant)
    downstream_v = base[:2]
    downstream_v_v = downstream_v[...]
    leaf_view = base[-2:]
    view_of_leaf_view = leaf_view[...]
    return dict(
        base=base,
        downstream_view=downstream_v,
        view_of_downstream_view=downstream_v_v,
        leaf_view=leaf_view,
        view_of_leaf_view=view_of_leaf_view,
    )


@pytest.mark.parametrize("base_constant", [True, False])
@pytest.mark.parametrize(
    "view_type",
    ["downstream_view", "view_of_downstream_view", "leaf_view", "view_of_leaf_view"],
)
def test_basic_view_relationship(view_type: str, base_constant: bool):
    # The following graph is created:
    #
    # base = mg.Tensor([0., 1., 2., 3.], constant=base_constant)
    #
    # downstream_v = base[:2]
    # downstream_v_v = downstream_v[...]
    #
    # leaf_view = base[-2:]
    # view_of_leaf_view = leaf_view[...]
    #
    graph = create_view_graph(base_constant)
    assert graph[view_type].base is graph["base"]


@pytest.mark.parametrize("base_constant", [True, False])
@pytest.mark.parametrize(
    "view_type",
    ["downstream_view", "view_of_downstream_view", "leaf_view", "view_of_leaf_view"],
)
def test_view_propagates_constant(view_type: str, base_constant: bool):
    # The following graph is created:
    #
    # base = mg.Tensor([0., 1., 2., 3.], constant=base_constant)
    #
    # downstream_v = base[:2]
    # downstream_v_v = downstream_v[...]
    #
    # leaf_view = base[-2:]
    # view_of_leaf_view = leaf_view[...]
    #
    graph = create_view_graph(base_constant)
    assert graph[view_type].constant is graph["base"].constant


@pytest.mark.parametrize(
    "terminal_node", ["base", "downstream_view", "view_of_downstream_view"]
)
@pytest.mark.parametrize(
    "view_type",
    ["downstream_view", "view_of_downstream_view", "leaf_view", "view_of_leaf_view"],
)
def test_grad_is_view_of_base_grad(terminal_node: str, view_type: str):
    # The following graph is created:
    #
    # base = mg.Tensor([0., 1., 2., 3.], constant=base_constant)
    #
    # downstream_v = base[:2]
    # downstream_v_v = downstream_v[...]
    #
    # leaf_view = base[-2:]
    # view_of_leaf_view = leaf_view[...]
    #
    graph = create_view_graph()
    graph[terminal_node].backward()
    assert graph[view_type].grad.base is graph["base"].grad
    assert graph["base"].grad.base is None


@pytest.mark.parametrize(
    "terminal_node", ["base", "downstream_view", "view_of_downstream_view"]
)
@pytest.mark.parametrize(
    "resume_node",
    [
        "base",
        "downstream_view",
        "view_of_downstream_view",
        "leaf_view",
        "view_of_leaf_view",
    ],
)
@pytest.mark.parametrize(
    "via_inplace_op", [True, False],
)
def test_disconnected_views_dissassociate_from_base_upon_entering_new_graph(
    terminal_node: str, resume_node: str, via_inplace_op: bool
):
    # caught mem-lock state leak for:
    # - via_inplace_op: True
    # - view_type: downstream_view
    # - terminal_node: downstream_view
    graph = create_view_graph()
    graph[terminal_node].backward()

    # After backprop continue using one of the tensors from
    # the graph and ensure behavior is okay. At the very least
    # we don't want any internal errors to raise because of
    # inplace op weirdness
    t = graph[resume_node]
    if via_inplace_op:
        t += 0
    else:
        t = +t

    assert t.base is None
    assert t.grad is None

    if len(resume_node) <= len(terminal_node) and "leaf" not in resume_node:
        # Ha... this is so hacky, but it turns out that
        # the names get longer as you get further from base
        #
        # Abusing this to predict when there should be an
        # invalid backprop
        t.backward()
        assert_allclose(t.grad, np.ones_like(t))
    else:
        # calling backprop from a view downstream from a
        # disconnected view should raise
        with pytest.raises(InvalidBackprop):
            t.backward()

    if (
        via_inplace_op is True
        and terminal_node == "downstream_view"
        and resume_node == "downstream_view"
    ):
        # documented edge case for mem-guard state leakage
        clear_all_mem_locking_state()


@pytest.mark.parametrize("after_backprop", [True, False])
@given(base=tensors(elements=st.floats(-100, 100), read_only=st.booleans()))
def test_basic_view_relationships(base: mg.Tensor, after_backprop: bool):
    assert base.base is None

    leaf = +base

    view1 = base[...]
    view2 = base[...]
    view_of_view = view2[...]
    if after_backprop:
        (base + view1 + view2 + view_of_view).sum().backward()

    assert leaf.constant is base.constant
    assert view1.constant is base.constant
    assert view2.constant is base.constant
    assert view_of_view.constant is base.constant

    assert leaf.base is None
    assert view1.base is base
    assert view2.base is base
    assert view_of_view.base is base

    if after_backprop:
        # involving any disconnected view in an op
        # removes its base and clears its grad
        +base
        +view1
        +view2
        +view_of_view
        assert base.base is None and base.grad is None
        assert view1.base is None and view1.grad is None
        assert view2.base is None and view2.grad is None
        assert view_of_view.base is None and view_of_view.grad is None


@given(base=tensors(shape=(3, 3), elements=st.floats(-100, 100), constant=False))
def test_view_owns_grad_in_correspondence_with_base(base: mg.Tensor):
    assert base.base is None
    leaf = +base
    top_half_view = base[:2]
    bottom_half_view = base[-2:]
    dangling_view = base[...]
    transposed_view = base.T

    assert base.grad is None
    assert transposed_view.grad is None
    assert dangling_view.grad is None
    assert top_half_view.grad is None
    assert bottom_half_view.grad is None
    assert leaf.grad is None

    (2 * top_half_view + 3 * bottom_half_view).sum().backward()

    expected_base_grad = np.zeros((3, 3), dtype="float64")
    expected_base_grad[0] += 2
    expected_base_grad[1] += 5
    expected_base_grad[2] += 3

    assert_allclose(base.grad, expected_base_grad)
    assert_allclose(transposed_view.grad, base.grad.T)
    assert_allclose(dangling_view.grad, base.grad)
    assert_allclose(top_half_view.grad, base.grad[:2])
    assert_allclose(bottom_half_view.grad, base.grad[-2:])

    assert leaf.grad is None


@pytest.mark.parametrize("base_inplace", [True, False])
@pytest.mark.parametrize("view_inplace", [True, False])
def test_resuming_graph_after_backprop_through_view(
    base_inplace: bool, view_inplace: bool
):
    base = mg.arange(4.0)
    view = base[-2:]
    (view * view[...]).backward()

    expected_grad = np.zeros_like(base)
    expected_grad[-2:] = 2 * base.data[-2:]
    assert_allclose(base.grad, expected_grad)
    assert_allclose(view.grad, base.grad[-2:])

    if view_inplace:
        view *= 3
    else:
        view = 3 * view[...]
    assert view.base is None

    view.backward()
    assert_allclose(base, np.arange(4.0))
    assert_allclose(view, 3 * np.arange(4.0)[-2:])
    assert_allclose(base.grad, expected_grad)
    assert_allclose(view.grad, np.ones_like(view))

    if base_inplace:
        base *= 2
    else:
        base = 2 * base[...]
    assert base.base is None

    base.backward()
    assert_allclose(base, 2 * np.arange(4.0))
    assert_allclose(view, 3 * np.arange(4.0)[-2:])
    assert_allclose(base.grad, np.ones_like(base))
    assert_allclose(view.grad, np.ones_like(view))
