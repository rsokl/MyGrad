from typing import Callable, List, Tuple

import hypothesis.strategies as st
from hypothesis import given, settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule
from numpy.testing import assert_array_equal

import mygrad as mg
from mygrad import Tensor
from mygrad._utils.duplicating_graph import DuplicatingGraph, Node
from tests.custom_strategies import tensors
from tests.utils.stateful import clear_all_mem_locking_state


@given(x=tensors(read_only=st.booleans()))
def test_contains_method(x: Tensor):
    view = x[...]
    graph = DuplicatingGraph(x)
    assert (x in graph) is True
    assert (view in graph) is True
    assert (mg.asarray(x) in graph) is False


@given(x=tensors(read_only=st.booleans()))
def test_basic_duplicating_graph_info(x: Tensor):
    y = x[...]
    graph = DuplicatingGraph(x)
    _x = graph[x].placeholder
    _y = graph[y].placeholder

    assert_array_equal(x, _x)
    assert_array_equal(y, _y)

    assert x is not _x
    assert y is not _y

    assert graph[x].parent is None
    assert graph[y].parent is x

    assert x.base is None
    assert _x.base is None
    assert y.base is x
    assert _y.base is _x

    assert _y.creator.variables[0] is _x
    assert tuple(_x._view_children)[0] is _y

    node1, node2 = graph
    assert node1.tensor is x
    assert node2.tensor is y

    assert (node.tensor is t for node, t in zip(graph.get_path_to_base(y), [y, x]))


@given(x0=tensors(read_only=st.booleans()))
def test_path_to_base(x0: Tensor):
    _ = x0[...]
    x1 = x0[...]
    _ = x0[...]
    _ = x1[...]
    x2 = x1[...]
    _ = x2[...]

    graph = DuplicatingGraph(x0)
    for tensor, expected_path in zip([x2, x1, x0], [[x2, x1, x0], [x1, x0], [x0]]):
        assert all(
            actual.tensor is expected
            for actual, expected in zip(graph.get_path_to_base(tensor), expected_path)
        )


@given(base=tensors(read_only=st.booleans()))
def test_graph_iteration_is_depth_first(base: Tensor):
    a1 = base[...]
    b1 = base[...]
    a2 = a1[...]
    a3_1 = a2[...]
    a3_2 = a2[...]
    a4_2 = a3_2[...]

    b2 = b1[...]

    c1 = base[...]
    graph = DuplicatingGraph(base)
    expected_order = [base, a1, a2, a3_1, a3_2, a4_2, b1, b2, c1]

    for actual, expected in zip((i.tensor for i in graph), expected_order):
        assert actual is expected


@given(x0=tensors(read_only=st.booleans()), num_views=st.integers(0, 3))
def test_memory_locking(x0: Tensor, num_views: int):
    """
    Ensure graph created among placeholders locks and frees memory
    as expected
    """
    was_writeable = x0.data.flags.writeable
    x = x0
    for _ in range(num_views):
        x = x[...]

    graph = DuplicatingGraph(x0)
    nodes = tuple(n.placeholder for n in graph.get_path_to_base(x))

    # delete spurious references to ops involving placeholders
    del graph, x, x0

    for node in nodes:
        assert not node.data.flags.writeable or num_views == 0

    nodes[0].backward()

    for node in nodes:
        assert node.data.flags.writeable is was_writeable


def view(x):
    return x[...]


def flip(x):
    return x[::-1]


@settings(deadline=None)
class GraphDuplicationCompare(RuleBasedStateMachine):
    """Creates a random view graph from a single 'base' tensor and
    checks the logic of DuplicatingGraph"""

    def __init__(self):
        super().__init__()
        self.base = mg.arange(6.0)
        self.node_list = [id(self.base)]  # type: List[int]
        self.parent_child_pairs = set()

    nodes = Bundle("nodes")

    @rule(target=nodes, view_op=st.sampled_from([view, flip]))
    def create_view_of_base(self, view_op: Callable[[Tensor], Tensor]):
        out = view_op(self.base)
        self.node_list.append(id(out))
        self.parent_child_pairs.add((id(self.base), id(out)))
        return out

    @rule(target=nodes, parent=nodes, view_op=st.sampled_from([view, flip]))
    def create_view_of_node(self, parent: Tensor, view_op: Callable[[Tensor], Tensor]):
        out = view_op(parent)
        self.node_list.append(id(out))
        self.parent_child_pairs.add((id(parent), id(out)))
        return out

    @rule()
    def perform_non_view_op_of_base(self):
        2 * self.base  # this shouldn't affect the view graph

    @rule(parent=nodes)
    def perform_non_view_op_of_node(self, parent: Tensor):
        2 * parent  # this shouldn't affect the view graph

    def teardown(self):
        graph = DuplicatingGraph(self.base)
        iter_nodes: Tuple[Node, ...] = tuple(graph)

        assert sorted(id(t.tensor) for t in iter_nodes) == sorted(self.node_list)

        # test that iteration obeys topological ordering
        visited = {id(graph.base.tensor)}
        for prev_node, node in zip(iter_nodes[:-1], iter_nodes[1:]):
            assert (node.parent is prev_node.tensor) or (id(node.parent) in visited)
            visited.add(id(node.tensor))

        for node in iter_nodes:
            assert_array_equal(
                node.tensor,
                node.placeholder,
                err_msg="tensor and placeholder have distinct data",
            )

            assert (
                node.tensor is not node.placeholder
            ), "tensor and placeholder are not distinct objects"
            if node.parent is not None:
                assert (
                    id(node.parent),
                    id(node.tensor),
                ) in self.parent_child_pairs, "the recorded parent is erroneous: \n"
            else:
                assert (
                    id(node.parent),
                    id(node.tensor),
                ) not in self.parent_child_pairs, "a parent should have been recorded"

            # check that respective tensor/placeholder base references are accurate
            if node.tensor.base is None:
                assert node.placeholder.base is None
            else:
                assert node.tensor.base is graph.base.tensor
                assert node.placeholder.base is graph.base.placeholder

            for t in node.placeholder._view_children:
                assert (
                    t.creator.variables[0] is node.placeholder
                ), "graph was not redirected consistently"

        clear_all_mem_locking_state()  # don't leak state into other tests


TestGraphComparison = GraphDuplicationCompare.TestCase
