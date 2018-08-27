"""
Tests Tensor/Operation implementation that uses topological sorting against the naive implementation.

Here, a rule-based state machine constructs computational graphs via 'rules' by which nodes (tensors)
are 'fused' via addition and multiplication operations. The state machine create and fuse nodes in
any patterns, invoke `null_gradients` and `clear_graph` arbitrarily as well.

The values and gradients of the nodes in the mygrad and naive graphs must match as an invariant to
any permutation of test states (i.e. permutations of the aforementioned rules)"""

import hypothesis.strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, invariant, precondition
from numpy.testing import assert_equal, assert_almost_equal
from mygrad import Tensor, add, multiply

from pytest import raises

from .simple_graph import _add, _multiply, Node


def _node_ID_str(num): return "v{}".format(num + 1)


class GraphCompare(RuleBasedStateMachine):
    def __init__(self):
        super(GraphCompare, self).__init__()
        self.node_list = []
        self.str_to_tensor_op = {"add": add, "multiply": multiply}
        self.str_to_node_op = {"add": _add, "multiply": _multiply}
        self.raised = False
    nodes = Bundle('nodes')

    @rule(target=nodes, value=st.floats(-10, 10), constant=st.booleans())
    def create_node(self, value, constant):
        n = Node(value, constant=constant)
        t = Tensor(value, constant=constant)
        self.node_list.append((n, t))
        return n, t

    @rule(target=nodes, a=nodes, b=nodes, op=st.sampled_from(["add", "multiply"]), constant=st.booleans())
    def fuse_nodes(self, a, b, op, constant):
        n_a, t_a = a
        n_b, t_b = b
        n_op = self.str_to_node_op[op]
        t_op = self.str_to_tensor_op[op]
        out = (n_op(n_a, n_b, constant=constant), t_op(t_a, t_b, constant=constant))
        self.node_list.append(out)
        return out
    
    @rule(items=nodes, clear_graph=st.booleans())
    def null_gradients(self, items, clear_graph):
        n, t = items
        n.null_gradients(clear_graph=clear_graph)
        t.null_gradients(clear_graph=clear_graph)

    @rule(items=nodes)
    def clear_graph(self, items):
        n, t = items
        n.clear_graph()
        t.clear_graph()

    @rule(items=nodes, grad=st.floats(-10, 10))
    def backprop(self, items, grad):
        n, t = items
        n.null_gradients(clear_graph=False)
        t.null_gradients(clear_graph=False)
        try:
            n.backward(grad, terminal_node=True)
        except Exception:
            with raises(Exception):
                t.backward(grad)
            self.raised = True
        else:
            t.backward(grad)

    @precondition(lambda self: not self.raised)
    @invariant()
    def all_agree(self):
        for num, (n, t) in enumerate(self.node_list):
            assert bool(n._ops) is bool(t._ops), _node_ID_str(num)
            assert_equal(n.data, t.data, err_msg=_node_ID_str(num))
            if n.grad is None or t.grad is None:
                assert n.grad is t.grad, _node_ID_str(num)
            else:
                assert_almost_equal(desired=n.grad, actual=t.grad, err_msg=_node_ID_str(num))
            assert not t._accum_ops, _node_ID_str(num)


TestGraphComparison = GraphCompare.TestCase

