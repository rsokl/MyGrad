import hypothesis.strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, rule, precondition
from numpy.testing import assert_equal
from mygrad import Tensor, add, multiply

from .simple_graph import _add, _multiply, Node


class GraphCompare(RuleBasedStateMachine):
    def __init__(self):
        super(GraphCompare, self).__init__()
        self.node_list = []
        self.str_to_tensor_op = {"add": add, "multiply": multiply}
        self.str_to_node_op = {"add": _add, "multiply": _multiply}

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

    @rule(items=nodes)
    def graph_states_agree(self, items):
        n, t = items
        assert bool(n._ops) is bool(t._ops)

    @rule(items=nodes)
    def values_agree(self, items):
        n, t = items
        assert_equal(n.data, t.data)

    @rule(items=nodes)
    def grads_agree(self, items):
        n, t = items
        if n.grad is None:
            assert t.grad is None
        else:
            assert_equal(n.grad, t.grad)


TestDBComparison = GraphCompare.TestCase
