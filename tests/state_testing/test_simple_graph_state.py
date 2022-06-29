"""
Tests Tensor/Operation implementation that uses topological sorting against the naive implementation.

Here, a rule-based state machine constructs computational graphs via 'rules' by which nodes (tensors)
are 'fused' via addition and multiplication operations. The state machine create and fuse nodes in
any patterns, invoke `null_gradients` and `clear_graph` arbitrarily as well.

The values and gradients of the nodes in the mygrad and naive graphs must match as an invariant to
any permutation of test states (i.e. permutations of the aforementioned rules)"""

from typing import List, Tuple

import hypothesis.strategies as st
from hypothesis import assume, settings
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule
from numpy.testing import assert_allclose, assert_equal
from pytest import raises

from mygrad import Tensor, add, multiply
from mygrad.errors import InvalidBackprop
from tests.utils.stateful import clear_all_mem_locking_state

from .simple_graph import Node, _add, _multiply


def _node_ID_str(num):
    return "v{}".format(num + 1)


@settings(max_examples=125, stateful_step_count=100, deadline=None)
class GraphCompare(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        # stores the corresponding node/tensor v1, v2, ... as they are
        # created via the unit test (through `create_node` or `fuse_nodes`)
        # `Node` is the naive implementation of `Tensor` that we are checking
        # against
        self.node_list = []  # type: List[Tuple[Node, Tensor]]
        self.str_to_tensor_op = {"add": add, "multiply": multiply}
        self.str_to_node_op = {"add": _add, "multiply": _multiply}
        self.raised = False

    nodes = Bundle("nodes")

    @rule(target=nodes, value=st.floats(-10, 10), constant=st.booleans())
    def create_node(self, value, constant):
        n = Node(value, constant=constant)
        t = Tensor(value, constant=constant)
        self.node_list.append((n, t))
        return n, t

    @rule(
        target=nodes,
        a=nodes,
        b=nodes,
        op=st.sampled_from(["add", "multiply"]),
        constant=st.sampled_from([True, None]),
    )
    def fuse_nodes(self, a, b, op, constant):
        """
        Combine any pair of nodes (tensors) using either addition or multiplication, producing
        a new node (tensor)"""
        n_a, t_a = a  # type: Node, Tensor
        n_b, t_b = b  # type: Node, Tensor
        n_op = self.str_to_node_op[op]
        t_op = self.str_to_tensor_op[op]
        out = (
            n_op(n_a, n_b, constant=bool(constant)),
            t_op(t_a, t_b, constant=constant),
        )
        self.node_list.append(out)
        return out

    @rule(items=nodes)
    def clear_graph(self, items):
        """
        Invoke `clear_graph` on the computational graph (naive and mygrad)
        """
        n, t = items  # type: Node, Tensor
        n.clear_graph()
        t.clear_graph()

    @rule(items=nodes, grad=st.floats(-10, 10))
    def backprop(self, items, grad):
        """
        Invoke `backward(grad)` on the computational graph (naive and mygrad) from a randomly-selected
        node in the computational graph and using a randomly-generated gradient value.

        An exception should be raised if `clear_graph` is invoked anywhere prior to the invoking node.
        """
        n, t = items  # type: Node, Tensor
        try:
            n.backward(grad, terminal_node=True)
        except InvalidBackprop:
            with raises(InvalidBackprop):
                t.backward(grad)
            self.raised = True
            assume(False)
        else:
            t.backward(grad)
            assert not t._ops
            assert not t.creator

    @invariant()
    def all_agree(self):
        """
        Ensure that all corresponding nodes/tensors have matching data and gradients
        across the respective graphs.
        """
        assert not self.raised, (
            "there is a problem with the state machine: "
            "invalid-backprop should be marked as 'assumed false' "
            "by Hypothesis"
        )
        for num, (n, t) in enumerate(self.node_list):
            assert bool(n._ops) is bool(t._ops), _node_ID_str(num)
            assert_equal(n.data, t.data, err_msg=_node_ID_str(num))
            if n.grad is None or t.grad is None:
                assert n.grad is t.grad, _node_ID_str(num)
            else:
                assert_allclose(
                    actual=t.grad,
                    desired=n.grad,
                    atol=1e-5,
                    rtol=1e-5,
                    err_msg=_node_ID_str(num),
                )

    def teardown(self):
        clear_all_mem_locking_state()


TestGraphComparison = GraphCompare.TestCase
