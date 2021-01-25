from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    TypeVar,
)

from numpy import ndarray

from mygrad._utils import WeakRefIterable
from mygrad.operation_base import BroadcastableOp, Operation

if TYPE_CHECKING:  # pragma: no cover
    from mygrad import Tensor


T = TypeVar("T")


def mirror_tensor(*, target: "Tensor", source: "Tensor"):
    """*Dev use only*

    Creates a shallow copy of attribute dictionary of ``self`` and assigns
    it to``tensor``, so that ``tensor`` has the same state as ``self`` and
    points to the same array data.

    This is used to facilitate "in-place" operations.
    """
    target.__dict__ = source.__dict__.copy()


def reroute_ops_through(*, target: "Tensor", source: "Tensor"):
    for op in source._ops:
        op = op()
        if op is None:
            continue

        op.variables = tuple(
            var_ if var_ is not source else target for var_ in op.variables
        )


def make_placeholder_tensor(
    original: "Tensor", *, base: Optional["Tensor"] = None
) -> "Tensor":
    """
    Creates a tensor that stands in the place of `original` in the computational graph.
    This does not create a copy of the array-data held by original; the mirrored tensor
    points to the same data.

    The resulting tensor should never be exposed to the user; it is used to accommodate
    in-place operations.

    Parameters
    ----------
    original : bool
        If True the placeholder holds a copy of the original data

    base : Optional[Tensor]
        Points the placeholder to the base tensor.

    Returns
    -------
    placeholder : Tensor
    """
    assert (
        original._grad is None
    ), "A placeholder copy can not be created for a tensor with a gradient"

    assert (
        not original._accum_ops
    ), "A placeholder copy cannot be created during backprop"

    placeholder = type(original)([])
    mirror_tensor(target=placeholder, source=original)
    placeholder._base = base
    # point all ops involving `self` to old_tensor instead
    reroute_ops_through(target=placeholder, source=original)
    return placeholder


class Node(NamedTuple):
    tensor: "Tensor"
    placeholder: "Tensor"
    # `tensor` is a view of `parent`
    parent: Optional["Tensor"] = None


class DuplicatingGraph:
    """Traces through the graph of all views of the base tensor and
    creates a corresponding graph of 'placeholders', used to permit
    a future in-place operation without mutating the current tensor
    graph.

    Provides the information needed to recreate a view-graph after
    an in-place operation has been performed on the base tensor.

    Upon initialization, this class mutates the graph downstream of the
    base tensor.
    """

    def _duplicate_graph(self, tensor: "Tensor"):
        """Recursively creates placeholders for all views downstream of `tensor`.

        Upon completion, the existing computational graph involves only placeholders.
        Note that the placeholders and original tensors point to the same array-data."""
        if not tensor._view_children:
            self.leafs.add(id(tensor))
            return

        for child in tensor._view_children:

            self._record_mapping(
                original=child,
                placeholder=make_placeholder_tensor(
                    original=child, base=self.base.placeholder
                ),
                parent=tensor,
            )

            self._duplicate_graph(child)

        self[tensor].placeholder._view_children = WeakRefIterable(
            [self[t].placeholder for t in tensor._view_children]
        )

    def __init__(self, base: "Tensor"):
        self.mappings: Dict[int, Node] = {}

        self._record_mapping(
            original=base, placeholder=make_placeholder_tensor(base, base=base.base)
        )
        self.base = self[base]

        self.leafs: Set[int] = set()
        # creates placeholders for each node in the view graph
        self._duplicate_graph(base)

    def __getitem__(self, item: "Tensor") -> Node:
        """Returns a node associated with a tensor"""
        return self.mappings[id(item)]

    def _record_mapping(
        self,
        original: "Tensor",
        placeholder: "Tensor",
        parent: Optional["Tensor"] = None,
    ):
        """
        Parameters
        ----------
        original : Tensor
            A tensor that will be involved in a mutated graph

        placeholder : Tensor
            Takes the place of the original in the computational graph

        parent : Optional[Tensor]
            The tensor of which ``original`` is a direct view
        """
        node = Node(tensor=original, placeholder=placeholder, parent=parent)
        self.mappings[id(node.tensor)] = node
        self.mappings[id(node.placeholder)] = node

    def _yield_children(self, tensor: "Tensor") -> Iterator[Node]:
        """Recursive helper function for DFS iteration"""
        yield self[tensor]
        for child in tensor._view_children:
            yield from self._yield_children(child)

    def __contains__(self, item):
        return id(item) in self.mappings

    def get_placeholder_if_exists(self, tensor: T) -> T:
        if tensor in self:
            return self[tensor].placeholder
        else:
            return tensor

    def __iter__(self) -> Iterator[Node]:
        """Returns all nodes in graph using DFS.

        Note that each node can only have one input edge, so no visitor
        information need be recorded

        We iterate based off of the placeholders' graph information
        since they will never be mutated."""
        yield from self._yield_children(self.base.placeholder)

    def get_path_to_base(self, tensor: "Tensor") -> List[Node]:
        """ Returns [leaf, (parent), ..., base]"""
        path = []
        node = self[tensor]
        while node.parent is not None:
            path.append(node)
            node = self[node.parent]
        path.append(self.base)
        return path

    def restore_old_graph(self):
        """ Reroute graph back to original tensors."""
        # call tuple to ensure iteration is completed
        # before information gets deleted / mutated
        for node in tuple(self):
            reroute_ops_through(target=node.tensor, source=node.placeholder)
            if node.placeholder._base is not None:
                node.tensor._base = self.base.tensor


class UnView(BroadcastableOp):
    """
    Creates an operation that connects a mutant base to
    the placeholder mutant-view and placeholder base that
    it is derived from.

    This effectively connects the mutant base to the upstream
    computational graph.
    """

    def __call__(
        self,
        placeholder_base: "Tensor",
        placeholder_mutant_view: "Tensor",
        mutant_base_data: ndarray,
        view_fn_sequence: Sequence[Callable[[ndarray], ndarray]],
    ):
        """
        Parameters
        ----------
        placeholder_mutant_view: Tensor
            The internal tensor that resulted from the in-place
            operation.

        placeholder_base: Tensor
            The placeholder for the base tensor involved in the
            in-place operation.

        mutant_base_data: ndarray
            The base tensor that was mutated by the in-place operation,
            and that will be exposed to the user.

        view_fn_sequence: Sequence[Callable[[ndarray], ndarray]]
            The sequence of view-functions used to create the
            view-tensor from the base

        Returns
        -------
        mutant_base_data : ndarray
            The array associated with the mutant base
        """
        self.variables = (placeholder_base, placeholder_mutant_view)
        self._view_fn_seq = view_fn_sequence
        return mutant_base_data

    def backward_var(self, grad: ndarray, index: int, **kwargs) -> ndarray:
        placeholder_base, placeholder_mutant_view = self.variables

        # Backprop through upstream base by zeroing out
        # all regions of `grad` associated with the downstream
        # view.
        #
        # Backprop through upstream view by taking the corresponding
        # view of the gradient
        #
        # E.g.
        #
        # base -> [1., 2., 3.]
        # view = base[:2] -> [1., 2.]
        # dℒ/d(out) = [g0, g1, g2]
        #
        # dℒ/d(base) = [0., 0., g2]
        # dℒ/d(view) = [g0, g1]
        if index == 0:  # compute dℒ/d(base)
            grad = grad.copy()
            grad_view = grad
            for fn in self._view_fn_seq:
                grad_view = fn(grad_view)

            assert grad_view.shape == self.variables[1].shape
            # check that grad_view shares memory with grad
            assert grad_view.base is grad
            grad_view *= 0

            return grad

        elif index == 1:  # compute dℒ/d(view)
            grad_view = grad
            for fn in self._view_fn_seq:
                grad_view = fn(grad_view)

            assert grad_view.shape == self.variables[1].shape
            return grad_view

        else:  # pragma: no cover
            raise ValueError(f"UnView: backward_var index: {index}")
