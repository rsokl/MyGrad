import typing
from typing import Dict, Iterator, List, NamedTuple, Optional, Set, TypeVar

from mygrad._utils import WeakRefIterable

if typing.TYPE_CHECKING:
    from mygrad import Tensor


T = TypeVar("T")


def mirror_tensor(*, target: "Tensor", source: "Tensor"):
    """ *Dev use only*

    Points all of the attributes of ``self`` to those of
    ``tensor`` so that they reference all of the same data structures.
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
        original.grad is None
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
        """Recursively creates placeholders for all views downstream of `tensor`"""
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
