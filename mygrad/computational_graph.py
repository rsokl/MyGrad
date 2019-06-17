import numpy as np
from graphviz import Digraph

from mygrad.tensor_base import Tensor


def build_graph(
    fin,
    names=None,
    *,
    render=True,
    save=False,
    dims=False,
    dtypes=False,
    sum_stats=False
):
    """ Builds and renders a computational graph.

        Parameters
        ----------
        fin : mygrad.Tensor
            The tensor object that will be the final node in the
            computational graph.

        names : Optional[Dict[str, Union[mygrad.Tensor, numpy.ndarray]]]
            A dictionary that maps names of Tensors to Tensor objects. If
            an argument is passed to names, the key name that maps to a Tensor
            included in the computational graph will be used as a label for the
            Tensor's node. If no argument is passed, the nodes on the
            computational graph will display the full Tensor.

            To use the names assigned in the local environment,
            pass ``names=locals()`` to the build_graph function.

            If different names are used from the local environment,
            the key must map to the exact Tensor object. A new Tensor or copy
            of the original Tensor should not be created as the value in the
            dictionary.

            Only instances of mygrad.Tensor or numpy.ndarray can have labels
            assigned to Nodes. If a list or tuple is used in an operation
            with a Tensor, and names is not None, the Node label will be
            set to *Constant*. If a list or tuple is used in multiple operations,
            a unique Node will be created for each time it is used.

            A scalar will always be used as the label for a 0-dimensional
            Tensor's Node.

        render : bool, optional (default=True)
            If True, build_graph will return a graphviz Digraph object that,
            when called, will render the computational graph in a Jupyter
            notebook or the Jupyter Qt console. If False, nothing is returned.

        save : bool, optional (default=False)
            If True, build_graph will save a rendered computational graph to
            the current working directory as ``computational_graph.pdf``.

        dims : bool, optional (default=False)
            If True, Tensor dimensions are added to Node labels. Dimensions
            will not be displayed for scalar values.

        dtypes : bool, optional (default=False)
            If True, Tensor data types are added to Node labels.

        sum_stats : bool, optional (default=False)
            If True, Tensor minimums, maximums, medians, and means are
            added to Node labels. These will not be displayed for scalar values.

        Returns
        -------
        Union[graphviz.Digraph, None]

        Notes
        -----
        build_graph requires that Graphviz is installed.
    """
    assert isinstance(fin, Tensor), "fin must be a Tensor"
    assert isinstance(names, (dict, type(None)))
    assert isinstance(render, bool)
    assert isinstance(save, bool)
    assert isinstance(dims, bool)
    assert isinstance(dtypes, bool)
    assert isinstance(sum_stats, bool)

    graph = Digraph(strict=True)
    graph.node_attr.update(fontsize="12")

    _add_node(fin, graph, names=names, dims=dims, dtypes=dtypes, sum_stats=sum_stats)

    if save:
        graph.render(filename="computational_graph", cleanup=True)

    if render:
        return graph


def _add_node(node, graph, op_id=None, **kwargs):
    """ Recursively traces computational graph and adds nodes to Digraph. """
    node_id = str(id(node))
    node_lab = repr(node)
    if kwargs["names"] is not None:
        for key in kwargs["names"]:
            if id(kwargs["names"][key]) == id(node):
                node_lab = key
                break
            elif id(kwargs["names"][key]) == id(node.data):
                node_lab = key + "\n*Constant*"
                node_id = str(id(node.data))
                break
        if node_lab == repr(node):
            if not node.ndim:
                node_lab = str(node.data)
            elif node._constant:
                node_lab = "*Constant*"
            else:
                node_lab = "Intermediary Tensor"

    if node.ndim:
        if kwargs["dims"]:
            node_lab += "\nDims: {}".format(node.shape)
        if kwargs["dtypes"]:
            node_lab += "\nDtype: {}".format(node.dtype)
        if kwargs["sum_stats"]:
            node_lab += "\nMin: {min}\nMedian: {med}\nMean: {mean}\nMax: {max}".format(
                min=np.amin(node.data),
                med=np.median(node.data),
                mean=np.mean(node.data),
                max=np.amax(node.data),
            )
    else:
        if kwargs["dtypes"]:
            node_lab += "\nDtype: {}".format(node.dtype)

    graph.node(name=node_id, label=node_lab)

    if node._creator is None:
        if op_id is not None:
            graph.edge(node_id, op_id)
        return
    else:
        op_lab = repr(node._creator).rpartition(".")[-1].split(" ")[0]
        if op_id is not None:
            graph.edge(node_id, op_id)
        op_id = str(id(node._creator))

        graph.node(name=op_id, label=op_lab, style="filled", fillcolor="red")
        graph.edge(op_id, node_id)

        for var in node._creator.variables:
            _add_node(var, graph, op_id=op_id, **kwargs)
