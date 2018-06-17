import pydot


def build_graph(fin):
    """ Builds and saves a network graph as a PNG file.

        Parameters
        ----------
        fin : Tensor
            The tensor object that should be the final node in the
            computational graph.

        Notes
        -----
        build_graph requires that Graphviz and pydot are installed.

        The PNG file will be written to the current working directory.
    """
    graph = pydot.Dot(graph_type='digraph')

    for out, op in fin._graph_dict.items():
        if op is not None:
            op_node = pydot.Node(op.__repr__().rpartition(".")[-1].replace(" object at ", "\n")[:-1], style="filled", fillcolor="red")
            graph.add_node(op_node)
            for var in op.variables:
                graph.add_edge(pydot.Edge(str(var.data) + "\n" + str(id(var)), op_node))
            graph.add_edge(pydot.Edge(op_node, out))

    graph.write_png('network_graph.png')
