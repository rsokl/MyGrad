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
            op_name = op.__repr__().rpartition(".")[-1].replace(" object at ", "\n")[:-1]
            op_node = pydot.Node(name=op_name, label=op_name.rpartition("\n")[0], style="filled", fillcolor="red")
            graph.add_node(op_node)

            for var in op.variables:
                var_name = str(var.data) + "\n" + str(id(var))
                var_node = pydot.Node(name=var_name, label=var_name.rpartition("\n")[0])
                graph.add_node(var_node)
                graph.add_edge(pydot.Edge(var_node, op_node))

            out_label = out.rpartition("\n")[0]
            out_node = pydot.Node(name=out, label=out_label)
            graph.add_node(out_node)
            graph.add_edge(pydot.Edge(op_node, out_node))

    graph.write_png('computational_graph.png')
