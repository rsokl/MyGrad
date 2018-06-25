import pydot
from mygrad.tensor_base import Tensor

def build_graph(fin, names=None):
    """ Builds and saves a computational graph as a PNG file.

        Parameters
        ----------
        fin : Tensor
            The tensor object that will be the final node in the
            computational graph.

        names : Optional[Dict[str, Tensor]]
            A dictionary that maps names of Tensors to Tensor objects. If
            an argument is passed to names, the key name that maps to a Tensor
            included in the computational graph will be used as a label for the
            Tensor's node.

            To use the names assigned in the local environment,
            pass `names=locals()` to the build_graph function.

            If different names are used from the local environment,
            the key must map to the exact Tensor object. A new Tensor or copy
            of the original Tensor should not be created as the value in the
            dictionary.


        Notes
        -----
        build_graph requires that Graphviz and pydot are installed.

        The PNG file will be written to the current working directory.
    """
    assert isinstance(fin, Tensor), "fin must be a Tensor"
    assert isinstance(names, dict), "names must be a dictionary"

    graph = pydot.Dot(graph_type='digraph')

    for out, op in fin._graph_dict.items():
        if op is not None:
            op_name = op.__repr__().rpartition(".")[-1].replace(" object at ", "\n")[:-1]
            op_node = pydot.Node(name=op_name, label=op_name.rpartition("\n")[0], style="filled", fillcolor="red")
            graph.add_node(op_node)

            for var in op.variables:
                var_name = str(var.data) + "\n" + str(id(var))
                var_label = var_name.rpartition("\n")[0]
                if names is not None:
                    for key in names:
                        if id(names[key]) == id(var):
                            var_label = key

                var_node = pydot.Node(name=var_name, label=var_label)
                graph.add_node(var_node)
                graph.add_edge(pydot.Edge(var_node, op_node))

            out_label = out.rpartition("\n")[0]
            if names is not None:
                out_id = out.rpartition("\n")[-1]
                for key in names:
                    if id(names[key]) == int(out_id):
                        out_label = key

            out_node = pydot.Node(name=out, label=out_label)
            graph.add_node(out_node)
            graph.add_edge(pydot.Edge(op_node, out_node))

    graph.write_png('computational_graph.png')
