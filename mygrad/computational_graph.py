from graphviz import Digraph
from mygrad.tensor_base import Tensor

def build_graph(fin, names=None, render=True, save=False):
    """ Builds and renders a computational graph.

        Parameters
        ----------
        fin : Tensor
            The tensor object that will be the final node in the
            computational graph.

        names : Optional[Dict[str, Tensor]]
            A dictionary that maps names of Tensors to Tensor objects. If
            an argument is passed to names, the key name that maps to a Tensor
            included in the computational graph will be used as a label for the
            Tensor's node. If no argument is passed, the nodes on the
            computational graph will display the full Tensor.

            To use the names assigned in the local environment,
            pass `names=locals()` to the build_graph function.

            If different names are used from the local environment,
            the key must map to the exact Tensor object. A new Tensor or copy
            of the original Tensor should not be created as the value in the
            dictionary.

        render : bool, optional (default=True)
            If True, build_graph will return a graphviz Digraph object that,
            when called, will render the computational graph in a Jupyter
            notebook or the Jupyter Qt console. If False, nothing is returned.

        save : bool, optional (default=False)
            If True, build_graph will save a rendered computational graph to
            the current working directory as `computational_graph.pdf`.

        Returns
        -------
        graphviz.Digraph

        Notes
        -----
        build_graph requires that Graphviz is installed.
    """
    assert isinstance(fin, Tensor), "fin must be a Tensor"
    assert isinstance(names, dict)
    assert isinstance(render, bool)
    assert isinstance(save, bool)

    graph = Digraph()

    for out, op in fin._graph_dict.items():
        if op is not None:
            op_name = op.__repr__().rpartition(".")[-1].replace(" object at ", "\n")[:-1]
            graph.node(name=op_name, label=op_name.rpartition("\n")[0], style="filled", fillcolor="red")

            for var in op.variables:
                var_name = str(var.data) + "\n" + str(id(var))
                var_label = var_name.rpartition("\n")[0]
                if names is not None:
                    for key in names:
                        if id(names[key]) == id(var):
                            var_label = key

                graph.node(name=var_name, label=var_label)
                graph.edge(var_name, op_name)

            out_label = out.rpartition("\n")[0]
            if names is not None:
                out_id = out.rpartition("\n")[-1]
                for key in names:
                    if id(names[key]) == int(out_id):
                        out_label = key

            graph.node(name=out, label=out_label)
            graph.edge(op_name, out)

    if save:
        graph.render(filename="computational_graph", cleanup=True)

    if render:
        return graph
