import mygrad as mg
from mygrad.math.arithmetic.ops import Add, Power, Divide


def test_op_tracks_graph():
    """Ensures that ``Operation.graph`` tracks operations as expected"""
    x = mg.Tensor(1)
    y = mg.Tensor(2)

    z = x * y
    assert z.creator.graph == {z.creator}

    f = z + 2
    assert f.creator.graph == {z.creator, f.creator}

    h = z - f
    assert h.creator.graph == {h.creator} | f.creator.graph

    i = ((h + 3) ** 2) / 5
    assert h.creator.graph < i.creator.graph
    assert len(i.creator.graph - h.creator.graph) == 3, "should be {Add, Subtract, Divide}"
    assert all(isinstance(x, (Add, Power, Divide)) for x in i.creator.graph - h.creator.graph)
