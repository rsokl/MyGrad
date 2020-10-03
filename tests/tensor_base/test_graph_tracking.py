import hypothesis.strategies as st
from hypothesis import given

import mygrad as mg
from mygrad.math.arithmetic.ops import Add, Divide, Power


@given(x_constant=st.booleans(), y_constant=st.booleans())
def test_op_tracks_graph(x_constant: bool, y_constant: bool):
    """Ensures that ``Operation.graph`` tracks operations as expected"""
    x = mg.Tensor(1, constant=x_constant)
    y = mg.Tensor(2, constant=y_constant)

    z = x * y
    assert set(z.creator.graph) == {z.creator}

    f = z + 2
    assert set(f.creator.graph) == {f.creator} | (
        {z.creator} if not z.constant else set()
    )

    h = z - f
    assert set(h.creator.graph) == {h.creator} | (
        set(f.creator.graph) if not f.constant else set()
    )

    i = ((h + 3) ** 4) / 5
    if i.constant:
        assert set(i.creator.graph) == {i.creator}
    else:
        assert h.creator.graph < i.creator.graph
        assert (
            len(i.creator.graph - h.creator.graph) == 3
        ), "should be {{Add, Subtract, Divide}}, but got {}".format(
            i.creator.graph - h.creator.graph
        )
        assert all(
            isinstance(x, (Add, Power, Divide))
            for x in i.creator.graph - h.creator.graph
        )
