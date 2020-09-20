from typing import Iterable

from hypothesis import given

from mygrad import Tensor
from tests.custom_strategies import tensors


def assert_list_of_tensors_match(list1: Iterable[Tensor], list2: Iterable[Tensor]):
    list1 = sorted(id(x) for x in list1)
    list2 = sorted(id(x) for x in list2)
    assert list1 == list2, "Lists of tensor-IDs do not match"


@given(base=tensors())
def test_view_children_tracking(base: Tensor):
    child = base[...]
    a = child.reshape((-1,))
    b = child[...]
    c = +child  # no view

    assert all(x.base is base for x in [child, a, b])
    assert c.base is None
    assert_list_of_tensors_match(base._view_children, [child])
    assert_list_of_tensors_match(child._view_children, [a, b])
