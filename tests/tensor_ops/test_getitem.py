from itertools import groupby
from operator import itemgetter

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import numpy as np
from hypothesis import given, settings
from mygrad.tensor_base import Tensor
from numpy.testing import assert_allclose

from ..custom_strategies import adv_integer_index, basic_indices
from ..wrappers.uber import backprop_test_factory, fwdprop_test_factory


def test_getitem():
    x = Tensor([1, 2, 3])
    a, b, c = x
    f = 2 * a + 3 * b + 4 * c
    f.backward()

    assert a.data == 1
    assert b.data == 2
    assert c.data == 3
    assert f.data == 20

    assert_allclose(a.grad, np.array(2))
    assert_allclose(b.grad, np.array(3))
    assert_allclose(c.grad, np.array(4))
    assert_allclose(x.grad, np.array([2, 3, 4]))


def get_item(*arrs, index, constant=False):
    o = arrs[0][index]
    if isinstance(o, Tensor):
        o._constant = constant
    return o


def basic_index_wrap(*arrs):
    return basic_indices(arrs[0].shape)


def adv_index_int_wrap(*arrs):
    return adv_integer_index(arrs[0].shape)


def adv_index_bool_wrap(*arrs):
    return hnp.arrays(shape=arrs[0].shape, dtype=bool)


@fwdprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
    kwargs=dict(index=basic_index_wrap),
)
def test_getitem_basicindex_fwdprop():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
    kwargs=dict(index=basic_index_wrap),
    vary_each_element=True,
)
def test_getitem_basicindex_bkwdprop():
    pass


@fwdprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
    kwargs=dict(index=adv_index_int_wrap),
)
def test_getitem_advindex_int_fwdprop():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
    kwargs=dict(index=adv_index_int_wrap),
    vary_each_element=True,
)
def test_getitem_advindex_int_bkwdprop():
    pass


@fwdprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
    kwargs=dict(index=adv_index_bool_wrap),
)
def test_getitem_advindex_bool_fwdprop():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: hnp.array_shapes(max_side=6, max_dims=4)},
    kwargs=dict(index=adv_index_bool_wrap),
    vary_each_element=True,
)
def test_getitem_advindex_bool_bkwdprop():
    pass


# test broadcast-compatible int-arrays
rows = np.array([0, 3], dtype=np.intp)
columns = np.array([0, 2], dtype=np.intp)


@fwdprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (4, 3)},
    kwargs=dict(index=np.ix_(rows, columns)),
)
def test_getitem_broadcast_index_fwdprop():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (4, 3)},
    kwargs=dict(index=np.ix_(rows, columns)),
    vary_each_element=True,
)
def test_getitem_broadcast_index_bkprop():
    pass


@fwdprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (3, 2, 4, 3)},
    kwargs=dict(index=(Ellipsis, 2, 0)),
)
def test_getitem_ellipsis_index_fwdprop():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (3, 2, 4, 3)},
    kwargs=dict(index=(Ellipsis, 2, 0)),
    vary_each_element=True,
)
def test_getitem_ellipsis_index_bkprop():
    pass


rows1 = np.array([False, True, False, True])
columns1 = [0, 2]


@fwdprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (4, 3)},
    kwargs=dict(index=np.ix_(rows1, columns1)),
)
def test_getitem_bool_int_fwdprop():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (4, 3)},
    kwargs=dict(index=np.ix_(rows1, columns1)),
    vary_each_element=True,
)
def test_getitem_bool_int_bkprop():
    pass


@fwdprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (4, 3)},
    kwargs=dict(index=(slice(1, 2), [1, 2])),
)
def test_getitem_basic_w_adv_fwdprop():
    pass


@settings(deadline=None)
@backprop_test_factory(
    mygrad_func=get_item,
    true_func=get_item,
    num_arrays=1,
    index_to_arr_shapes={0: (4, 3)},
    kwargs=dict(index=(slice(1, 2), [1, 2])),
    vary_each_element=True,
)
def test_getitem_basic_w_adv_bkprop():
    pass


@st.composite
def arbitrary_indices(draw, shape):
    # given a list of integers, find continuous sequences
    # e.g. [1, 3, 4, 5, 7, 8] -> [(1,), (3, 4, 5), (7, 8)]
    def group_continuous_integers(ls):
        return [
            tuple(map(itemgetter(1), g))
            for k, g in groupby(enumerate(ls), lambda x: x[0] - x[1])
        ]

    shape_inds = list(range(len(shape)))
    index = []  # stores tuples of (dim, index)

    # add integers, slices
    basic_inds = sorted(draw(st.lists(st.sampled_from(shape_inds), unique=True)))

    if len(basic_inds) > 0:
        basic_dims = tuple(shape[i] for i in basic_inds)

        # only draw ints and slices
        # will handle possible ellipsis and newaxis objects later
        # as these can make boolean indices difficult to handle later
        basics = draw(hnp.basic_indices(shape=basic_dims, allow_ellipsis=False))
        if not isinstance(basics, tuple):
            basics = (basics,)

        # will not necessarily index all dims from basic_inds as
        # `basic_indices` can return indices with omitted trailing slices
        index += [tup for tup in zip(basic_inds, basics)]

        for i in basic_inds[: len(basics)]:
            shape_inds.pop(shape_inds.index(i))

    if len(shape_inds) > 0:
        # add integer arrays to index
        int_arr_inds = sorted(draw(st.lists(st.sampled_from(shape_inds), unique=True)))

        if len(int_arr_inds) > 0:
            int_arr_dims = tuple(shape[i] for i in int_arr_inds)
            int_arrs = draw(hnp.integer_array_indices(shape=int_arr_dims))
            index += [tup for tup in zip(int_arr_inds, int_arrs)]

            for i in int_arr_inds:
                shape_inds.pop(shape_inds.index(i))

        if len(shape_inds) > 0:
            # add boolean arrays to index
            bool_inds = sorted(draw(st.lists(st.sampled_from(shape_inds), unique=True)))

            if len(bool_inds) > 0:
                grouped_bool_inds = group_continuous_integers(bool_inds)
                bool_dims = [tuple(shape[i] for i in ind) for ind in grouped_bool_inds]

                # if multiple boolean array indices, the number of trues must be the same
                # such that the output of ind.nonzero() for each index are broadcast compatible
                # this must also be the same as the trailing index of each integer array, if any used
                if len(int_arr_inds):
                    max_trues = max(i.shape[-1] for i in int_arrs)
                else:
                    max_trues = st.integers(
                        min_value=0, max_value=min(bool_dims, key=lambda x: np.prod(x))
                    )

                index += [
                    (
                        i[0],
                        draw(
                            hnp.arrays(shape=sh, dtype=bool).filter(
                                lambda x: x.sum() in (1, max_trues)
                            )
                        ),
                    )
                    for i, sh in zip(grouped_bool_inds, bool_dims)
                ]

                for i in bool_inds:
                    shape_inds.pop(shape_inds.index(i))

    grouped_shape_inds = group_continuous_integers(sorted(shape_inds))
    if len(grouped_shape_inds) == 1:
        # unused indices form a continuous stretch of dimensions
        # so can replace with an ellipsis

        # to test ellipsis vs omitted slices
        # randomly select if the unused indices are trailing
        if max(shape_inds) + 1 == len(shape):
            if draw(st.booleans()):
                index += [(min(shape_inds), Ellipsis)]
        else:
            index += [(min(shape_inds), Ellipsis)]

    else:
        # so that current chosen index's work,
        # fill in remaining any gaps with empty slices
        index += [(i, slice(None)) for i in shape_inds]

    index = sorted(index, key=lambda x: x[0])
    out_ind = tuple()

    for i in index:
        out_ind += (i[1],)

    # can now randomly add in newaxis objects
    newaxis_pos = sorted(
        draw(st.lists(st.integers(min_value=0, max_value=len(index)), unique=True)),
        reverse=True,
    )
    for i in newaxis_pos:
        index.insert(i, np.newaxis)

    return out_ind


@settings(deadline=None)
@given(
    a=hnp.arrays(
        shape=hnp.array_shapes(max_side=4, min_dims=1, max_dims=10),
        dtype=float,
        elements=st.floats(-10, 10),
    ),
    data=st.data(),
)
def test_arbitrary_indices_strategy(a, data):
    shape = a.shape
    index = data.draw(arbitrary_indices(shape))

    # if index does not comply with numpy indexing
    # rules, numpy will raise an error
    a[index]
