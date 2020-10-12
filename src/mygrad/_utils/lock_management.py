"""
Provides utilities responsible for locking/releasing array writeability.
"""
from collections import Counter
from typing import TYPE_CHECKING, Dict, Generator, Iterable
from weakref import ref

import numpy as np

from mygrad._utils import WeakRefIterable

if TYPE_CHECKING:
    from mygrad import Tensor
    from mygrad._utils import WeakRef

# arr-id -> num active ops involving arr
_array_counter = Counter()

# arr-id -> weak-ref of arr, for arrays participating in live ops
_array_tracker = dict()  # type: Dict[int, WeakRef[Tensor]]

# maps base-array ID to ID of view that can't be unlocked until
# base is unlocked
_views_waiting_for_unlock = dict()  # type: Dict[int, int] # base-id -> view-id

__all__ = [
    "lock_arr_writeability",
    "release_writeability_lock_on_op",
]


def array_is_tracked(arr: np.ndarray) -> bool:
    """Returns True if the provided array, or a view of it, is currently
    involved in one or more mygrad operation."""
    arr_id = id(arr)
    return arr_id in _array_tracker and _array_tracker[arr_id]() is not None


def lock_arr_writeability(arr: np.ndarray, force_lock: bool = False) -> np.ndarray:
    """Increments the count of active ops that an array is involved in
    and makes the array read-only

    Parameters
    ----------
    arr : numpy.ndarray

    force_lock : bool, optional (default=False)
        If True, and array that is already read-only will be tracked
        for unlocking

    Returns
    -------
    numpy.ndarray
        The locked array"""
    arr_id = id(arr)
    if not array_is_tracked(arr):
        if not force_lock and not arr.flags.writeable:
            # array is natively read-only; don't do anything
            return arr
        # keeps track of array so we can clean up the array
        # counter when tracked arrays fall out of scope
        _array_tracker[arr_id] = ref(arr)
        _array_counter[arr_id] = 1
    else:
        _array_counter[arr_id] += 1
    if arr.flags.writeable is True:
        arr.flags.writeable = False
    return arr


def unique_arrs_and_bases(
    tensors: Iterable["Tensor"],
) -> Generator[np.ndarray, None, None]:
    """
    Yields unique (by-ID) arrays from an iterable. If an array
    has a base, the base is yielded first (assuming that base
    object has not already been yielded).
    """
    seen = set()
    for t in tensors:
        arr = t.data
        arr_id = id(arr)
        if arr_id not in seen:

            # important note!
            # We must yield array bases first so that base's
            # writeability is restored first.
            # Then view's writeability can be restored
            if arr.base is not None:
                base_id = id(arr.base)
                if base_id not in seen:
                    seen.add(base_id)
                    yield arr.base
            seen.add(arr_id)
            yield arr


def _release_lock_on_arr_writeability(arr: np.ndarray):
    arr_id = id(arr)
    num_active_ops = _array_counter[arr_id]

    if num_active_ops == 1:
        # final active op involving array is being de-referenced:
        # okay to unlock array
        del _array_counter[arr_id]

        if arr.base is not None and arr.base.flags.writeable is False:
            # Array is view and must wait until its base is released
            # before it can be unlocked
            # Thus we are still tracking this array
            _views_waiting_for_unlock[id(arr.base)] = arr_id
        else:
            # we no longer need to track the array
            arr.flags.writeable = True
            _array_tracker.pop(arr_id, None)
    elif num_active_ops > 0:
        _array_counter[arr_id] = num_active_ops - 1

    if (
        arr.base is None
        and arr.flags.writeable
        and (arr_id in _views_waiting_for_unlock)
    ):
        # array was base of view waiting to be unlocked..
        #
        # Either:
        #    view no longer exists
        #    or view is involved in new op
        #    or view can now get unlocked
        # under all conditions view will no longer be waiting to be unlocked
        view_arr_id = _views_waiting_for_unlock.pop(arr_id)

        if _array_counter[view_arr_id] > 0:
            # view involved in new op
            return

        try:
            view_arr = _array_tracker.pop(view_arr_id)()
            if view_arr is None:
                return
        except KeyError:
            # view array is no longer available for unlocking
            return

        view_arr.flags.writeable = True


def release_writeability_lock_on_op(arr_refs: WeakRefIterable[np.ndarray]):
    """Marks each array (and for a view, its base) to have its
    writeability lock released.

    An array is made writeable only once all of its locks
    have been released.

    Parameters
    ----------
    arr_refs : WeakRefIterable[np.ndarray]
        The arrays to be unlocked. Only one lock is released
        on each array, even if the same array occurs
        multiple times in the iterable."""
    for arr in arr_refs:
        _release_lock_on_arr_writeability(arr)
