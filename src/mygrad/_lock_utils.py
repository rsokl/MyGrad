"""
Provides utilities responsible for locking/releasing array writeability.
"""
from collections import Counter
from typing import Generator, Iterable
from weakref import WeakValueDictionary

import numpy as np

from mygrad._utils import WeakRefIterable

_array_counter = Counter()
_array_tracker = WeakValueDictionary()
_views_waiting_for_unlock = WeakValueDictionary()

__all__ = [
    "lock_array_and_base_writeability",
    "lock_arr_writeability",
    "release_writeability_lock_on_op",
]


def lock_arr_writeability(arr: np.ndarray, force_lock: bool = False):
    arr_id = id(arr)
    if arr_id not in _array_tracker:
        if not force_lock and not arr.flags.writeable:
            # array is natively read-only; don't do anything
            return
        # keeps track of array so we can clean up the array
        # counter when tracked arrays fall out of scope
        _array_tracker[arr_id] = arr
        _array_counter[arr_id] = 1
    else:
        _array_counter[arr_id] += 1
    if arr.flags.writeable is True:
        arr.flags.writeable = False


def _unique_arrs_and_bases(
    arrs: Iterable[np.ndarray],
) -> Generator[np.ndarray, None, None]:
    """
    Yields unique (by-ID) arrays from an iterable. If an array
    has a base, the base is yielded first (assuming that base
    object has not already been yielded).
    """
    seen = set()
    for arr in arrs:
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


def lock_array_and_base_writeability(arrs: Iterable[np.ndarray]):
    """Adds a lock on each of the provided arrays.

    If an array is a view, then its base also has a lock
    placed on it.

    Parameters
    ----------
    arrs : Iterable[ndarray]
        The arrays to be locked. Only one lock is placed
        on each array, even if the same array occurs
        multiple times in the iterable.
    """
    for arr in _unique_arrs_and_bases(arrs):
        lock_arr_writeability(arr)


def _release_lock_on_arr_writeability(arr: np.ndarray):
    arr_id = id(arr)

    if arr_id in _array_counter:
        _array_counter[arr_id] -= 1

        assert _array_counter[arr_id] >= 0  # TODO: remove this

        if _array_counter[arr_id] == 0:
            _array_counter.pop(arr_id, None)
            _array_tracker.pop(arr_id, None)

            if not arr.flags.writeable:

                if arr.base is not None and arr.base.flags.writeable is False:
                    # array is view and must wait until its base is released
                    # before it can be unlocked
                    _views_waiting_for_unlock[id(arr.base)] = arr
                else:
                    arr.flags.writeable = True

    if (
        arr.base is None
        and arr.flags.writeable
        and (arr_id in _views_waiting_for_unlock)
    ):
        # array was base of view waiting to be unlocked..
        # unlock the view
        view_arr = _views_waiting_for_unlock[arr_id]
        view_arr_id = id(view_arr)

        if view_arr_id not in _array_counter:
            view_arr.flags.writeable = True
            _views_waiting_for_unlock.pop(arr_id)


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
    cnt = 0  # counts number of living references
    weak_ref_cnt = len(arr_refs.data)  # gives total num weak-references
    for arr in _unique_arrs_and_bases(arr_refs):
        cnt += 1
        _release_lock_on_arr_writeability(arr)

    if cnt != weak_ref_cnt:
        for item in set(_array_counter) - set(_array_tracker):
            _array_counter.pop(item)
