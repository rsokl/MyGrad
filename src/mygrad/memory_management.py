from collections import Counter
from typing import Iterable
from weakref import WeakValueDictionary

import numpy as np

from mygrad._utils import WeakRefIterable

_array_counter = Counter()
_array_tracker = WeakValueDictionary()

__all__ = ["lock_array_and_base_memories", "release_op_memory"]


def lock_arr_memory(arr: np.ndarray, force_lock: bool = False):
    arr_id = id(arr)
    if arr_id not in _array_tracker:
        if not force_lock and not arr.flags.writeable:
            # array is natively read-only; don't do anything
            return
        _array_tracker[arr_id] = arr
        _array_counter[arr_id] = 1
    else:
        _array_counter[arr_id] += 1
    if arr.flags.writeable is True:
        arr.flags.writeable = False


def _unique_arrs_and_bases(arrs: Iterable[np.ndarray]):
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


def lock_array_and_base_memories(arrs: Iterable[np.ndarray]):
    for arr in _unique_arrs_and_bases(arrs):
        lock_arr_memory(arr)


def _release_arr_memory(arr: np.ndarray):
    arr_id = id(arr)
    _array_counter[arr_id] -= 1
    # assert _array_counter[arr_id] >= 0
    if _array_counter[arr_id] == 0:
        _array_counter.pop(arr_id)
        _array_tracker.pop(arr_id)
        if not arr.flags.writeable:
            arr.flags.writeable = True


def release_op_memory(arr_refs: WeakRefIterable[np.ndarray]):
    cnt = 0
    weak_ref_cnt = len(arr_refs.data)
    for arr in _unique_arrs_and_bases(arr_refs):
        _release_arr_memory(arr)

    if cnt != weak_ref_cnt:
        for item in set(_array_counter) - set(_array_tracker):
            _array_counter.pop(item)
