"""
Provides utilities responsible for locking/releasing array writeability.
"""

import atexit
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING
from weakref import ref

import numpy as np

from mygrad._utils import ContextTracker, WeakRef, WeakRefIterable

if TYPE_CHECKING:  # pragma: no cover
    from mygrad.tensor_base import Tensor as TensorType


# arr-id -> num active ops involving arr
# (a plain dict rather than a Counter: this is on the hot path of every op)
_array_counter: dict[int, int] = {}

# arr-id -> weak-ref of arr, for arrays participating in live ops
_array_tracker: dict[int, WeakRef[np.ndarray]] = dict()

# maps base-array ID to ID of view that can't be unlocked until
# base is unlocked
_views_waiting_for_unlock: defaultdict[int, set[int]] = defaultdict(set)

__all__ = [
    "lock_arr_writeability",
    "release_writeability_lock_on_op",
    "mem_guard_off",
    "mem_guard_on",
    "mem_guard_active",
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
    arr_ref = _array_tracker.get(arr_id)
    if arr_ref is None or arr_ref() is None:  # `arr` is not tracked
        if (
            not force_lock
            and not arr.flags.writeable
            and (arr.base is None or not array_is_tracked(arr.base))
        ):
            # array is natively read-only; don't do anything
            return arr
        # keeps track of array so we can clean up the array
        # counter when tracked arrays fall out of scope
        _array_tracker[arr_id] = ref(arr)
        _array_counter[arr_id] = 1
    else:
        # a view waiting on its base stays tracked without a counter entry
        _array_counter[arr_id] = _array_counter.get(arr_id, 0) + 1
    if arr.flags.writeable is True:
        arr.flags.writeable = False
    return arr


def unlock_view_with_base(view: np.ndarray):
    """Has a read-only view, whose base is locked by mygrad, made writeable once
    that base's lock is released.

    E.g. a view taken of an array while it is involved in an op would otherwise
    remain read-only after the op is done."""
    base = view.base
    if view.flags.writeable or base is None or not array_is_tracked(base):
        return
    view_id = id(view)
    _array_tracker[view_id] = ref(view)
    _views_waiting_for_unlock[id(base)].add(view_id)


def lock_arrs_of_tensors(
    tensors: Iterable["TensorType"],
) -> WeakRefIterable[np.ndarray]:
    """Locks the unique (by-ID) arrays of ``tensors``, along with the bases of
    any that are views, returning weak references to them in the order in which
    their locks must be released.

    A base array is locked - and thus released - before any of its views, so
    that its writeability is restored first; only then can the views' be."""
    locked: WeakRefIterable[np.ndarray] = WeakRefIterable()
    refs = locked.data
    seen = set()
    for t in tensors:
        arr = t.data
        arr_id = id(arr)
        if arr_id in seen:
            continue
        base = arr.base
        if base is not None:
            # a base must be locked (and thus released) before its views
            base_id = id(base)
            if base_id not in seen:
                seen.add(base_id)
                refs.append(ref(lock_arr_writeability(base)))
        seen.add(arr_id)
        refs.append(ref(lock_arr_writeability(arr)))
    return locked


def _release_lock_on_arr_writeability(arr: np.ndarray):
    """
    Decrements the number of active ops the array participates in.
    An array no longer participating in any ops will have its
    writeability restored.
    """
    arr_id = id(arr)
    num_active_ops = _array_counter.get(arr_id, 0)

    if num_active_ops == 1:
        # final active op involving array is being de-referenced:
        # okay to unlock array
        del _array_counter[arr_id]

        if arr.base is not None and arr.base.flags.writeable is False:
            # Array is view and must wait until its base is released
            # before it can be unlocked
            # Thus we are still tracking this array
            _views_waiting_for_unlock[id(arr.base)].add(arr_id)
        else:
            # we no longer need to track the array
            arr.flags.writeable = True
            _array_tracker.pop(arr_id, None)
            if not _array_tracker and _views_waiting_for_unlock:
                # If no arrays are being tracked, then there can't
                # be any views waiting to be unlocked.
                # Clean up!
                _views_waiting_for_unlock.clear()
    elif num_active_ops > 0:
        _array_counter[arr_id] = num_active_ops - 1

    if (
        arr_id in _views_waiting_for_unlock  # cheapest, and usually False
        and arr.base is None
        and arr.flags.writeable
    ):
        # array was base of view waiting to be unlocked..
        #
        # Either:
        #    view no longer exists
        #    or view is involved in new op
        #    or view can now get unlocked
        # under all conditions view will no longer be waiting to be unlocked
        for view_arr_id in tuple(_views_waiting_for_unlock[arr_id]):
            if _array_counter.get(view_arr_id, 0) > 0:
                # view involved in new op
                continue

            _views_waiting_for_unlock[arr_id].remove(view_arr_id)

            try:
                view_arr = _array_tracker.pop(view_arr_id)()
                if view_arr is None:
                    continue
            except KeyError:
                # view array is no longer available for unlocking
                continue

            try:
                view_arr.flags.writeable = True
            except ValueError:  # pragma: no cover
                # sometimes this raises.. but it is not
                # reproducible and is very rare
                pass

        if not _views_waiting_for_unlock[arr_id]:
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
    for arr in arr_refs:
        _release_lock_on_arr_writeability(arr)


# keeps the weakrefs made by `release_locks_when_collected` alive until they fire
_pending_releases: set[WeakRef] = set()

# Discarding the weakrefs at exit means that their callbacks never fire during
# interpreter shutdown, when the state that they touch may be partially torn
# down. (Restoring array writeability at exit is moot.)
atexit.register(_pending_releases.clear)


def release_locks_when_collected(op: object, arr_refs: WeakRefIterable[np.ndarray]):
    """Calls ``release_writeability_lock_on_op(arr_refs)`` once ``op`` is garbage
    collected.

    Like ``weakref.finalize(op, release_writeability_lock_on_op, arr_refs)``, minus
    its extra machinery; this is called for every op, and is roughly twice as fast.
    Pending releases are discarded, rather than run, at interpreter exit."""

    def _release(op_ref: WeakRef):
        _pending_releases.discard(op_ref)
        release_writeability_lock_on_op(arr_refs)

    _pending_releases.add(ref(op, _release))


MEM_GUARD = os.environ.get("MYGRAD_MEM_GUARD", True)

if MEM_GUARD in {"True", "true", "1", 1, True}:
    MEM_GUARD = True
elif MEM_GUARD in {"False", "false", "0", 0, False}:  # pragma: no cover
    MEM_GUARD = False
else:  # pragma: no cover
    from warnings import warn

    warn(
        f"Environment variable MYGRAD_MEM_GUARD was set to an unknown value {MEM_GUARD}. "
        f"Proceeding with `MEM_GUARD=True`"
    )
    MEM_GUARD = True


class MemStateContext(ContextTracker):
    @property
    def state(self):
        return MEM_GUARD

    @state.setter
    def state(self, value: bool):
        if not isinstance(value, bool):  # pragma: no cover
            raise TypeError(
                f"MEM_GUARD must be set to a boolean value, got {value} (type={type(value)})"
            )

        global MEM_GUARD
        MEM_GUARD = value


class _NoMemGuard(MemStateContext):
    """A context manager used to suspend memory-locking behavior

    Examples
    --------
    >>> from mygrad import  mem_guard_off
    >>> with mem_guard_off:
    ...     # array-memory locking is turned off
    ...     pass
    ... # previous memory-locking behavior is restored

    This can also be used as a decorator

    >>> @mem_guard_off
    >>> def f():
    ...     # array-memory locking is turned off within function
    ...     return

    """

    _enter_set_value = False


class _WithMemGuard(MemStateContext):
    """A context manager used to enable memory-locking behavior

    Examples
    --------
    >>> from mygrad import mem_guard_on
    >>> with mem_guard_on:
    ...     # array-memory locking is turned on
    ...     pass
    ... # previous memory-locking behavior is restored

    This can also be used as a decorator

    >>> @mem_guard_on
    >>> def f():
    ...     # array-memory locking is turned on within function
    ...     return

    """

    _enter_set_value = True


mem_guard_off = _NoMemGuard()
mem_guard_on = _WithMemGuard()


def turn_memory_guarding_off():
    """Globally disables all memory-guarding mechanisms, except
    for in contexts where they are explicitly enabled.

    Notes
    -----
    With memory guarding disabled, arrays participating in active
    computational graphs are not protected from being mutated by
    the user. Mutating such an array will corrupt the derivatives
    that are computed via back-propagation, and will produce
    incorrect results.

    This can speed up computations involving many small tensors
    substantially.

    If you want to disable memory guarding at the system level, you
    can set the system environment variable MYGRAD_MEM_GUARD=False.
    NOTE THAT THIS IS NOT RECOMMENDED.

    See Also
    --------
    turn_memory_guarding_on : Globally enables all memory-guarding mechanisms
    mem_guard_off : context manager & decorator for suspending memory guarding
    mem_guard_on : context manager & decorator for enabling memory guarding

    Examples
    --------
    The following demonstrates how one can unwittingly corrupt
    backpropagation through a computational graph

    >>> import mygrad as mg
    >>> import numpy as np
    >>> mg.turn_memory_guarding_off()  # speeds up calculations, but with risks involved..
    >>> x = np.arange(3.)
    >>> y = mg.ones_like(x)
    >>> z = x * y
    >>> x[:] = 0  # mutates x, corrupting state associated with z
    >>> z.backward()
    >>> y.grad  # would be array([0., 1., 2.]) if graph wasn't corrupted
    array([0., 0., 0.])
    """
    global MEM_GUARD
    MEM_GUARD = False


def turn_memory_guarding_on():
    """Globally enables all memory-guarding mechanisms, except
    for in contexts where they are explicitly disabled.

    Notes
    -----
    Memory guarding is enabled by default. It ensures that arrays
    that are participating in computational graphs cannot be mutated
    (at least unwittingly..), which provides important assurances that
    the state of the computational graph is not corrupted for
    back-propagation.

    Memory guarding can slow down computations involving many small tensors.
    Realistic worst-case benchmarks suggest a ~50% slowdown.

    If performance is important, it is recommended that you test your code leaving
    memory guarding enabled. Presuming the code runs without any errors regarding
    writing to read-only arrays, you can proceed to disable memory guarding and
    enjoy the concomitant speedups.

    Note also that running your code in a `no_autodiff` context will automatically
    disable memory guarding.

    See Also
    --------
    turn_memory_guarding_off : Globally enables all memory-guarding mechanisms
    mem_guard_off : context manager & decorator for suspending memory guarding
    mem_guard_on : context manager & decorator for enabling memory guarding
    no_autodiff : context manager for disabling graph-tracking for back propagation

    Examples
    --------
    The following demonstrates how memory guarding prevents one from
    unwittingly corrupting an active computational graph

    >>> import mygrad as mg
    >>> import numpy as np
    >>> # (Note that memory guarding is on by default, so
    >>> # this call isn't usually needed...)
    >>> mg.turn_memory_guarding_on()
    >>> x = np.arange(3.)
    >>> y = mg.ones_like(x)
    >>> z = x * y
    >>> try:
    ...     x[:] = 0  # raises because `x` is made read-only
    ... except ValueError:
    ...     pass
    >>> z.backward()
    >>> y.grad  # correct gradient is computed
    array([0., 1., 2.])
    """
    global MEM_GUARD
    MEM_GUARD = True


def mem_guard_active() -> bool:
    """Indicates whether or not memory guarding is active.

    See Also
    --------
    turn_memory_guarding_on : Globally enables all memory-guarding mechanisms
    turn_memory_guarding_off : Globally enables all memory-guarding mechanisms
    mem_guard_off : context manager & decorator for suspending memory guarding
    mem_guard_on : context manager & decorator for enabling memory guarding
    """
    return MEM_GUARD


def force_lock_tensor_and_creators(tensor: "TensorType"):
    tensor_refs = lock_arrs_of_tensors(tensor.creator.variables)
    lock_arr_writeability(tensor.data, force_lock=True)
    tensor_refs.append(tensor.data)
    release_locks_when_collected(tensor.creator, tensor_refs)
