"""
Provides utilities responsible for locking/releasing array writeability.
"""
import os
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, DefaultDict, Dict, Generator, Iterable
from weakref import ref

import numpy as np

from mygrad._utils import ContextTracker, WeakRefIterable

if TYPE_CHECKING:  # pragma: no cover
    from typing import Set

    from mygrad import Tensor
    from mygrad._utils import WeakRef

# arr-id -> num active ops involving arr
_array_counter = Counter()  # type: Counter[int, int]

# arr-id -> weak-ref of arr, for arrays participating in live ops
_array_tracker = dict()  # type: Dict[int, WeakRef[Tensor]]

# maps base-array ID to ID of view that can't be unlocked until
# base is unlocked
_views_waiting_for_unlock = defaultdict(
    set
)  # type: DefaultDict[int, Set[int]] # base-id -> set of view-ids

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
    if not array_is_tracked(arr):
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
    """
    Decrements the number of active ops the array participates in.
    An array no longer participating in any ops will have its
    writeability restored.
    """
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
        for view_arr_id in tuple(_views_waiting_for_unlock[arr_id]):

            if _array_counter[view_arr_id] > 0:
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
    """ A context manager used to suspend memory-locking behavior

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
    """ A context manager used to enable memory-locking behavior

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
    """ Globally disables all memory-guarding mechanisms, except
    for in contexts where they are explicitly enabled.

    Notes
    -----
    With memory guarding disables, arrays participating in active
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
    """ Globally enables all memory-guarding mechanisms, except
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
    >>> # memory guarding is on by default
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
    """ Indicates whether or not memory guarding is active.

    See Also
    --------
    turn_memory_guarding_on : Globally enables all memory-guarding mechanisms
    turn_memory_guarding_off : Globally enables all memory-guarding mechanisms
    mem_guard_off : context manager & decorator for suspending memory guarding
    mem_guard_on : context manager & decorator for enabling memory guarding
    """
    return MEM_GUARD
