from mygrad._utils import lock_management as mem


def clear_all_mem_locking_state():
    mem._views_waiting_for_unlock.clear()
    mem._array_tracker.clear()
    mem._array_counter.clear()
