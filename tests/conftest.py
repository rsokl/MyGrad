import os
import warnings

import pytest
from hypothesis import Verbosity, settings

import mygrad as mg
import mygrad._utils.graph_tracking as track
import mygrad._utils.lock_management as lock
from tests.utils import clear_all_mem_locking_state

settings.register_profile("ci", deadline=1000)
settings.register_profile("intense", deadline=None, max_examples=1000)
settings.register_profile("dev", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))

COVERAGE_MODE = bool(os.getenv("MYGRAD_COVERAGE_MODE", False))


@pytest.fixture(autouse=True)
def seal_memguard() -> bool:
    """Ensure test cannot mutate MEM_GUARD value"""
    initial_value = lock.MEM_GUARD

    yield initial_value
    if lock.MEM_GUARD is not initial_value:
        warnings.warn("test toggled MEM_GUARD value")
        lock.MEM_GUARD = initial_value
        assert False

    lock.MEM_GUARD = initial_value


@pytest.fixture()
def no_autodiff():
    with mg.no_autodiff:
        yield None


@pytest.fixture(autouse=True)
def seal_graph_tracking() -> bool:
    """Ensure test cannot mutate TRACK_GRAPH value"""
    initial_value = track.TRACK_GRAPH
    yield initial_value

    if track.TRACK_GRAPH is not initial_value:
        warnings.warn("test toggled TRACK_GRAPH value")
        track.TRACK_GRAPH = initial_value
        assert False

    track.TRACK_GRAPH = initial_value


@pytest.fixture(autouse=True)
def raise_on_mem_locking_state_leakage() -> bool:
    """Ensure mem-locking state is isolated to each test, and raise if
    a test leaks state"""
    clear_all_mem_locking_state()

    yield None

    if any([lock._views_waiting_for_unlock, lock._array_tracker, lock._array_counter]):
        warnings.warn(
            f"leak\nviews waiting:{lock._views_waiting_for_unlock}"
            f"\narr-tracker:{lock._array_tracker}"
            f"\narr-counter{lock._array_counter}"
        )
        # coverage mode seems to mess with mem-guard synchronization
        assert COVERAGE_MODE

    clear_all_mem_locking_state()
