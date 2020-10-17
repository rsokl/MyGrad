import os
import warnings

import pytest
from hypothesis import Verbosity, settings

import mygrad._utils.graph_tracking as track
import mygrad._utils.lock_management as lock

settings.register_profile("ci", deadline=1000)
settings.register_profile("intense", deadline=None, max_examples=1000)
settings.register_profile("dev", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture()
def seal_memguard() -> bool:
    """Ensure test cannot mutate MEM_GUARD value"""
    initial_value = lock.MEM_GUARD

    yield initial_value
    if lock.MEM_GUARD is not initial_value:
        warnings.warn("test toggled MEM_GUARD value")

    lock.MEM_GUARD = initial_value


@pytest.fixture()
def seal_graph_tracking() -> bool:
    """Ensure test cannot mutate TRACK_GRAPH value"""
    initial_value = track.TRACK_GRAPH
    yield initial_value

    if track.TRACK_GRAPH is not initial_value:
        warnings.warn("test toggled TRACK_GRAPH value")

    track.TRACK_GRAPH = initial_value
