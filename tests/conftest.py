import os

from hypothesis import Verbosity, settings

settings.register_profile("ci", deadline=1000)
settings.register_profile("intense", deadline=None, max_examples=1000)
settings.register_profile("dev", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "default"))
