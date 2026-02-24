"""This module is the global package of features_rlgym,
it is mainly used to guide the user in case they forgot
to install a dependency and try to access it"""

import importlib


class MissingModule:
    """A little class to simplify the detection of a missing module"""

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise ImportError(
            f"Module 'features_rlgym.{self._name}' not installed. "
            f"Install features_rlgym[{self._name}] to use it."
        )


for submodule in ["api", "rocket_league"]:
    try:
        globals()[submodule] = importlib.import_module(f"features_rlgym.{submodule}")
    except ModuleNotFoundError:
        globals()[submodule] = MissingModule(submodule)
