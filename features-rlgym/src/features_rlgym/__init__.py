# features_rlgym/__init__.py
import importlib

class MissingModule:
    def __init__(self, name):
        self._name = name
    def __getattr__(self, attr):
        raise ImportError(f"Module 'features_rlgym.{self._name}' not installed. "
                          f"Install features_rlgym[{self._name}] to use it.")

for submodule in ["api", "rocket_league"]:
    try:
        globals()[submodule] = importlib.import_module(f"features_rlgym.{submodule}")
    except ModuleNotFoundError:
        globals()[submodule] = MissingModule(submodule)