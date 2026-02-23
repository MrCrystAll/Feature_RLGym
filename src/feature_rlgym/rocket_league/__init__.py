"""This modules contains all the stuff necessary to add features to your Rocket League bot"""

from . import features
from .config import create_config, create_default_config

__all__ = ["features", "create_config", "create_default_config"]
