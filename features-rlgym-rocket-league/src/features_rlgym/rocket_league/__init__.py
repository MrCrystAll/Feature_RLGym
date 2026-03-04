"""
This module contains the following features for the Rocket League environment:

Ball prediction :func:`features_rlgym.rocket_league.features.feature_ball_pred.add_ball_pred_feature`

"""

from . import features
from .config import create_config, create_default_config

__all__ = ["features", "create_config", "create_default_config"]
