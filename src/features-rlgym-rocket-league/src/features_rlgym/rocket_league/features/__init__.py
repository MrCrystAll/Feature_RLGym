"""The module containing all the features for Rocket League"""

from .ball import add_ball_feature
from .self import add_self_feature
from .boost_pads import add_boost_pad_timers_feature
from .others import add_others_feature
from .ball_pred import add_ball_pred_feature

__all__ = [
    "add_ball_feature",
    "add_self_feature",
    "add_boost_pad_timers_feature",
    "add_others_feature",
    "add_ball_pred_feature",
]
