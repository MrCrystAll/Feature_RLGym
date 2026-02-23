"""The module for an empty obs builder"""

from collections.abc import Hashable
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ObsBuilder
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction

from feature_rlgym.api.feature_config import FeatureConfig


class EmptyBuilder(ObsBuilder[Hashable, np.ndarray, GameState, int]):
    """A builder that does nothing...

    Yes. Nothing.
    """

    def build_obs(
        self, agents: List[Hashable], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[Hashable, np.ndarray]:
        return {agent: np.asarray([]) for agent in agents}

    def reset(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_obs_space(self, agent: Hashable) -> int:
        return 0


def create_config():
    return FeatureConfig(
        EmptyBuilder(),
        RepeatAction(LookupTableAction())
    )
    
def create_default_config(zero_padding: int = 3):
    from feature_rlgym.rocket_league.features import add_ball_feature, add_self_feature, add_others_feature, add_boost_pad_timers_feature
    
    config = create_config()
    add_ball_feature(config)
    add_self_feature(config)
    add_boost_pad_timers_feature(config)
    add_others_feature(config, zero_padding=zero_padding)
    
    return config