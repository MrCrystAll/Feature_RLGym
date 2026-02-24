"""The module for the boost pads features"""

from collections.abc import Hashable
from typing import Any

import numpy as np

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM, BOOST_LOCATIONS

from features_rlgym.api.feature import Feature
from features_rlgym.api.feature_config import FeatureConfig


class FeatureBoostPadTimers(Feature[Hashable, np.ndarray, np.ndarray, GameState, int]):
    """A feature to add boost pad timers to the obs"""

    def __init__(
        self,
        pad_timer_normalization: float | np.ndarray = 1 / 10,
    ) -> None:
        self.pad_timer_coef = pad_timer_normalization

    def get_obs_additional_size(self, agent: Hashable) -> int:
        return len(BOOST_LOCATIONS)

    def apply_to_observation_builder(
        self,
        obs: dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        _new_obs = {}

        for agent, _obs in obs.items():
            _pads = (
                state.inverted_boost_pad_timers
                if state.cars[agent].team_num == ORANGE_TEAM
                else state.boost_pad_timers
            ) * self.pad_timer_coef

            _added_obs = [_pads]

            _new_obs[agent] = np.concatenate((_obs, *_added_obs))

        return _new_obs


def add_boost_pad_timers_feature(
    config: FeatureConfig,
    pad_timer_normalization: float | np.ndarray = 1 / 10,
):
    """Adds a boost pad timer feature to the config

    :param config: The config to add the feature on
    :type config: FeatureConfig
    :param pad_timer_normalization: Boost pad timers normalization coefficient, defaults to 1/10
    :type pad_timer_normalization: float | np.ndarray, optional
    """
    config.obs_builder.add_feature(FeatureBoostPadTimers(pad_timer_normalization))
