from collections.abc import Hashable
import math
from typing import Any

import numpy as np

from feature_rlgym.api.feature import Feature

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM

from feature_rlgym.api.feature_config import FeatureConfig


class FeatureBall(Feature[Hashable, np.ndarray, np.ndarray, GameState, int]):
    def __init__(
        self,
        position_normalization: float | np.ndarray = 1 / 2300,
        linear_velocity_normalization: float | np.ndarray = 1 / 2300,
        angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
    ) -> None:
        self.pos_coef = position_normalization
        self.lin_vel_coef = linear_velocity_normalization
        self.ang_vel_coef = angular_velocity_normalization

    def get_obs_additional_size(self, agent: Hashable) -> int:
        return 9

    def apply_to_observation_builder(
        self,
        obs: dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        _new_obs = {}

        for agent, _obs in obs.items():
            _ball = (
                state.inverted_ball
                if state.cars[agent].team_num == ORANGE_TEAM
                else state.ball
            )

            _added_obs = [
                _ball.position * self.pos_coef,
                _ball.linear_velocity * self.lin_vel_coef,
                _ball.angular_velocity * self.ang_vel_coef,
            ]

            _new_obs[agent] = np.concatenate((_obs, *_added_obs), dtype=_obs.dtype)

        return _new_obs


def add_ball_feature(
    config: FeatureConfig,
    position_normalization: float | np.ndarray = 1 / 2300,
    linear_velocity_normalization: float | np.ndarray = 1 / 2300,
    angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
):
    config.obs_builder.add_feature(
        FeatureBall(
            position_normalization,
            linear_velocity_normalization,
            angular_velocity_normalization,
        )
    )
