"""The module containing the ball feature"""

from collections.abc import Hashable
import math
from typing import Any

import numpy as np

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM

from features_rlgym.api.feature import Feature
from features_rlgym.api.feature_config import FeatureConfig


class FeatureBall(Feature[Hashable, np.ndarray, np.ndarray, GameState, int]):
    """A feature to add the ball in the obs builder"""

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

            _new_obs[agent] = np.concatenate((_obs, *_added_obs))

        return _new_obs


def add_ball_feature(
    config: FeatureConfig,
    position_normalization: float | np.ndarray = 1 / 2300,
    linear_velocity_normalization: float | np.ndarray = 1 / 2300,
    angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
):
    """Adds a feature to add the ball to the obs

    :param config: The config to add the feature on
    :type config: FeatureConfig
    :param position_normalization: Position normalization coefficient, defaults to 1/2300
    :type position_normalization: float | np.ndarray, optional
    :param linear_velocity_normalization: Linear velocity normalization coefficient, defaults to 1/2300
    :type linear_velocity_normalization: float | np.ndarray, optional
    :param angular_velocity_normalization: Angular velocity normalization coefficient, defaults to 1/math.pi
    :type angular_velocity_normalization: float | np.ndarray, optional
    """
    config.obs_builder.add_feature(
        FeatureBall(
            position_normalization,
            linear_velocity_normalization,
            angular_velocity_normalization,
        )
    )
