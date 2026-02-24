"""The module containing the ball prediction feature"""

from collections.abc import Hashable
import math
from typing import Any

import numpy as np

from rlgym.rocket_league.api import GameState, PhysicsObject

from rlgym_tools.rocket_league.shared_info_providers.ball_prediction_provider import (
    BallPredictionProvider,
)

import RocketSim as rs

from features_rlgym.api.feature import Feature
from features_rlgym.api.feature_config import FeatureConfig


class FeatureBallPrediction(Feature[Hashable, np.ndarray, np.ndarray, GameState, int]):
    """A feature to add the ball in the obs builder"""

    BALL_PRED_SIZE = 9

    def __init__(
        self,
        limit_seconds: float | int,
        step_seconds: float | int,
        target_seconds: float | int | slice,
        game_mode=rs.GameMode.SOCCAR,
        position_normalization: float | np.ndarray = 1 / 2300,
        linear_velocity_normalization: float | np.ndarray = 1 / 2300,
        angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
    ) -> None:
        if isinstance(target_seconds, slice):
            self._target_seconds = slice(
                int(target_seconds.start) // int(step_seconds),
                int(target_seconds.stop) // int(step_seconds),
            )
        else:
            self._target_seconds = slice(
                int(target_seconds) // int(step_seconds),
                int(target_seconds) // int(step_seconds),
            )
        self.pos_coef = position_normalization
        self.lin_vel_coef = linear_velocity_normalization
        self.ang_vel_coef = angular_velocity_normalization
        self.ball_prediction_provider = BallPredictionProvider(
            limit_seconds, step_seconds, game_mode
        )

    def create_shared_info(self, shared_info: dict[str, Any]) -> dict[str, Any]:
        return self.ball_prediction_provider.create(shared_info)

    def step_shared_info(
        self, agents: list[Hashable], state: GameState, shared_info: dict[str, Any]
    ) -> dict[str, Any]:
        return self.ball_prediction_provider.step(agents, state, shared_info)

    def reset_shared_info(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[str, Any]:
        return self.ball_prediction_provider.set_state(
            agents, initial_state, shared_info
        )

    def get_obs_additional_size(self, agent: Hashable) -> int:
        return (
            self._target_seconds.stop - self._target_seconds.start
        ) * self.BALL_PRED_SIZE

    def apply_to_observation_builder(
        self,
        obs: dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        _new_obs = {}

        for agent, _obs in obs.items():
            _ball_pred: list[PhysicsObject] = shared_info["ball_prediction"][
                self._target_seconds
            ]

            _added_obs = []

            for _bp in _ball_pred:
                _added_obs.extend(
                    [
                        _bp.position * self.pos_coef,
                        _bp.linear_velocity * self.lin_vel_coef,
                        _bp.angular_velocity * self.ang_vel_coef,
                    ]
                )

            _new_obs[agent] = np.concatenate((_obs, *_added_obs))

        return _new_obs


def add_ball_pred_feature(
    config: FeatureConfig,
    limit_seconds: int = 5,
    step_seconds: int = 1,
    target_seconds: int | slice = 4,
    gamemode: int = rs.GameMode.SOCCAR,
    position_normalization: float | np.ndarray = 1 / 2300,
    linear_velocity_normalization: float | np.ndarray = 1 / 2300,
    angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
):
    """Adds a feature to add the ball to the obs

    :param config: The config to add the feature on
    :type config: FeatureConfig
    :param limit_seconds: The total amount of time to predict (in seconds), defaults to 5
    :type limit_seconds: float | int, optional
    :param step_seconds: The duration in between 2 predictions (in seconds), defaults to 1
    :type step_seconds: float | int, optional
    :param target_seconds: The targetted duration of the prediction (what the bot will see), defaults to 4
    :type target_seconds: float | int | slice, optional
    :param gamemode: _description_, defaults to rs.GameMode.SOCCAR
    :type gamemode: int, optional
    :param position_normalization: Position normalization coefficient, defaults to 1/2300
    :type position_normalization: float | np.ndarray, optional
    :param linear_velocity_normalization: Linear velocity normalization coefficient, defaults to 1/2300
    :type linear_velocity_normalization: float | np.ndarray, optional
    :param angular_velocity_normalization: Angular velocity normalization coefficient, defaults to 1/math.pi
    :type angular_velocity_normalization: float | np.ndarray, optional
    """
    feature = FeatureBallPrediction(
        limit_seconds,
        step_seconds,
        target_seconds,
        gamemode,
        position_normalization,
        linear_velocity_normalization,
        angular_velocity_normalization,
    )

    config.obs_builder.add_feature(feature)
    config.shared_info_provider.add_feature(feature)
