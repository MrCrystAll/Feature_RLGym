from collections.abc import Hashable
import math
from typing import Any

import numpy as np

from feature_rlgym.api.feature import Feature

from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM

from feature_rlgym.api.feature_config import FeatureConfig


class FeatureOthers(Feature[Hashable, np.ndarray, np.ndarray, GameState, int]):
    """A feature to add other cars to the obs"""

    def __init__(
        self,
        position_normalization: float | np.ndarray = 1 / 2300,
        linear_velocity_normalization: float | np.ndarray = 1 / 2300,
        angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
    ) -> None:
        self.pos_coef = position_normalization
        self.lin_vel_coef = linear_velocity_normalization
        self.ang_vel_coef = angular_velocity_normalization
        self._state: GameState | None = None

    def get_obs_additional_size(self, agent: Hashable) -> int:
        if self._state is None:
            return 0
        return (len(self._state.cars) - 1) * 9

    def on_obs_builder_reset(
        self,
        agents: list[Hashable],
        initial_state: GameState,
        shared_info: dict[str, Any],
    ):
        self._state = initial_state

    def apply_to_observation_builder(
        self,
        obs: dict[Hashable, np.ndarray],
        state: GameState,
        shared_info: dict[str, Any],
    ) -> dict[Hashable, np.ndarray]:
        self._state = state
        _new_obs = {}

        for agent, _obs in obs.items():
            _agent_car = state.cars[agent]
            _inverted = _agent_car.team_num == ORANGE_TEAM

            _teammates = []
            _opponents = []

            for _agent, _car in state.cars.items():
                if _agent == agent:
                    continue

                if _car.team_num == _agent_car.team_num:
                    _teammates.extend(self._generate_player_obs(_car, _inverted))
                else:
                    _opponents.extend(self._generate_player_obs(_car, _inverted))

            _added_obs = [*_teammates, *_opponents]

            _new_obs[agent] = np.concatenate((_obs, *_added_obs), dtype=_obs.dtype)

        return _new_obs

    def _generate_player_obs(self, car: Car, inverted):
        if inverted:
            physics = car.inverted_physics
        else:
            physics = car.physics

        return [
            physics.position * self.pos_coef,
            physics.linear_velocity * self.lin_vel_coef,
            physics.angular_velocity * self.ang_vel_coef,
        ]


def add_others_feature(
    config: FeatureConfig,
    position_normalization: float | np.ndarray = 1 / 2300,
    linear_velocity_normalization: float | np.ndarray = 1 / 2300,
    angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
):
    """Adds a feature to add other cars to the obs

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
        FeatureOthers(
            position_normalization,
            linear_velocity_normalization,
            angular_velocity_normalization,
        )
    )
