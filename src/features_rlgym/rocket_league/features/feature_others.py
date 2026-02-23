"""The module for the feature to add others to the obs"""

from collections.abc import Hashable
import math
import random
from typing import Any, Self

import numpy as np

from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import ORANGE_TEAM

from features_rlgym.api.feature import Feature
from features_rlgym.api.feature_config import FeatureConfig


class FeatureOthers(Feature[Hashable, np.ndarray, np.ndarray, GameState, int]):
    """A feature to add other cars to the obs"""

    PLAYER_SIZE = 9

    def __init__(
        self,
        position_normalization: float | np.ndarray = 1 / 2300,
        linear_velocity_normalization: float | np.ndarray = 1 / 2300,
        angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
    ) -> None:
        self.pos_coef = position_normalization
        self.lin_vel_coef = linear_velocity_normalization
        self.ang_vel_coef = angular_velocity_normalization
        self.zero_padding = 0
        self._shuffle = False
        self._state: GameState | None = None

    def get_obs_additional_size(self, agent: Hashable) -> int:
        if self._state is None:
            return 0
        if self.zero_padding > 0:
            return (self.zero_padding * 2 - 1) * self.PLAYER_SIZE
        return (len(self._state.cars) - 1) * self.PLAYER_SIZE

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

            _self_obs = self._generate_player_obs(_agent_car, _inverted)

            _teammates = []
            _opponents = []

            for _agent, _car in state.cars.items():
                if _agent == agent:
                    continue

                if _car.team_num == _agent_car.team_num:
                    _teammates.append(self._generate_player_obs(_car, _inverted))
                else:
                    _opponents.append(self._generate_player_obs(_car, _inverted))

            if self.zero_padding > 0:
                # Padding for multi game mode
                while len(_teammates) < self.zero_padding - 1:
                    _teammates.append(np.zeros_like(_self_obs))
                while len(_opponents) < self.zero_padding:
                    _opponents.append(np.zeros_like(_self_obs))

            if self._shuffle:
                random.shuffle(_teammates)
                random.shuffle(_opponents)

            _np_teammates = np.asarray(_teammates).reshape(
                (len(_teammates) * self.PLAYER_SIZE,)
            )
            _np_opponents = np.asarray(_opponents).reshape(
                (len(_opponents) * self.PLAYER_SIZE,)
            )

            _new_obs[agent] = np.concatenate(
                (_obs, _np_teammates, _np_opponents), dtype=_obs.dtype
            )

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

    def pad_per_team(self, zero_padding: int = 3) -> Self:
        self.zero_padding = zero_padding
        return self

    def shuffle(self, shuffle: bool) -> Self:
        self._shuffle = shuffle
        return self


def add_others_feature(
    config: FeatureConfig,
    position_normalization: float | np.ndarray = 1 / 2300,
    linear_velocity_normalization: float | np.ndarray = 1 / 2300,
    angular_velocity_normalization: float | np.ndarray = 1 / math.pi,
    zero_padding: int = 0,
    shuffle: bool = False,
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
    :param zero_padding: The amount of padding per team, defaults to 0 (no padding)
    :type zero_padding: int, optional
    :param shuffle: Whether to shuffle (when padding) the teammates and opponents
    :type shuffle: bool, optional
    """
    config.obs_builder.add_feature(
        FeatureOthers(
            position_normalization,
            linear_velocity_normalization,
            angular_velocity_normalization,
        )
        .pad_per_team(zero_padding)
        .shuffle(shuffle)
    )
