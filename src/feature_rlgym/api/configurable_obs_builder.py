from abc import abstractmethod
from typing import Any, Dict, Generic, List, Self

from rlgym.api import ObsBuilder, AgentID, StateType, ObsType, ObsSpaceType

from feature_rlgym.api.feature import Feature


class ConfigurableObsBuilder(
    Generic[AgentID, ObsType, StateType, ObsSpaceType],
    ObsBuilder[AgentID, ObsType, StateType, ObsSpaceType],
):
    def __init__(
        self, obs_builder: ObsBuilder[AgentID, ObsType, StateType, ObsSpaceType]
    ) -> None:
        self._obs_builder = obs_builder
        self.features: list[Feature] = []

    def reset(
        self,
        agents: List[AgentID],
        initial_state: StateType,
        shared_info: Dict[str, Any],
    ) -> None:
        self._obs_builder.reset(agents, initial_state, shared_info)

    def additionnal_info(
        self,
        agents: list[AgentID],
        existing_obs: dict[AgentID, ObsType],
        state: StateType,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, ObsType]:
        for feature in self.features:
            existing_obs = feature.apply_to_observation_builder(
                existing_obs, state, shared_info
            )
        return existing_obs

    def build_obs(
        self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]
    ) -> Dict[AgentID, ObsType]:
        return self.additionnal_info(
            agents,
            self._obs_builder.build_obs(agents, state, shared_info),
            state,
            shared_info,
        )

    def add_feature(self, feature: Feature) -> Self:
        """Adds a feature to the obs builder

        :param feature: The feature to add
        :type feature: Feature
        :return: The obs builder
        :rtype: Self
        """
        self.features.append(feature)

        return self

    def get_obs_space(self, agent: AgentID) -> ObsSpaceType:
        _base_space = self._obs_builder.get_obs_space(agent)

        for _feature in self.features:
            _feature_space = _feature.get_obs_additional_size(agent)

            _base_space += _feature_space

        return _base_space
