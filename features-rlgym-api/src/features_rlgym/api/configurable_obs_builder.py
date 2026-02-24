"""The module of the configurable obs builder"""

from typing import Any, Dict, Generic, List

from rlgym.api import ObsBuilder, AgentID, StateType, ObsType, ObsSpaceType

from features_rlgym.api.feature import Feature


class ConfigurableObsBuilder(
    Generic[AgentID, ObsType, StateType, ObsSpaceType],
    ObsBuilder[AgentID, ObsType, StateType, ObsSpaceType],
):
    """An observation builder where you can add features"""

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
        """This function is being triggered once the "main" obs builder finished building
        the observations, it adds info after the creation

        :param agents: The agents used to build the observations
        :type agents: list[AgentID]
        :param existing_obs: The created observations
        :type existing_obs: dict[AgentID, ObsType]
        :param state: The state the observations were created on
        :type state: StateType
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        :return: The new modified observations
        :rtype: dict[AgentID, ObsType]
        """
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

    def add_feature(self, feature: Feature):
        """Adds a feature to the obs builder

        :param feature: The feature to add
        :type feature: Feature
        """
        self.features.append(feature)

    def get_obs_space(self, agent: AgentID) -> ObsSpaceType:
        _base_space = self._obs_builder.get_obs_space(agent)

        _base_space += sum(
            map(lambda feat: feat.get_obs_additional_size(agent), self.features)
        )

        return _base_space
