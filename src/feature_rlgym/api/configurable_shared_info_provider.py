"""The module of the configurable shared info provider"""

from typing import Any, Dict, Generic, List

from rlgym.api import SharedInfoProvider, AgentID, StateType

from feature_rlgym.api.feature import Feature


class ConfigurableSharedInfoProvider(
    Generic[AgentID, StateType], SharedInfoProvider[AgentID, StateType]
):
    """A shared info provider where you can add features"""

    def __init__(self) -> None:
        self.features: list[Feature] = []

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        for feature in self.features:
            shared_info = feature.create_shared_info(shared_info)
        return shared_info

    def set_state(
        self,
        agents: List[AgentID],
        initial_state: StateType,
        shared_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        for feature in self.features:
            shared_info = feature.reset_shared_info(agents, initial_state, shared_info)
        return shared_info

    def step(
        self, agents: List[AgentID], state: StateType, shared_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        for feature in self.features:
            shared_info = feature.step_shared_info(agents, state, shared_info)
        return shared_info

    def add_feature(self, feature: Feature):
        """Adds a feature to the shared info provider

        :param feature: The feature to add
        :type feature: Feature
        """
        self.features.append(feature)
