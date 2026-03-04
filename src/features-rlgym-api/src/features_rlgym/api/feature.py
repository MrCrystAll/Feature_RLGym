"""The module of the Feature base class"""

from typing import Any, Generic

from rlgym.api import AgentID, ObsType, StateType, ActionType, ObsSpaceType


class Feature(Generic[AgentID, ObsType, ActionType, StateType, ObsSpaceType]):
    """A feature is used to easily add an element to a bot"""

    def on_obs_builder_reset(
        self,
        agents: list[AgentID],
        initial_state: StateType,
        shared_info: dict[str, Any],
    ):
        """Called when the obs builder resets

        :param agents: The agents of the environment
        :type agents: list[AgentID]
        :param initial_state: The state after reset
        :type initial_state: StateType
        :param shared_info: The shared info of the environment
        :type shared_info: dict[str, Any]
        """

    def get_obs_additional_size(self, agent: AgentID) -> ObsSpaceType:
        """Returns the additional size added by the feature to the obs

        :param agent: The agent to get the additional size of
        :type agent: AgentID
        :return: The additional size added by the feature to the obs
        :rtype: ObsSpaceType
        """
        raise NotImplementedError()

    def apply_to_observation_builder(
        self, obs: dict[AgentID, ObsType], state: StateType, shared_info: dict[str, Any]
    ) -> dict[AgentID, ObsType]:
        """Adds (or remove) fields from the obs returned by the obs builder

        :param obs: The dictionnary of observations returned by the obs builder
        :type obs: dict[AgentID, ObsType]
        :param state: The state used to build the obs
        :type state: StateType
        :param shared_info: The shared info of the environment used to build the obs
        :type shared_info: dict[str, Any]
        :return: The modified obs
        :rtype: dict[AgentID, ObsType]
        """
        return obs

    def apply_to_action_parser(
        self,
        actions: dict[AgentID, ActionType],
        state: StateType,
        shared_info: dict[str, Any],
    ) -> dict[AgentID, ActionType]:
        """Modifies the action parser's returned actions after the action parser runs

        :param actions: The actions returned by the action parser
        :type actions: dict[AgentID, ActionType]
        :param state: The state the actions were built on
        :type state: StateType
        :param shared_info: The shared info used to build the actions
        :type shared_info: dict[str, Any]
        :return: The modified actions
        :rtype: dict[AgentID, ActionType]
        """
        return actions

    def create_shared_info(self, shared_info: dict[str, Any]) -> dict[str, Any]:
        """This function is called during the shared info creation

        :param shared_info: The previous shared information dictionary
        :type shared_info: dict[str, Any]
        :return: The modified shared information dictionary
        :rtype: dict[str, Any]
        """
        return shared_info

    def reset_shared_info(
        self,
        agents: list[AgentID],
        initial_state: StateType,
        shared_info: dict[str, Any],
    ) -> dict[str, Any]:
        """This function is called during the shared info reset / set_state

        :param agents: List of AgentIDs for which this SharedInfoProvider will manage the SharedInfo
        :type agents: list[AgentID]
        :param initial_state: The initial state of the environment
        :type initial_state: StateType
        :param shared_info: The previous shared information dictionary
        :type shared_info: dict[str, Any]
        :return: The modified shared information dictionary
        :rtype: dict[str, Any]
        """
        return shared_info

    def step_shared_info(
        self, agents: list[AgentID], state: StateType, shared_info: dict[str, Any]
    ) -> dict[str, Any]:
        """This function is called during the shared info step / update

        :param agents: List of AgentIDs for which
            this SharedInfoProvider should manage the SharedInfo
        :type agents: list[AgentID]
        :param state: The new state of the environment
        :type state: StateType
        :param shared_info: The previous shared information dictionary
        :type shared_info: dict[str, Any]
        :return: The modified shared information dictionary
        :rtype: dict[str, Any]
        """
        return shared_info
