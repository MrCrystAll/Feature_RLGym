from typing import Generic

from rlgym.api import (
    ObsType,
    ActionType,
    ObsSpaceType,
    ActionSpaceType,
    AgentID,
    EngineActionType,
    StateType,
    ObsBuilder,
    ActionParser,
)

from feature_rlgym.api.configurable_obs_builder import ConfigurableObsBuilder
from feature_rlgym.api.configurable_shared_info_provider import (
    ConfigurableSharedInfoProvider,
)


class FeatureConfig(
    Generic[
        AgentID,
        ObsType,
        ActionType,
        ObsSpaceType,
        ActionSpaceType,
        EngineActionType,
        StateType,
    ]
):
    def __init__(
        self,
        obs_builder: ObsBuilder[AgentID, ObsType, StateType, ObsSpaceType],
        action_parser: ActionParser[
            AgentID, ActionType, EngineActionType, StateType, ActionSpaceType
        ],
    ) -> None:
        self._base_obs_builder = obs_builder
        self.obs_builder = ConfigurableObsBuilder(obs_builder)

        self._base_action_parser = action_parser
        self.action_parser = action_parser

        self.shared_info_provider = ConfigurableSharedInfoProvider()


def create_config(obs_builder: ObsBuilder, act_parser: ActionParser) -> FeatureConfig:
    """Creates a config with your obs builder and action parser

    :param obs_builder: Your obs builder
    :type obs_builder: ObsBuilder
    :param act_parser: Your action parser
    :type act_parser: ActionParser
    :return: A config where you can add features
    :rtype: FeatureConfig
    """
    return FeatureConfig(obs_builder, act_parser)
