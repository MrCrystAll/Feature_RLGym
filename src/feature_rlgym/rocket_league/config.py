"""Basic configurations for Rocket League"""

from feature_rlgym.api.feature_config import FeatureConfig


def create_config() -> FeatureConfig:
    """Creates a default config with nothing

    :return: The default config
    :rtype: FeatureConfig
    """
    from feature_rlgym.api import create_config as api_create_config

    from feature_rlgym.rocket_league.empty_builder import EmptyBuilder
    from rlgym.rocket_league.action_parsers import RepeatAction, LookupTableAction

    return api_create_config(EmptyBuilder(), RepeatAction(LookupTableAction()))


def create_default_config(zero_padding: int = 3) -> FeatureConfig:
    """Creates a default config with the equivalent of the DefaultObs setup (ball, self player, padded opps and teammates and boost pads)

    :param zero_padding: Number of max cars per team, if not 0 the obs will be zero padded, defaults to 3
    :type zero_padding: int, optional
    :return: The default config
    :rtype: FeatureConfig
    """
    from feature_rlgym.rocket_league.features import (
        add_ball_feature,
        add_self_feature,
        add_others_feature,
        add_boost_pad_timers_feature,
    )

    config = create_config()
    add_ball_feature(config)
    add_self_feature(config)
    add_boost_pad_timers_feature(config)
    add_others_feature(config, zero_padding=zero_padding)

    return config
