# Barebone example of how to use the package

# =======================================================================

# These are Rocket League specific imports
# but the same can be applied for other environments if you made a package
# using the base api, or if another environment is supported

from features_rlgym.rocket_league.config import create_config

# I choose ball prediction here because
# it's the only feature (for now) that isn't in the default obs builder
from features_rlgym.rocket_league.features import add_ball_pred_feature

# You need this to create the environment
from rlgym.api import RLGym

from rlgym.rocket_league.state_mutators import (
    MutatorSequence,
    KickoffMutator,
    FixedTeamSizeMutator,
)
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine

config = create_config()  # Creates a basic obs builder and action parser
add_ball_pred_feature(config)  # Adds the ball prediction to the obs builder

env = RLGym(
    obs_builder=config.obs_builder,  # Set the configured obs
    action_parser=config.action_parser,  # Set the configured action parser
    shared_info_provider=config.shared_info_provider,  # Set the configured shared info provider
    # ------------------------ This is the rest of your environment config ------------------------
    state_mutator=MutatorSequence(FixedTeamSizeMutator(2, 2), KickoffMutator()),
    reward_fn=GoalReward(),
    transition_engine=RocketSimEngine(),
)

# Do whatever you want with the environment
