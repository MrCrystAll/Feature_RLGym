import time

import numpy as np

from feature_rlgym.api.feature_config import create_config
from feature_rlgym.rocket_league.empty_builder import EmptyBuilder

from rlgym.rocket_league.action_parsers import LookupTableAction
from rlgym.rocket_league.reward_functions import TouchReward
from rlgym.rocket_league.done_conditions import GoalCondition, NoopCondition
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    MutatorSequence,
)

from rlgym.api import RLGym

from rlgym_tools.rocket_league.renderers.rocketsimvis_renderer import (
    RocketSimVisRenderer,
)
from rlgym_tools.rocket_league.state_mutators.random_physics_mutator import (
    RandomPhysicsMutator,
)

from feature_rlgym.rocket_league.features import (
    add_ball_feature,
    add_self_feature,
    add_boost_pad_timers_feature,
    add_others_feature,
)

if __name__ == "__main__":
    obs_builder = EmptyBuilder()
    act_parser = LookupTableAction()

    config = create_config(obs_builder, act_parser)

    add_ball_feature(config)
    add_self_feature(config)
    add_boost_pad_timers_feature(config)
    add_others_feature(config)

    env = RLGym(
        obs_builder=config.obs_builder,
        action_parser=config.action_parser,
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=NoopCondition(),
        transition_engine=RocketSimEngine(),
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(3, 3), RandomPhysicsMutator()
        ),
        renderer=RocketSimVisRenderer(),
    )

    running = True

    print("Running env")
    while running:
        try:
            env.reset()

            _obs_space = config.obs_builder.get_obs_space(None)

            print(_obs_space)

            truncated = {agent: False for agent in env.agents}
            terminated = {agent: False for agent in env.agents}

            while not (any(truncated.values()) or any(terminated.values())):
                env.render()
                time.sleep(1.0 / 120.0)

                agent_actions = {agent: np.asarray([89]) for agent in env.agents}

                obs, _, terminated, truncated = env.step(agent_actions)

                print(obs[env.agents[0]].shape)

        except KeyboardInterrupt:
            print("Stopping")
            running = False
            break

    env.close()
