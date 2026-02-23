if __name__ == "__main__":
    import time

    import numpy as np

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

    from feature_rlgym.rocket_league import create_default_config
    from feature_rlgym.rocket_league.features import add_ball_pred_feature

    config = create_default_config()

    add_ball_pred_feature(config)

    env = RLGym(
        obs_builder=config.obs_builder,
        action_parser=config.action_parser,
        reward_fn=TouchReward(),
        termination_cond=GoalCondition(),
        truncation_cond=NoopCondition(),
        transition_engine=RocketSimEngine(),
        state_mutator=MutatorSequence(
            FixedTeamSizeMutator(2, 2), RandomPhysicsMutator()
        ),
        renderer=RocketSimVisRenderer(),
        shared_info_provider=config.shared_info_provider,
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
