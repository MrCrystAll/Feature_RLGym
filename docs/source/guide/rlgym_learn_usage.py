def build_rlgym_v2_env():
    from features_rlgym.rocket_league.config import create_config
    from features_rlgym.rocket_league.features import add_ball_pred_feature

    from rlgym.api import RLGym

    from rlgym.rocket_league.state_mutators import (
        MutatorSequence,
        KickoffMutator,
        FixedTeamSizeMutator,
    )
    from rlgym.rocket_league.reward_functions import GoalReward
    from rlgym.rocket_league.sim import RocketSimEngine

    config = create_config()
    add_ball_pred_feature(config)

    return RLGym(
        obs_builder=config.obs_builder,
        action_parser=config.action_parser,
        shared_info_provider=config.shared_info_provider,
        state_mutator=MutatorSequence(FixedTeamSizeMutator(2, 2), KickoffMutator()),
        reward_fn=GoalReward(),
        transition_engine=RocketSimEngine(),
    )


if __name__ == "__main__":
    # All your config, see https://github.com/JPK314/rlgym-learn
    # for more details about rlgym-learn
    config = {}

    from rlgym_learn import LearningCoordinator

    learning_coordinator = LearningCoordinator(
        build_rlgym_v2_env,
        agent_controllers={
            # Your agents
            ...
        },
        config=config,
    )
