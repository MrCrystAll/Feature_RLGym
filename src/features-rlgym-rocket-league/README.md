# Features RLGym | Rocket League

This package contains built-in features for Rocket League, this allows you to add stuff to your bot without having to explore whatever the creator modified to make it work, just add the feature and your bot will have it.

## Example - Adding ball prediction

You want to add ball prediction to your bot? Simple.

```python
from features_rlgym.rocket_league.features import add_ball_pred_feature
from features_rlgym.api import create_config

from rlgym.api import RLGym

from my_obs_builder import MyObsBuilder # Whatever it is
from my_action_parser import MyActionParser # Whatever it is

# Declare the builder and action parser
obs_builder = MyObsBuilder()
action_parser = MyActionParser()

# Create the configuration with those 2
config = create_config(obs_builder, action_parser)

add_ball_pred_feature(config)

env = RLGym(
    obs_builder=config.obs_builder,
    action_parser=config.action_parser,
    shared_info_provider=config.shared_info_provider
    ... # The rest of your objects (Rewards, etc.)
)

# And that's it !
```
Your bot now has ball prediction in its obs and can act accordingly.

Q: What if i want to modify the ball prediction parameters ?

A: Simple, the `add_ball_pred_feature` accepts multiple arguments, all of them have default values that you can change at will. You can use your autocomplete or the documentation to discover them.
