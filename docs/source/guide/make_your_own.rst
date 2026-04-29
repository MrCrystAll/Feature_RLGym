Making a feature for an environment
===================================

Making a package for a new environment
--------------------------------------

To be able to create a package for a new environment, you need to install the API for the features.

.. code-block:: bash

    pip install features-rlgym[api]


This will give you the most important class: `Feature` (:class:`features_rlgym.api.feature.Feature`)

You can use this class to create your own feature, let's take for example, the ball feature of Rocket League:

.. literalinclude:: ../../../src/features-rlgym-rocket-league/src/features_rlgym/rocket_league/features/ball.py
    :lineno-start: 16
    :lines: 16-55


You might see in the first line that there are 5 parameters to the Feature class, these 5 arguments depend on your environment.

* AgentID: If you use only a string, you can use string, in the case of Rocket League, we can use anything as long as it is hashable.
* Obs type: This is what you expect your obs builder to return per agent.
* Action type: This is what you expect your action parser to return per agent.
* State type: This is your environment state type, GameState in the case of Rocket League.
* Obs space type: This is what you expect the obs space to be, necessary if you modify the obs, because you need to update the space size/shape.

The purpose of this feature is to add the ball to the observation, therefore, i override the `apply_to_observation_builder` function.

This function contains the current obs as an argument and expects you to return the new obs, meaning you have to handle the concatenation yourself.

This is what i do here:

.. literalinclude:: ../../../src/features-rlgym-rocket-league/src/features_rlgym/rocket_league/features/ball.py
    :lineno-start: 32
    :lines: 32-55

Since you modified the obs, you have to modify the shape as well, otherwise, the end user will be using the wrong shape and it might lead to unexpected crashes.

To accomplish that, you have to override the `get_obs_additional_size` method to return the amount of added (or removed) fields.

.. literalinclude:: ../../../src/features-rlgym-rocket-league/src/features_rlgym/rocket_league/features/ball.py
    :lineno-start: 29
    :lines: 29-30

This returns 9 because i added:

* The ball position (3 fields)
* The ball velocity (3 fields)
* The ball angular velocity (3 fields)