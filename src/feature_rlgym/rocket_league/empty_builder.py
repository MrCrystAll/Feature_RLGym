from collections.abc import Hashable
from typing import Any, Dict, List

import numpy as np
from rlgym.api import ObsBuilder
from rlgym.rocket_league.api import GameState


class EmptyBuilder(ObsBuilder[Hashable, np.ndarray, GameState, int]):
    def build_obs(
        self, agents: List[Hashable], state: GameState, shared_info: Dict[str, Any]
    ) -> Dict[Hashable, np.ndarray]:
        return {agent: np.asarray([]) for agent in agents}

    def reset(
        self,
        agents: List[Hashable],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_obs_space(self, agent: Hashable) -> int:
        return 0
