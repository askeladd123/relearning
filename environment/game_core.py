# environment/game_core.py

import copy
import logging
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

class GameCore:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = self.initial_state()

    def initial_state(self) -> Dict[str, Any]:
        return {
            "worms": [
                {"id": 0, "health": 100, "x": 3.20, "y": 1.75},
                {"id": 1, "health": 100, "x": 6.00, "y": 2.00},
            ],
            "map": [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0, 1, 1, 1],
            ],
        }

    def expected_players(self) -> int:
        return 2

    def step(self, player_id: int, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        logger.trace("GameCore.step: player_id=%d action=%r", player_id, action)
        state = self.state
        worms = state["worms"]
        grid = state["map"]
        rows = len(grid)
        cols = len(grid[0])

        idx = player_id - 1
        worm = worms[idx]

        if action.get("action") == "walk":
            dx = float(action.get("dx", 0.0))
            new_x = worm["x"] + dx
            new_x = max(0.0, min(new_x, cols - 1e-6))
            ty = int(worm["y"])
            tx = int(new_x)
            if 0 <= ty < rows and grid[ty][tx] == 0:
                worm["x"] = new_x

        return copy.deepcopy(state), 0.0

    def game_over(self) -> bool:
        return False

    def final_info(self) -> Dict[str, Any]:
        return {"winner_id": 0, "final_state": self.state}
