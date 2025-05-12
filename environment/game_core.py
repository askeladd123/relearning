import copy
import logging
import math
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
        state = self.state
        worms = state["worms"]
        idx = player_id - 1
        worm = worms[idx]

        if action.get("action") == "walk":
            dx = float(action.get("dx", 0.0))
            new_x = worm["x"] + dx
            # clamp x within map bounds
            max_x = len(state["map"][0]) - 0.01
            worm["x"] = max(0.0, min(new_x, max_x))

            # apply gravity instantly after walking
            col = int(math.floor(worm["x"]))
            height = len(state["map"])
            found_ground = False
            # search downward for the first solid tile
            for row in range(int(math.floor(worm["y"])) + 1, height):
                if state["map"][row][col] == 1:
                    # land on top of this tile
                    worm["y"] = float(row)
                    found_ground = True
                    break
            if not found_ground:
                # fell off bottom into water -> dead
                worm["y"] = float(height)
                worm["health"] = 0

        return copy.deepcopy(state), 0.0

    def get_state_with_nicks(self, clients: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        state_copy = copy.deepcopy(self.state)
        for ws, info in clients.items():
            pid = info["id"] - 1
            if 0 <= pid < len(state_copy["worms"]):
                state_copy["worms"][pid]["nick"] = info.get("nick", f"Player {pid + 1}")
        return state_copy

    def game_over(self) -> bool:
        return False

    def final_info(self) -> Dict[str, Any]:
        return {"winner_id": 0, "final_state": self.state}
