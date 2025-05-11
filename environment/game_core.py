# Pure game logic (no networking)

from typing import Any, Dict, Tuple


class GameCore:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = self.initial_state()

    def initial_state(self) -> Dict[str, Any]:
        return {
            "worms": [
                {"id": 0, "health": 100, "x": 1, "y": 1},
                {"id": 1, "health": 100, "x": 6, "y": 2},
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
        reward = 0.0
        # TODO: apply real rules here
        return self.state, reward

    def game_over(self) -> bool:
        return False

    def final_info(self) -> Dict[str, Any]:
        return {"winner_id": 0, "final_state": self.state}
