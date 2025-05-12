# Pure game logic (no networking, world‑unit coordinates)

from typing import Any, Dict, Tuple
import copy


class GameCore:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = self.initial_state()

    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    def expected_players(self) -> int:
        return 2

    # ------------------------------------------------------------------ #
    def step(self, player_id: int, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
        """Dummy step – just echo state, zero reward."""
        # TODO: implement physics & rules here
        return copy.deepcopy(self.state), 0.0

    # ------------------------------------------------------------------ #
    def game_over(self) -> bool:
        return False

    def final_info(self) -> Dict[str, Any]:
        return {"winner_id": 0, "final_state": self.state}
