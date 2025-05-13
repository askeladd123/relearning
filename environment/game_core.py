# File: environment/game_core.py
import copy
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

class GameCore:
    def __init__(self, expected_players: int = 2) -> None:
        self.expected = expected_players
        self.map = self._load_random_map()
        self.state: Dict[str, Any] = self.initial_state()

    def _load_random_map(self) -> List[List[int]]:
        maps_dir = Path(__file__).resolve().parent / "maps"
        files = list(maps_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"No map files found in {maps_dir!r}")
        choice = random.choice(files)
        with open(choice, "r") as f:
            data = json.load(f)
        width = len(data[0])
        if any(len(row) != width for row in data):
            raise ValueError(f"Map {choice.name} is not rectangular")
        return data  # type: ignore

    def expected_players(self) -> int:
        return self.expected

    def initial_state(self) -> Dict[str, Any]:
        floor = []
        rows = len(self.map)
        cols = len(self.map[0])
        for r in range(1, rows):
            for c in range(cols):
                if self.map[r][c] == 1 and self.map[r - 1][c] == 0:
                    floor.append((r, c))

        if len(floor) < self.expected:
            raise ValueError(f"Not enough floor tiles ({len(floor)}) for {self.expected} players")

        chosen = random.sample(floor, self.expected)
        worms = []
        for pid, (r, c) in enumerate(chosen):
            x = c + 0.5
            y = float(r) - 0.25
            worms.append({"id": pid, "health": 100, "x": x, "y": y})

        return {"worms": worms, "map": copy.deepcopy(self.map)}

    def step(
        self, player_id: int, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        state = self.state
        worms = state["worms"]
        idx = player_id - 1
        worm = worms[idx]
        effects: Dict[str, Any] = {}

        if action.get("action") == "walk":
            # … unchanged …
            return copy.deepcopy(state), 0.0, effects

        if action.get("action") == "attack":
            damage_total = 0.0
            weapon = action.get("weapon")
            effects = {"weapon": weapon, "trajectory": []}

            if weapon == "kick":
                # … unchanged …
                return copy.deepcopy(state), damage_total, effects

            elif weapon == "bazooka":
                # … unchanged …
                return copy.deepcopy(state), damage_total, effects

            elif weapon == "grenade":
                dx_total = float(action.get("dx", 0.0))
                sign = 1.0 if dx_total >= 0 else -1.0
                width = max(-5.0, min(dx_total, 5.0))
                height = abs(width) / 2.0
                x0, y0 = worm["x"], worm["y"]
                step_size = 0.1
                t = step_size
                effects["impact"] = {}

                # Continue until we hit terrain or a worm
                while True:
                    # XY projection
                    x_proj = x0 + sign * t
                    u = (t / abs(width)) if width != 0 else 0.0
                    # inverse-V height profile
                    h_ratio = 1.0 - abs(1.0 - 2.0 * u)
                    y_proj = y0 - height * h_ratio

                    effects["trajectory"].append({"x": x_proj, "y": y_proj})
                    tx, ty = int(math.floor(x_proj)), int(math.floor(y_proj))

                    # terrain or out-of-bounds collision
                    if (
                        tx < 0
                        or tx >= len(self.map[0])
                        or ty < 0
                        or ty >= len(self.map)
                        or self.map[ty][tx] == 1
                    ):
                        effects["impact"] = {"x": x_proj, "y": y_proj}
                        break

                    # worm collision
                    hit = False
                    for other in worms:
                        if other["id"] == worm["id"] or other["health"] <= 0:
                            continue
                        if math.hypot(other["x"] - x_proj, other["y"] - y_proj) <= 0.5:
                            dmg = 40.0
                            other["health"] = max(0.0, other["health"] - dmg)
                            damage_total += dmg
                            effects["impact"] = {"x": x_proj, "y": y_proj}
                            hit = True
                            break
                    if hit:
                        break

                    t += step_size

                return copy.deepcopy(state), damage_total, effects

            else:
                logger.warning("Unknown weapon: %s", weapon)
                effects = {}
                return copy.deepcopy(state), 0.0, effects

        return copy.deepcopy(state), 0.0, effects

    def get_state_with_nicks(self, clients: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        state_copy = copy.deepcopy(self.state)
        for ws, info in clients.items():
            pid = info["id"] - 1
            if 0 <= pid < len(state_copy["worms"]):
                state_copy["worms"][pid]["nick"] = info.get("nick", f"Player {pid + 1}")
        return state_copy
