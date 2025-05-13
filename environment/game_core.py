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
            dx = float(action.get("dx", 0.0))
            dx = max(-2.0, min(dx, 2.0))
            new_x = worm["x"] + dx
            worm["x"] = max(0.0, min(new_x, len(self.map[0]) - 0.01))
            col = int(math.floor(worm["x"]))
            height = len(self.map)
            OFFSET = 0.25
            row_here = int(math.floor(worm["y"]))
            if 0 <= row_here < height and self.map[row_here][col] == 1:
                worm["y"] = float(row_here) - OFFSET
            else:
                for row in range(row_here + 1, height):
                    if self.map[row][col] == 1:
                        worm["y"] = float(row) - OFFSET
                        break
                else:
                    worm["y"] = float(height)
                    worm["health"] = 0
            return copy.deepcopy(state), 0.0, effects

        if action.get("action") == "attack":
            damage_total = 0.0
            weapon = action.get("weapon")
            effects = {"weapon": weapon, "trajectory": []}

            if weapon == "kick":
                threshold, damage = 1.0, 80.0
                for other in worms:
                    if other["id"] == worm["id"] or other["health"] <= 0:
                        continue
                    dist = math.hypot(other["x"] - worm["x"], other["y"] - worm["y"])
                    if dist <= threshold:
                        other["health"] = max(0.0, other["health"] - damage)
                        damage_total += damage
                        effects["impact"] = {"x": other["x"], "y": other["y"]}
                        break
                return copy.deepcopy(state), damage_total, effects

            elif weapon == "bazooka":
                angle = math.radians(float(action.get("angle_deg", 0.0)))
                dx_unit, dy_unit = math.cos(angle), -math.sin(angle)
                x0, y0 = worm["x"], worm["y"]
                step, max_range = 0.1, 10.0
                t = step
                effects["impact"] = {}
                while t <= max_range:
                    x_proj = x0 + dx_unit * t
                    y_proj = y0 + dy_unit * t
                    effects["trajectory"].append({"x": x_proj, "y": y_proj})
                    tx, ty = int(math.floor(x_proj)), int(math.floor(y_proj))
                    if (
                        tx < 0
                        or tx >= len(self.map[0])
                        or ty < 0
                        or ty >= len(self.map)
                        or self.map[ty][tx] == 1
                    ):
                        effects["impact"] = {"x": x_proj, "y": y_proj}
                        break
                    for other in worms:
                        if other["id"] == worm["id"] or other["health"] <= 0:
                            continue
                        if math.hypot(other["x"] - x_proj, other["y"] - y_proj) <= 0.5:
                            dmg = 50.0
                            other["health"] = max(0.0, other["health"] - dmg)
                            damage_total += dmg
                            effects["impact"] = {"x": x_proj, "y": y_proj}
                            t = max_range + 1
                            break
                    t += step
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

                while True:
                    x_proj = x0 + sign * t
                    u = (t / abs(width)) if width != 0 else 0.0
                    y_proj = y0 - 4 * height * u * (1 - u)

                    effects["trajectory"].append({"x": x_proj, "y": y_proj})
                    tx, ty = int(math.floor(x_proj)), int(math.floor(y_proj))

                    if (
                        tx < 0
                        or tx >= len(self.map[0])
                        or ty < 0
                        or ty >= len(self.map)
                        or self.map[ty][tx] == 1
                    ):
                        effects["impact"] = {"x": x_proj, "y": y_proj}
                        break

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
                return copy.deepcopy(state), 0.0, {}

        return copy.deepcopy(state), 0.0, effects

    def get_state_with_nicks(self, clients: Dict[Any, Dict[str, Any]]) -> Dict[str, Any]:
        state_copy = copy.deepcopy(self.state)
        for ws, info in clients.items():
            pid = info["id"] - 1
            if 0 <= pid < len(state_copy["worms"]):
                state_copy["worms"][pid]["nick"] = info.get("nick", f"Player {pid + 1}")
        return state_copy
