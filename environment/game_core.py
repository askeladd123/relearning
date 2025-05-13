# File: environment/game_core.py
import copy
import logging
import math
from typing import Any, Dict, Tuple, List

logger = logging.getLogger(__name__)


class GameCore:
    def __init__(self) -> None:
        self.state: Dict[str, Any] = self.initial_state()

    # ──────────────────────────────────────────────────────────────────────
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

    def expected_players(self) -> int:  # unchanged
        return 2

    # ──────────────────────────────────────────────────────────────────────
    def step(
        self, player_id: int, action: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
        """
        Advance the game by one action and return (new_state, reward, effects).

        Only the **grenade** math differs from the original version:
        it now takes a single `dx` world‑unit parameter and follows an
        inverse‑V path whose peak height is |dx|/2.
        """
        state = self.state
        worms = state["worms"]
        idx = player_id - 1
        worm = worms[idx]

        effects: Dict[str, Any] = {}

        # ─── Movement ────────────────────────────────────────────────────
        if action.get("action") == "walk":
            dx = float(action.get("dx", 0.0))
            new_x = worm["x"] + dx
            worm["x"] = max(0.0, min(new_x, len(state["map"][0]) - 0.01))

            # gravity (very simple: fall straight down until terrain)
            col = int(math.floor(worm["x"]))
            height = len(state["map"])
            for row in range(int(math.floor(worm["y"])) + 1, height):
                if state["map"][row][col] == 1:
                    worm["y"] = float(row)
                    break
            else:  # fell into water
                worm["y"] = float(height)
                worm["health"] = 0

            return copy.deepcopy(state), 0.0, effects

        # ─── Attack ──────────────────────────────────────────────────────
        if action.get("action") == "attack":
            damage_total = 0.0
            weapon = action.get("weapon")
            effects["weapon"] = weapon
            effects["trajectory"] = []  # type: List[Dict[str, float]]
            effects["impact"] = {}

            # — Kick (unchanged) —
            if weapon == "kick":
                threshold = 1.0
                force = float(action.get("force", 0.0))
                for other in worms:
                    if other["id"] == worm["id"] or other["health"] <= 0:
                        continue
                    dist = math.hypot(other["x"] - worm["x"], other["y"] - worm["y"])
                    if dist <= threshold:
                        other["health"] = max(0.0, other["health"] - force)
                        damage_total += force
                effects = {}

            # — Bazooka (unchanged) —
            elif weapon == "bazooka":
                angle = math.radians(float(action.get("angle_deg", 0.0)))
                dx = math.cos(angle)
                dy = -math.sin(angle)
                x0, y0 = worm["x"], worm["y"]
                step_size = 0.1
                max_range = 10.0
                t = step_size
                while t <= max_range:
                    x_proj = x0 + dx * t
                    y_proj = y0 + dy * t
                    effects["trajectory"].append({"x": x_proj, "y": y_proj})

                    tile_x = int(math.floor(x_proj))
                    tile_y = int(math.floor(y_proj))
                    if (
                        tile_x < 0
                        or tile_x >= len(state["map"][0])
                        or tile_y < 0
                        or tile_y >= len(state["map"])
                        or state["map"][tile_y][tile_x] == 1
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
                            t = max_range + 1  # stop outer loop
                            break
                    t += step_size

            # — Grenade (new inverse‑V, takes only dx) —
            elif weapon == "grenade":
                dx_total = float(action.get("dx", 0.0))
                if dx_total == 0.0:
                    # explode in place if someone sends dx=0
                    effects["impact"] = {"x": worm["x"], "y": worm["y"]}
                else:
                    x0, y0 = worm["x"], worm["y"]
                    sign = 1.0 if dx_total > 0 else -1.0
                    width = abs(dx_total)
                    height = width / 2.0  # agreed peak rule

                    step_size = 0.1
                    t = step_size
                    while t <= width + 1e-6:  # include end point
                        # horizontal progress 0…width
                        x_proj = x0 + sign * t
                        u = t / width  # 0…1
                        h_ratio = 1.0 - abs(1.0 - 2.0 * u)  # 0→1→0 triangular
                        y_proj = y0 - height * h_ratio       # up = smaller y
                        effects["trajectory"].append({"x": x_proj, "y": y_proj})

                        tile_x = int(math.floor(x_proj))
                        tile_y = int(math.floor(y_proj))

                        # out of bounds or hits terrain
                        if (
                            tile_x < 0
                            or tile_x >= len(state["map"][0])
                            or tile_y < 0
                            or tile_y >= len(state["map"])
                            or state["map"][tile_y][tile_x] == 1
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

            else:
                logger.warning("Unknown weapon: %s", weapon)
                effects = {}

            return copy.deepcopy(state), damage_total, effects

        # ─── All other actions ────────────────────────────────────────────
        return copy.deepcopy(state), 0.0, effects

    # ──────────────────────────────────────────────────────────────────────
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

