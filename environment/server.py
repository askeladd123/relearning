#!/usr/bin/env python3
"""
Continuous W.O.R.M.S. match server.

Keeps the same WebSocket connections alive across many games so that RL clients
can train without reconnecting.

‼ 2025‑05‑14: Added MAX_TURNS_PER_GAME so a game can never run forever.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict

import websockets
from websockets.exceptions import ConnectionClosed

# ────────────────────────────────────────────────────────────────────────────────
HOST, PORT = "127.0.0.1", 8765
TIME_LIMIT_MS = 15_000
MAX_TURNS_PER_GAME = 1_000          # ← NEW ── safe upper bound for one match
# ────────────────────────────────────────────────────────────────────────────────

# make game_core importable when run as script
sys.path.append(str(Path(__file__).resolve().parent))
from game_core import GameCore

logger = logging.getLogger("server")


class WSState(IntEnum):
    CONNECTING = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3


class WormsServer:
    def __init__(self, expected_players: int, max_turns: int = MAX_TURNS_PER_GAME) -> None:
        self.expected = expected_players
        self.max_turns = max_turns
        self.core = GameCore(expected_players=self.expected)
        self.clients: dict[Any, dict[str, Any]] = {}
        self.turn_order: list[Any] = []
        self.idx = 0
        self.turn_counter = 0
        self.game_id = 0

    # ───────────────────────────── connection handling ─────────────────────────

    async def accept(self, ws: Any) -> None:
        """Handle a brand‑new WebSocket until it closes."""
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
        except (asyncio.TimeoutError, json.JSONDecodeError):
            await ws.close(code=4000, reason="Expected CONNECT")
            return

        if msg.get("type") != "CONNECT":
            await ws.close(code=4002, reason="First message must be CONNECT")
            return

        pid = len(self.clients) + 1
        nick = msg.get("nick", f"Player {pid}")
        self.clients[ws] = {"id": pid, "nick": nick}
        self.turn_order.append(ws)
        await ws.send(json.dumps({"type": "ASSIGN_ID", "player_id": pid}))
        logger.info("player %d (%s) connected", pid, nick)

        try:
            await ws.wait_closed()
        finally:
            self._remove(ws)

    def _remove(self, ws: Any) -> None:
        """Forget a websocket that closed or errored."""
        if ws in self.turn_order:
            idx = self.turn_order.index(ws)
            self.turn_order.remove(ws)
            # adjust current index if we removed an earlier entry
            if idx <= self.idx and self.idx > 0:
                self.idx -= 1
        if ws in self.clients:
            pid = self.clients.pop(ws)["id"]
            logger.info("player %d disconnected", pid)

    # ───────────────────────────── WebSocket helpers ───────────────────────────

    async def _safe_send(self, ws: Any, msg: Dict[str, Any]) -> bool:
        try:
            await ws.send(json.dumps(msg))
            return True
        except ConnectionClosed:
            self._remove(ws)
            return False

    async def _broadcast(self, msg: Dict[str, Any]) -> None:
        for ws in list(self.clients):
            await self._safe_send(ws, msg)

    # ────────────────────────── single‑game loop ───────────────────────────────

    async def _play_single_game(self) -> None:
        # fresh random turn order for this game
        self.turn_order = [ws for ws in self.clients if ws.state == WSState.OPEN]
        random.shuffle(self.turn_order)

        # reset core & counters
        self.core = GameCore(expected_players=self.expected)
        self.turn_counter = 0
        self.idx = 0
        self.game_id += 1

        initial = self.core.get_state_with_nicks(self.clients)
        await self._broadcast(
            {"type": "NEW_GAME", "game_id": self.game_id, "state": initial}
        )
        logger.info("=== GAME %d started with %d players ===",
                    self.game_id, len(self.turn_order))

        while True:
            # 1. sudden death by turn limit ─────────────────────────────────────
            if self.turn_counter >= self.max_turns:
                # pick winner: most HP strictly > others, else draw
                worms = self.core.state["worms"]
                healthiest = max(worms, key=lambda w: w["health"], default=None)
                tied = [
                    w for w in worms
                    if w["health"] == healthiest["health"]  # type: ignore[index]
                ] if healthiest else []
                winner_id = None
                if healthiest and healthiest["health"] > 0 and len(tied) == 1:
                    winner_id = healthiest["id"] + 1  # store as 1‑based
                await self._broadcast(
                    {
                        "type": "GAME_OVER",
                        "game_id": self.game_id,
                        "winner_id": winner_id,
                        "final_state": self.core.state,
                        "reason": "turn_limit",
                    }
                )
                logger.info("game %d ended by turn limit (%d turns)",
                            self.game_id, self.turn_counter)
                return
            # ───────────────────────────────────────────────────────────────────

            # 2. ordinary win condition (only one worm alive)
            alive = [w for w in self.core.state["worms"] if w["health"] > 0]
            if len(alive) <= 1:
                winner = alive[0]["id"] + 1 if alive else None
                await self._broadcast(
                    {
                        "type": "GAME_OVER",
                        "game_id": self.game_id,
                        "winner_id": winner,
                        "final_state": self.core.state,
                    }
                )
                logger.info("game %d finished naturally, winner=%s",
                            self.game_id, winner)
                return

            # 3. if no one left in turn_order (should not happen) abort game
            if not self.turn_order:
                logger.warning("no active players left, aborting game %d", self.game_id)
                return

            # 4. choose the current websocket in round‑robin fashion
            if self.idx >= len(self.turn_order):
                self.idx = 0
            ws = self.turn_order[self.idx]
            if ws.state != WSState.OPEN:
                self._remove(ws)
                continue

            pid = self.clients[ws]["id"]
            worm = self.core.state["worms"][pid - 1]

            # eliminated worm? announce once and skip its turn
            if worm["health"] <= 0:
                await self._broadcast({"type": "PLAYER_ELIMINATED", "player_id": pid})
                self.turn_order.pop(self.idx)
                continue

            # 5. TURN_BEGIN ─ send state snapshot only to current player
            begin_msg = {
                "type": "TURN_BEGIN",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": self.core.get_state_with_nicks(self.clients),
                "time_limit_ms": TIME_LIMIT_MS,
            }
            if not await self._safe_send(ws, begin_msg):
                continue  # disconnected during send

            # 6. wait for ACTION
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                msg = json.loads(raw)
                if msg.get("type") != "ACTION" or msg.get("player_id") != pid:
                    raise ValueError
            except (asyncio.TimeoutError, ValueError):
                # treat illegal / missing action as "stand"
                msg = {"action": {"action": "stand"}}
            except ConnectionClosed:
                self._remove(ws)
                continue

            # 7. apply action in core
            action = msg.get("action", {})
            new_state, reward, effects = self.core.step(pid, action)

            await self._broadcast(
                {
                    "type": "TURN_RESULT",
                    "turn_index": self.turn_counter,
                    "player_id": pid,
                    "state": new_state,
                    "reward": reward,
                    "effects": effects,
                }
            )

            # 8. advance turn
            next_idx = (self.idx + 1) % len(self.turn_order)
            next_pid = self.clients[self.turn_order[next_idx]]["id"]
            await self._broadcast({"type": "TURN_END", "next_player_id": next_pid})

            self.idx += 1
            self.turn_counter += 1

    # ───────────────────────── orchestrator: endless loop ─────────────────────

    async def orchestrator(self) -> None:
        while True:
            while len(self.clients) < self.expected:
                await asyncio.sleep(0.1)
            await self._play_single_game()


# ──────────────────────────────── main entry ──────────────────────────────────
async def main() -> None:
    parser = argparse.ArgumentParser(description="W.O.R.M.S. continuous server")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set logging level",
    )
    parser.add_argument("--max-players", type=int, default=2,
                        help="number of worms per game")
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS_PER_GAME,
                        help="hard cap on turns before declaring a draw")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    server = WormsServer(expected_players=args.max_players, max_turns=args.max_turns)
    async with websockets.serve(server.accept, HOST, PORT):
        asyncio.create_task(server.orchestrator())
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
