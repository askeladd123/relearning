#!/usr/bin/env python3
"""
Continuous W.O.R.M.S. match server.
Keeps the same WebSocket connections alive across *many* games so that RL
clients can train without reconnecting.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict

import websockets
from websockets.exceptions import ConnectionClosed

# ─── CLI / logging ──────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    parser = argparse.ArgumentParser(description="W.O.R.M.S. continuous server")
    parser.add_argument("--log-level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO",
                        help="set logging level")
    args = parser.parse_args()
    level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("server")

logger = setup_logging()

# Add ./environment to import path for game_core.py
sys.path.append(str(Path(__file__).resolve().parent))
from game_core import GameCore  # noqa: E402  pylint: disable=wrong-import-position

HOST, PORT = "127.0.0.1", 8765

class WSState(IntEnum):
    CONNECTING = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3


class WormsServer:
    """Accepts WebSockets and runs an infinite series of games."""

    # ––––– Connection handling ––––––––––––––––––––––––––––––––––––––––––––

    def __init__(self) -> None:
        self.core = GameCore()
        self.clients: dict[Any, dict[str, Any]] = {}
        self.turn_order: list[Any] = []
        self.idx = 0
        self.turn_counter = 0
        self.game_id = 0  # increments every time we start a fresh episode

    async def accept(self, ws: Any) -> None:
        """Handle the initial CONNECT handshake, then just keep the socket alive."""
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
        logger.info("new connection: player %d (%s)", pid, nick)

        try:
            await ws.wait_closed()
        finally:
            self._remove(ws)

    def _remove(self, ws: Any) -> None:
        """Forget a disconnected socket and fix the turn index."""
        if ws in self.turn_order:
            idx = self.turn_order.index(ws)
            self.turn_order.remove(ws)
            if idx <= self.idx and self.idx > 0:
                self.idx -= 1
        if ws in self.clients:
            pid = self.clients.pop(ws)["id"]
            logger.info("player %d disconnected", pid)

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

    # ––––– Game orchestration ––––––––––––––––––––––––––––––––––––––––––––––

    async def _play_single_game(self) -> None:
        """Run one episode until someone wins or all but one disconnect."""
        # fresh state ↓↓↓
        self.core = GameCore()
        self.turn_counter = 0
        self.idx = 0
        self.game_id += 1
        initial_state = self.core.get_state_with_nicks(self.clients)

        # Tell everyone a new game is starting
        await self._broadcast({
            "type": "NEW_GAME",
            "game_id": self.game_id,
            "state": initial_state,
        })

        while True:
            # End if ≤1 living worms
            alive = [w for w in self.core.state["worms"] if w["health"] > 0]
            if len(alive) <= 1:
                winner = alive[0]["id"] + 1 if alive else None
                await self._broadcast({
                    "type": "GAME_OVER",
                    "game_id": self.game_id,
                    "winner_id": winner,
                    "final_state": self.core.state,
                })
                return  # ►► back to outer loop to maybe start another episode

            # No players at all? stop the loop and wait for newcomers
            if not self.turn_order:
                return

            # Ensure idx is valid
            if self.idx >= len(self.turn_order):
                self.idx = 0

            ws = self.turn_order[self.idx]
            if ws.state != WSState.OPEN:
                self._remove(ws)
                continue

            pid = self.clients[ws]["id"]
            worm = self.core.state["worms"][pid - 1]

            # Skip dead worms
            if worm["health"] <= 0:
                await self._broadcast({
                    "type": "PLAYER_ELIMINATED",
                    "player_id": pid,
                })
                self.turn_order.pop(self.idx)
                continue

            # TURN_BEGIN
            begin = {
                "type": "TURN_BEGIN",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": self.core.get_state_with_nicks(self.clients),
                "time_limit_ms": 15000,
            }
            if not await self._safe_send(ws, begin):
                continue

            # Wait for ACTION
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                msg = json.loads(raw)
                if msg.get("type") != "ACTION" or msg.get("player_id") != pid:
                    raise ValueError
            except (asyncio.TimeoutError, ValueError):
                self.idx += 1
                self.turn_counter += 1
                continue
            except ConnectionClosed:
                self._remove(ws)
                continue

            action = msg.get("action", {})
            new_state, reward, effects = self.core.step(pid, action)
            await self._broadcast({
                "type": "TURN_RESULT",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": new_state,
                "reward": reward,
                "effects": effects,
            })

            # TURN_END
            next_idx = (self.idx + 1) % len(self.turn_order)
            next_pid = self.clients[self.turn_order[next_idx]]["id"]
            await self._broadcast({"type": "TURN_END", "next_player_id": next_pid})

            # Advance
            self.idx += 1
            self.turn_counter += 1

    async def orchestrator(self) -> None:
        """Forever wait for enough players, then run one game after another."""
        while True:
            # Wait until we have the required number of players
            while len(self.clients) < self.core.expected_players():
                await asyncio.sleep(0.1)
            await self._play_single_game()
            # Loop immediately: either players left → new game, or we go back to waiting

# ─── main entry point ───────────────────────────────────────────────────────
async def main() -> None:
    server = WormsServer()

    async with websockets.serve(server.accept, HOST, PORT):
        asyncio.create_task(server.orchestrator())
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
