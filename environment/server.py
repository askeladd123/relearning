# File: environment/server.py
#!/usr/bin/env python3
"""
Continuous W.O.R.M.S. match server.
Keeps the same WebSocket connections alive across many games so that RL clients can train without reconnecting.
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

# ensure game_core is importable
sys.path.append(str(Path(__file__).resolve().parent))
from game_core import GameCore

HOST, PORT = "127.0.0.1", 8765

class WSState(IntEnum):
    CONNECTING = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3

class WormsServer:
    def __init__(self, expected_players: int) -> None:
        self.expected = expected_players
        # initial core only used until first game start
        self.core = GameCore(expected_players=self.expected)
        self.clients: dict[Any, dict[str, Any]] = {}
        self.turn_order: list[Any] = []
        self.idx = 0
        self.turn_counter = 0
        self.game_id = 0

    async def accept(self, ws: Any) -> None:
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

        try:
            await ws.wait_closed()
        finally:
            self._remove(ws)

    def _remove(self, ws: Any) -> None:
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

    async def _play_single_game(self) -> None:
        # --- ← new game start: force a fresh map & state each time
        self.core = GameCore(expected_players=self.expected)
        self.turn_counter = 0
        self.idx = 0
        self.game_id += 1

        initial = self.core.get_state_with_nicks(self.clients)
        await self._broadcast({
            "type": "NEW_GAME",
            "game_id": self.game_id,
            "state": initial,
        })

        while True:
            alive = [w for w in self.core.state["worms"] if w["health"] > 0]
            if len(alive) <= 1:
                winner = alive[0]["id"] + 1 if alive else None
                await self._broadcast({
                    "type": "GAME_OVER",
                    "game_id": self.game_id,
                    "winner_id": winner,
                    "final_state": self.core.state,
                })
                return

            if not self.turn_order:
                return

            if self.idx >= len(self.turn_order):
                self.idx = 0

            ws = self.turn_order[self.idx]
            if ws.state != WSState.OPEN:
                self._remove(ws)
                continue

            pid = self.clients[ws]["id"]
            worm = self.core.state["worms"][pid - 1]

            if worm["health"] <= 0:
                await self._broadcast({
                    "type": "PLAYER_ELIMINATED",
                    "player_id": pid,
                })
                self.turn_order.pop(self.idx)
                continue

            begin = {
                "type": "TURN_BEGIN",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": self.core.get_state_with_nicks(self.clients),
                "time_limit_ms": 15000,
            }
            if not await self._safe_send(ws, begin):
                continue

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

            next_idx = (self.idx + 1) % len(self.turn_order)
            next_pid = self.clients[self.turn_order[next_idx]]["id"]
            await self._broadcast({"type": "TURN_END", "next_player_id": next_pid})

            self.idx += 1
            self.turn_counter += 1

    async def orchestrator(self) -> None:
        while True:
            while len(self.clients) < self.expected:
                await asyncio.sleep(0.1)
            await self._play_single_game()

async def main() -> None:
    parser = argparse.ArgumentParser(description="W.O.R.M.S. continuous server")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set logging level",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=2,
        help="number of worms per game",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    global logger
    logger = logging.getLogger("server")

    server = WormsServer(expected_players=args.max_players)
    async with websockets.serve(server.accept, HOST, PORT):
        asyncio.create_task(server.orchestrator())
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
