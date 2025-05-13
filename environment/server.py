#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

def setup_logging() -> logging.Logger:
    parser = argparse.ArgumentParser(
        description="W.O.R.M.S. server"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set logging level"
    )
    args = parser.parse_args()
    level = getattr(logging, args.log_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("server")

logger = setup_logging()

sys.path.append(str(Path(__file__).resolve().parent))
from game_core import GameCore

HOST, PORT = "127.0.0.1", 8765

class WSState(IntEnum):
    CONNECTING = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3

class WormsServer:
    def __init__(self) -> None:
        self.core = GameCore()
        self.clients: dict[Any, dict[str, Any]] = {}
        self.turn_order: list[Any] = []
        self.idx = 0
        self.turn_counter = 0

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
        logger.info("new connection: player %d (%s)", pid, nick)

        try:
            await ws.wait_closed()
        finally:
            self.remove(ws)

    def remove(self, ws: Any) -> None:
        if ws in self.turn_order:
            idx = self.turn_order.index(ws)
            self.turn_order.remove(ws)
            if idx <= self.idx and self.idx > 0:
                self.idx -= 1
        if ws in self.clients:
            pid = self.clients.pop(ws)["id"]
            logger.info("player %d disconnected", pid)

    async def game_loop(self) -> None:
        while True:
            # End game when ≤1 alive
            alive = [w for w in self.core.state["worms"] if w["health"] > 0]
            if len(alive) <= 1:
                winner = alive[0]["id"] + 1 if alive else None
                await self.broadcast({
                    "type": "GAME_OVER",
                    "winner_id": winner,
                    "final_state": self.core.state
                })
                return

            if not self.turn_order:
                return
            if self.idx >= len(self.turn_order):
                self.idx = 0

            ws = self.turn_order[self.idx]
            if ws.state != WSState.OPEN:
                self.remove(ws)
                continue

            pid = self.clients[ws]["id"]
            worm = self.core.state["worms"][pid - 1]

            # If dead, emit elimination and skip
            if worm["health"] <= 0:
                await self.broadcast({"type": "PLAYER_ELIMINATED", "player_id": pid})
                self.turn_order.pop(self.idx)
                continue

            # TURN_BEGIN
            begin = {
                "type": "TURN_BEGIN",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": self.core.get_state_with_nicks(self.clients),
                "time_limit_ms": 15000
            }
            if not await self.safe_send(ws, begin):
                continue

            # Await ACTION
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
                self.remove(ws)
                continue

            # Apply and broadcast result
            action = msg.get("action", {})
            new_state, reward, effects = self.core.step(pid, action)
            await self.broadcast({
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
            await self.broadcast({"type": "TURN_END", "next_player_id": next_pid})

            self.idx += 1
            self.turn_counter += 1

    async def broadcast(self, msg: dict) -> None:
        for ws in list(self.clients):
            await self.safe_send(ws, msg)

    async def safe_send(self, ws: Any, msg: dict) -> bool:
        try:
            await ws.send(json.dumps(msg))
            return True
        except ConnectionClosed:
            self.remove(ws)
            return False

async def main() -> None:
    server = WormsServer()

    # Once we've got enough players, kick off the game loop
    async def starter():
        while len(server.turn_order) < server.core.expected_players():
            await asyncio.sleep(0.1)
        asyncio.create_task(server.game_loop())

    async with websockets.serve(server.accept, HOST, PORT):
        asyncio.create_task(starter())
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
