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

# Removed custom TRACE level; using built-in logging levels

def setup_logging() -> logging.Logger:
    parser = argparse.ArgumentParser(description="W.O.R.M.S. server")
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
        self.game_started = False

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

        if not self.game_started and len(self.turn_order) == self.core.expected_players():
            self.game_started = True
            asyncio.create_task(self.game_loop())

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
        while self.turn_order:
            if self.idx >= len(self.turn_order):
                self.idx = 0

            ws = self.turn_order[self.idx]
            if ws.state != WSState.OPEN:
                self.remove(ws)
                continue

            pid = self.clients[ws]["id"]
            nick = self.clients[ws]["nick"]

            begin_msg = {
                "type": "TURN_BEGIN",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": self.core.get_state_with_nicks(self.clients),
                "time_limit_ms": 15000
            }

            if not await self.safe_send(ws, begin_msg):
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
                self.remove(ws)
                continue

            action = msg.get("action", {})
            new_state, reward = self.core.step(pid, action)

            await self.broadcast({
                "type": "TURN_RESULT",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": new_state,
                "reward": reward
            })

            self.idx += 1
            self.turn_counter += 1

    async def broadcast(self, msg: dict) -> None:
        for ws in list(self.turn_order):
            await self.safe_send(ws, msg)

    async def safe_send(self, ws: Any, msg: dict) -> bool:
        try:
            await ws.send(json.dumps(msg))
            return True
        except ConnectionClosed:
            self.remove(ws)
            return False

async def main() -> None:
    async with websockets.serve(WormsServer().accept, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
