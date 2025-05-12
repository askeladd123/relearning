#!/usr/bin/env python3
"""Turn-based WebSocket server for W.O.R.M.S. (websockets ≥ 12.x)."""
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

# ─── Define custom TRACE level ─────────────────────────────────────
TRACE_LEVEL_NUM = 5
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")
logging.TRACE = TRACE_LEVEL_NUM                      # make logging.TRACE available
def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)
logging.Logger.trace = trace

# ─── Parse --log-level & configure ────────────────────────────────
def setup_logging() -> logging.Logger:
    parser = argparse.ArgumentParser(description="W.O.R.M.S. server")
    parser.add_argument(
        "--log-level",
        choices=["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
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

# ─── Import game logic ─────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent))
from game_core import GameCore  # noqa: E402

HOST, PORT = "127.0.0.1", 8765

class WSState(IntEnum):
    CONNECTING = 0
    OPEN       = 1
    CLOSING    = 2
    CLOSED     = 3

class WormsServer:
    def __init__(self) -> None:
        self.core = GameCore()
        self.clients: dict[Any, int] = {}
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
        self.clients[ws] = pid
        self.turn_order.append(ws)
        await ws.send(json.dumps({"type": "ASSIGN_ID", "player_id": pid}))
        logger.info("new connection: player %d (%s)", pid, msg.get("nick", "?"))

        if not self.game_started and len(self.turn_order) == self.core.expected_players():
            self.game_started = True
            logger.info("roster full (%d players), starting game", len(self.turn_order))
            asyncio.create_task(self.game_loop())

        try:
            await ws.wait_closed()
        finally:
            self.remove(ws)

    async def game_loop(self) -> None:
        logger.info("game loop started")
        while self.turn_order:
            if self.idx >= len(self.turn_order):
                self.idx = 0

            if self.core.game_over():
                await self.broadcast({"type": "GAME_OVER", **self.core.final_info()})
                logger.info("game over")
                return

            ws = self.turn_order[self.idx]
            if ws.state != WSState.OPEN:
                self.remove(ws, quiet=True)
                continue

            pid = self.clients[ws]
            begin_msg = {
                "type": "TURN_BEGIN",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": self.core.state,
                "time_limit_ms": 15_000,
            }
            logger.trace("→ TURN_BEGIN to player %d", pid)
            if not await self.safe_send(ws, begin_msg):
                continue

            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                logger.trace("← raw message from %d: %s", pid, raw)
                msg = json.loads(raw)
                if msg.get("type") != "ACTION" or msg.get("player_id") != pid:
                    raise ValueError
            except (asyncio.TimeoutError, ValueError):
                await self.safe_send(ws, {"type": "ERROR", "msg": "timeout or invalid action"})
                logger.warning("player %d timed out or sent invalid action", pid)
                self.idx += 1
                self.turn_counter += 1
                continue
            except ConnectionClosed:
                self.remove(ws)
                continue

            action = msg.get("action", {})
            new_state, reward = self.core.step(pid, action)
            logger.trace("applied action %r for player %d → reward %.2f", action, pid, reward)

            await self.broadcast({
                "type": "TURN_RESULT",
                "turn_index": self.turn_counter,
                "player_id": pid,
                "state": new_state,
                "reward": reward,
            })

            self.idx += 1
            self.turn_counter += 1
            if self.turn_order:
                next_ws = self.turn_order[self.idx % len(self.turn_order)]
                await self.broadcast({
                    "type": "TURN_END",
                    "next_player_id": self.clients[next_ws],
                })

        logger.info("match ended (no players left)")

    async def broadcast(self, msg: dict) -> None:
        data = json.dumps(msg)
        logger.trace("broadcasting %s", msg.get("type"))
        dead: list[Any] = []
        for ws in list(self.turn_order):
            try:
                await ws.send(data)
            except ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            self.remove(ws, quiet=True)

    async def safe_send(self, ws: Any, msg: dict) -> bool:
        try:
            await ws.send(json.dumps(msg))
            return True
        except ConnectionClosed:
            self.remove(ws, quiet=True)
            return False

    def remove(self, ws: Any, *, quiet: bool = False) -> None:
        if ws in self.turn_order:
            idx = self.turn_order.index(ws)
            self.turn_order.remove(ws)
            if idx <= self.idx and self.idx > 0:
                self.idx -= 1
        if ws in self.clients:
            pid = self.clients.pop(ws)
            if not quiet:
                logger.info("player %d disconnected", pid)

async def main() -> None:
    async with websockets.serve(WormsServer().accept, HOST, PORT):
        logger.info("Listening on ws://%s:%d", HOST, PORT)
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
