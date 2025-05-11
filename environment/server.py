#!/usr/bin/env python3
"""
Turn‑based WebSocket server (handshake = CONNECT → ASSIGN_ID).

Works with websockets ≥ 12.x (no .open / .closed attributes).
"""
from __future__ import annotations

import asyncio
import json
import sys
from enum import IntEnum
from pathlib import Path
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

# --------------------------------------------------------------------------- #
# import pure game logic
# --------------------------------------------------------------------------- #
sys.path.append(str(Path(__file__).resolve().parent))
from game_core import GameCore  # noqa: E402

HOST, PORT = "127.0.0.1", 8765


class WSState(IntEnum):
    """Local alias of websockets.protocol.State values (to avoid importing
    a private enum)."""

    CONNECTING = 0
    OPEN = 1
    CLOSING = 2
    CLOSED = 3


class WormsServer:
    """
    Handles exactly one match:
      • waits until GameCore.expected_players() sockets connect
      • cycles turns until game_over()
      • stops if every player disconnects
    """

    def __init__(self) -> None:
        self.core = GameCore()
        self.clients: dict[Any, int] = {}          # websocket → player_id
        self.turn_order: list[Any] = []            # rotation of sockets
        self.game_started = False
        self.idx = 0                               # current turn index

    # --------------------------------------------------------------------- #
    # Connection‑lifetime handler
    # --------------------------------------------------------------------- #
    async def accept(self, ws: Any) -> None:
        """Handshake + linger until socket closes (without reading)."""
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(raw)
        except (asyncio.TimeoutError, json.JSONDecodeError):
            await ws.close(code=4000, reason="Expected CONNECT")
            return

        if msg.get("type") != "CONNECT":
            await ws.close(code=4002, reason="First message must be CONNECT")
            return

        # 1. register player
        pid = len(self.clients) + 1
        self.clients[ws] = pid
        self.turn_order.append(ws)
        await ws.send(json.dumps({"type": "ASSIGN_ID", "player_id": pid}))
        print(f"[server] Player {pid} connected ({msg.get('nick','?')})")

        # 2. start game once roster is full
        if (
            not self.game_started
            and len(self.turn_order) == self.core.expected_players()
        ):
            self.game_started = True
            asyncio.create_task(self.game_loop())

        # 3. keep connection alive without reading again
        try:
            await ws.wait_closed()
        finally:
            self.remove(ws)

    # --------------------------------------------------------------------- #
    # Game loop (runs once per match)
    # --------------------------------------------------------------------- #
    async def game_loop(self) -> None:
        print("[server] Game started")
        while self.turn_order:
            # wrap index if list shrank
            if self.idx >= len(self.turn_order):
                self.idx = 0

            # abort if everyone left
            if not self.turn_order:
                break

            # check for finished game
            if self.core.game_over():
                await self.broadcast({"type": "GAME_OVER", **self.core.final_info()})
                print("[server] Game over broadcasted")
                return

            ws = self.turn_order[self.idx]
            # drop socket if it has entered closing state
            if ws.state != WSState.OPEN:
                self.remove(ws, quiet=True)
                continue

            pid = self.clients[ws]

            # ------------------------------------------------ TURN_BEGIN --
            try:
                await ws.send(
                    json.dumps(
                        {
                            "type": "TURN_BEGIN",
                            "player_id": pid,
                            "state": self.core.state,
                            "time_limit_ms": 15000,
                        }
                    )
                )
            except ConnectionClosed:
                self.remove(ws)
                continue

            # ------------------------------------------------ wait ACTION --
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=15)
                msg = json.loads(raw)
                if msg.get("type") != "ACTION" or msg.get("player_id") != pid:
                    raise ValueError
            except (asyncio.TimeoutError, ValueError):
                await self.safe_send(
                    ws,
                    {"type": "ERROR", "msg": "timeout or invalid action"},
                )
                self.idx += 1
                continue
            except ConnectionClosed:
                self.remove(ws)
                continue

            # ------------------------------------------------ apply step --
            new_state, reward = self.core.step(pid, msg.get("action", {}))
            await self.broadcast(
                {
                    "type": "TURN_RESULT",
                    "player_id": pid,
                    "state": new_state,
                    "reward": reward,
                }
            )

            # ------------------------------------------------ end turn ----
            self.idx += 1
            if self.turn_order:  # may have shrunk!
                next_ws = self.turn_order[self.idx % len(self.turn_order)]
                await self.broadcast(
                    {
                        "type": "TURN_END",
                        "next_player_id": self.clients[next_ws],
                    }
                )

        print("[server] Match ended (no players left)")

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    async def broadcast(self, msg: dict) -> None:
        """Send JSON to every player; drop those that closed."""
        data = json.dumps(msg)
        dead: list[Any] = []
        for ws in list(self.turn_order):
            try:
                await ws.send(data)
            except ConnectionClosed:
                dead.append(ws)
        for ws in dead:
            self.remove(ws, quiet=True)

    async def safe_send(self, ws: Any, msg: dict) -> None:
        """Send but ignore ConnectionClosed."""
        try:
            await ws.send(json.dumps(msg))
        except ConnectionClosed:
            self.remove(ws, quiet=True)

    def remove(self, ws: Any, *, quiet: bool = False) -> None:
        """Drop socket from lists; adjust index if needed."""
        if ws in self.turn_order:
            idx = self.turn_order.index(ws)
            self.turn_order.remove(ws)
            # keep idx pointing to same logical player
            if idx <= self.idx and self.idx > 0:
                self.idx -= 1
        if ws in self.clients:
            pid = self.clients.pop(ws)
            if not quiet:
                print(f"[server] Player {pid} disconnected")


# --------------------------------------------------------------------------- #
# entry‑point
# --------------------------------------------------------------------------- #
async def main() -> None:
    async with websockets.serve(WormsServer().accept, HOST, PORT):
        print(f"Listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
