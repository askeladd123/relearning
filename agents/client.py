#!/usr/bin/env python3
"""
Reference bot that now keeps its WebSocket open across many games.
It rests while eliminated and wakes up when NEW_GAME arrives.
"""
import argparse
import asyncio
import json
import logging
import random

import websockets

# ─── CLI / logging ──────────────────────────────────────────────────────────
def setup_logging() -> logging.Logger:
    parser = argparse.ArgumentParser(description="W.O.R.M.S. bot client")
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
    return logging.getLogger("client")

logger = setup_logging()

HOST, PORT = "127.0.0.1", 8765

ACTIONS = [
    {"action": "stand"},
    {"action": "walk", "dx": 1.0},
    {"action": "attack", "weapon": "kick", "force": 70.0},
    {"action": "attack", "weapon": "bazooka", "angle_deg": 90.0},
    {"action": "attack", "weapon": "grenade", "dx": 4.0},  # ⇐ new signature
]

async def start_client() -> None:
    uri = f"ws://{HOST}:{PORT}"
    logger.info("connecting to %s", uri)

    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "CONNECT", "nick": "bot"}))
        player_id: int | None = None
        eliminated = False

        async for raw in ws:
            msg = json.loads(raw)
            t = msg.get("type")

            if t == "ASSIGN_ID":
                player_id = msg["player_id"]
                logger.info("assigned player_id=%d", player_id)

            elif t == "PLAYER_ELIMINATED" and msg.get("player_id") == player_id:
                eliminated = True
                logger.info("I have been eliminated this game")

            elif t == "NEW_GAME":
                eliminated = False
                logger.info("new episode %s started – back in the game!", msg.get("game_id"))

            elif (t == "TURN_BEGIN"
                  and msg.get("player_id") == player_id
                  and not eliminated):
                action = random.choice(ACTIONS)
                payload = {"type": "ACTION", "player_id": player_id, "action": action}
                await ws.send(json.dumps(payload))
                logger.debug("did action: %r", payload)

if __name__ == "__main__":
    asyncio.run(start_client())
