#!/usr/bin/env python3
"""Simple random bot for W.O.R.M.S."""
import argparse
import asyncio
import json
import logging
import random

import websockets

def setup_logging() -> logging.Logger:
    parser = argparse.ArgumentParser(
        description="W.O.R.M.S. bot client"
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
    return logging.getLogger("client")

logger = setup_logging()

HOST, PORT = "127.0.0.1", 8765

ACTIONS = [
    {"action": "stand"},
    {"action": "walk", "dx": 1.0},
    {"action": "attack", "weapon": "kick", "force": 70.0},
    {"action": "attack", "weapon": "bazooka", "angle_deg": 90.0},
    {"action": "attack", "weapon": "grenade", "angle_deg": 30.0, "force": 50.0},
]

async def start_client() -> None:
    uri = f"ws://{HOST}:{PORT}"
    logger.info("connecting to %s", uri)
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "CONNECT", "nick": "bot"}))
        logger.info("sent CONNECT")
        player_id: int | None = None
        async for message in ws:
            msg = json.loads(message)
            t = msg.get("type")

            if t == "ASSIGN_ID":
                player_id = msg["player_id"]
                logger.info("assigned player_id=%d", player_id)
                continue

            if t == "PLAYER_ELIMINATED":
                if msg.get("player_id") == player_id:
                    logger.info("I have been eliminated, closing client")
                    await ws.close()
                    break
                continue

            if t == "TURN_BEGIN" and msg.get("player_id") == player_id:
                action = random.choice(ACTIONS)
                payload = {"type": "ACTION", "player_id": player_id, "action": action}
                await ws.send(json.dumps(payload))
                logger.debug("did action: %r", payload)

if __name__ == "__main__":
    asyncio.run(start_client())
