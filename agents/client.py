# File: agents/client.py
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

def setup_logging() -> logging.Logger:
    parser = argparse.ArgumentParser(description="W.O.R.M.S. bot client")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set logging level",
    )
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

# List of possible actions
ACTIONS = [
    {"action": "stand"},
    {"action": "walk", "dx": 1.0},
    {"action": "attack", "weapon": "kick"},
    {"action": "attack", "weapon": "bazooka", "angle_deg": 30.0},
    {"action": "attack", "weapon": "grenade", "dx": 2.0},
]

# Specify a probability for each action; must sum to 1.0
# e.g. make "walk" less likely by giving it a smaller weight.
ACTION_PROBS = [
    0.10,  # stand
    0.05,  # walk
    0.25,  # kick
    0.30,  # bazooka
    0.30,  # grenade
]

# Validate at import time
if len(ACTION_PROBS) != len(ACTIONS):
    raise ValueError(
        f"ACTION_PROBS length ({len(ACTION_PROBS)}) does not match "
        f"ACTIONS length ({len(ACTIONS)})"
    )
total_prob = sum(ACTION_PROBS)
if abs(total_prob - 1.0) > 1e-8:
    raise ValueError(f"ACTION_PROBS must sum to 1.0, but sum to {total_prob}")

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
                logger.info(
                    "new episode %s started – back in the game!",
                    msg.get("game_id"),
                )

            elif t == "TURN_BEGIN" and msg.get("player_id") == player_id and not eliminated:
                # pick an action according to ACTION_PROBS
                action = random.choices(ACTIONS, weights=ACTION_PROBS, k=1)[0]
                payload = {"type": "ACTION", "player_id": player_id, "action": action}
                await ws.send(json.dumps(payload))
                logger.debug("did action: %r", payload)

            elif t == "TURN_RESULT" and msg.get("player_id") == player_id:
                reward = msg.get("reward", 0.0)
                logger.info("received reward=%.1f", reward)

if __name__ == "__main__":
    asyncio.run(start_client())
