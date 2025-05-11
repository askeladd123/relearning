import asyncio
import json
import random

import websockets

HOST, PORT = "127.0.0.1", 8765
ACTIONS = [
    {"action": "stand"},
    {"action": "walk", "amount-x": 12.0},
    {"action": "attack", "weapon": "kick", "force": 70.0},
    {"action": "attack", "weapon": "bazooka", "angle": 120.0},
    {"action": "attack", "weapon": "grenade", "angle": 15.0, "force": 50.0},
]


async def start_client() -> None:
    uri = f"ws://{HOST}:{PORT}"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"type": "CONNECT", "nick": "bot"}))
        player_id: int | None = None
        async for message in ws:
            msg = json.loads(message)
            print("←", msg)
            t = msg.get("type")
            if t == "ASSIGN_ID":
                player_id = msg["player_id"]
                continue
            if t == "TURN_BEGIN" and msg["player_id"] == player_id:
                action = random.choice(ACTIONS)
                await ws.send(
                    json.dumps(
                        {
                            "type": "ACTION",
                            "player_id": player_id,
                            "action": action,
                        }
                    )
                )
                print("→ ACTION", action)


if __name__ == "__main__":
    asyncio.run(start_client())
