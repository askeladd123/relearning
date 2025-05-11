import asyncio
import json
import random
import websockets

HOST = '127.0.0.1'
PORT = 8765

# Complex actions list
ACTIONS = [
    {'action': 'stand'},
    {'action': 'walk',   'amount-x': 12.0},
    {'action': 'attack', 'weapon': 'kick',    'force': 70.0},
    {'action': 'attack', 'weapon': 'bazooka', 'angle': 120.0},
    {'action': 'attack', 'weapon': 'grenade', 'angle': 15.0,  'force': 50.0}
]

async def start_client():
    uri = f"ws://{HOST}:{PORT}"
    async with websockets.connect(uri) as websocket:
        print(f"Connected to ws://{HOST}:{PORT}")
        while True:
            input("Press Enter to send a random action...")
            action = random.choice(ACTIONS)
            request = {'actions': action}
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            try:
                data = json.loads(response)
                print(f"Sent action: {action}\nReceived response: {data}\n")
            except json.JSONDecodeError:
                print("Received invalid response")

if __name__ == '__main__':
    asyncio.run(start_client())
