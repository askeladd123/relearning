#!/usr/bin/env python3
import asyncio
import json
import random
import websockets

HOST = '127.0.0.1'
PORT = 8765

async def handler(websocket):
    print("Client connected")
    try:
        async for message in websocket:
            try:
                request = json.loads(message)
            except json.JSONDecodeError:
                continue

            print(request)

            # Inspect request['actions'] here if needed
            response = {
                'score': 50,
                'finished': False,
                'new-environment': {
                    'worms': [
                        {'id': 0, 'health': 100, 'x': 12, 'y': 34},
                        {'id': 1, 'health':  90, 'x': 56, 'y': 78},
                        {'id': 2, 'health':   0, 'x': 91, 'y': 23},
                    ],
                    'map': [
                        [1,0,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0,0],
                        [1,1,1,0,0,0,1,0],
                        [1,1,1,0,0,1,1,1]
                    ]
                }
            }
            await websocket.send(json.dumps(response))
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, HOST, PORT):
        print(f"WebSocket server listening on ws://{HOST}:{PORT}")
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    asyncio.run(main())
