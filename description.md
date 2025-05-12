*Turn‑based Reinforcement‑Learning playground*

The goal is to train an AI to play W.O.R.M.S.. I also want to be able to test the game against the agents. That is why I chose to modularize into components, that communicate through websockets:
- server / environment: manages communication with clients and simulates the environment
- cli client / ai agent: choses actions based on the gives environment, and learns
- web client: user can take the role as a player and manually chose actions against the AI. This is visualize as a simple game interface in the browser

---

## 1  High‑level Picture

```

┌──────────┐      WebSocket/JSON       ┌─────────────┐
│  Browser │◀─────────────────────────▶│  Server     │
│ (Pixi UI)│                           │ (env + core)│
└────┬─────┘                           └────┬────────┘
     │ WebSocket/JSON                        │
     │                                       │
     ▼                                       ▼
┌──────────┐      WebSocket/JSON       ┌─────────────┐
│  RL Bot  │◀─────────────────────────▶│  GameCore   │
│ (Python) │                           │  (pure logic)
└──────────┘                           └─────────────┘

````

* **Server / Environment** (`environment/`)  
  Runs a match, owns the authoritative **`GameCore`** logic, and speaks the
  JSON protocol over **WebSockets**.  
* **Agents** (`agents/`)  
  CLI clients—human or reinforcement‑learning models—that connect, read
  `TURN_BEGIN`, decide an `ACTION`, and learn from `TURN_RESULT`.  
* **Web Client** (`frontend/`)  
  Pixi.js interface for humans: shows the map full‑width, lets you pick
  actions via dropdown + dynamic parameter inputs, and renders worms in
  sub‑tile positions.

Everything communicates through **one common surface**: the JSON messages
documented in `json-docs.md`.

---

## 2  Runtime Flow

1. **CONNECT** Clients open `ws://127.0.0.1:8765` and send
   ```json
   { "type": "CONNECT", "nick": "<name>" }
````

The server replies `ASSIGN_ID`.

2. **Turn loop**

   ```
   TURN_BEGIN → ACTION → TURN_RESULT → TURN_END
   ```

   repeats until `GAME_OVER`.

3. Clients render or learn from the **Game State** snapshot inside each
   `TURN_BEGIN` / `TURN_RESULT`.  The state contains:

   * `worms` with **world‑unit** `(x,y)` floats
   * `map` grid with `1 = terrain`, `0 = empty`

4. **Physics & rules** (to‑be‑implemented) live entirely in `GameCore.step()`.

---

## 3  Coordinate System

* **Tile grid** Index `(0,0)` is top‑left.
* **World units** 1 world‑unit ≙ 1 tile.  Worms can be at `3.2, 1.75` etc.
* **Screen mapping** (client side)

  ```
  tile   = min(canvasW / cols, canvasH / rows)
  offset = ((canvasW-tile*cols)/2, (canvasH-tile*rows)/2)
  pixelX = offset.x + xWorld * tile
  pixelY = offset.y + yWorld * tile
  ```

This keeps the grid centred and lets agents move smoothly within a tile.

---

## 4  Components in Detail

| Folder         | Key files                            | Role                                  |
| -------------- | ------------------------------------ | ------------------------------------- |
| `environment/` | `server.py`, `game_core.py`          | Match orchestration + pure game logic |
| `agents/`      | `client.py` (baseline random bot)    | Example RL agent; train & act here    |
| `frontend/`    | `src/script.js`, `public/index.html` | Pixi.js UI + dynamic action form      |
| root           | `json-docs.md`                       | Authoritative protocol spec           |

### Build / Run Quick‑start

```bash
# Python side
cd environment && python server.py

# Web UI
cd frontend
npm i            # (once)
npx esbuild src/script.js --bundle --outfile=public/bundle.js --format=esm
npx serve public # or any static server

# Random bot
cd agents && python client.py
```

---

## 5  Extensibility Guidelines

* **Physics upgrades** Add terrain slopes or water by extending the `map`
  encoding; clients ignore unknown keys.
* **New actions** Introduce additional `action` variants in `json-docs.md`
  (clients will ignore ones they don’t understand).
* **More players** `GameCore.expected_players()` controls how many `CONNECT`s
  start a match.
* **Rewards** Fill the `reward` field in `TURN_RESULT`; RL agents train on it.

---
