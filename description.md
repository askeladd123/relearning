*Turn-based Reinforcement-Learning playground*

The goal is to train an AI to play free-for-all W.O.R.M.S. This means that each agent controls one worm, and the goal is to kill all other opponents.

I want to be able to test the game against the AI agents. That is why I chose to modularize into components that communicate through WebSockets:
- server / environment: manages communication with clients and simulates the environment  
- CLI client / AI agent: chooses actions based on the given environment, and learns  
- Web client: user can take the role as a player and manually choose actions against the AI; visualized via a simple browser interface

---

## Scoring

Each action yields a numeric **reward** which agents use to learn:

- **Damage points**  
  You earn points equal to the actual HP removed from any opponent.  
  - _Example_: you bazooka-hit a worm at 30 HP with a 60-damage shot → you earn **30 points**.

- **Kill bonus**  
  If an action reduces an opponent’s HP to exactly 0, you get an extra **100 points**.  
  - _Example_: you kick-hit a worm at 1 HP (kick = 80 damage) → you earn **1 (damage) + 100 (kill bonus) = 101 points**.

---

## 1  High-level Picture

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
  Runs a match, owns the authoritative **`GameCore`** logic, and speaks the JSON protocol over WebSockets.  

* **Agents** (`agents/`)  
  CLI clients—human or reinforcement-learning models—that connect, read `TURN_BEGIN`, decide an `ACTION`, and learn from `TURN_RESULT`.  

* **Web Client** (`frontend/`)  
  Pixi.js interface for humans: shows the map full-width, lets you pick actions via dropdown + dynamic parameter inputs, and renders worms in sub-tile positions.

Everything communicates through **one common surface**: the JSON messages documented in `json-docs.md`.

---

## 2  Runtime Flow

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

3. Clients render or learn from the **Game State** snapshot inside each `TURN_BEGIN` / `TURN_RESULT`. The state contains:

   * `worms` with **world-unit** `(x,y)` floats
   * `map` grid with `1 = terrain`, `0 = empty`

4. **Physics & rules** (fully in `GameCore.step()`).

---

## 3  Coordinate System

* **Tile grid** Index `(0,0)` is top-left.
* **World units** 1 world-unit ≙ 1 tile; worms can be at `3.2, 1.75`, etc.
* **Screen mapping** (client side):

  ```js
  tile   = Math.min(canvasW / cols, canvasH / rows);
  offset = [(canvasW - tile*cols)/2, (canvasH - tile*rows)/2];
  pixelX = offset[0] + xWorld * tile;
  pixelY = offset[1] + yWorld * tile;
  ```

---

## 4  Components in Detail

| Folder         | Key files                            | Role                                  |
| -------------- | ------------------------------------ | ------------------------------------- |
| `environment/` | `server.py`, `game_core.py`          | Match orchestration + pure game logic |
| `agents/`      | `client.py` (baseline random bot)    | Example RL agent; train & act here    |
| `frontend/`    | `src/script.js`, `public/index.html` | Pixi.js UI + dynamic action form      |
| root           | `json-docs.md`                       | Authoritative protocol spec           |

### Build / Run Quick-start

```bash
# Python side
cd environment && python server.py

# Web UI
cd frontend
npm install        # (once)
npx esbuild src/script.js --bundle --outfile=public/bundle.js --format=esm
npx serve public

# Bot
cd agents && python client.py
```
