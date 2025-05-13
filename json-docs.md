# W\.O.R.M.S. WebSocket JSON Protocol

All messages are JSON objects sent over a WebSocket.
Each object **must** include a `"type"` field that selects one of the structures below.
Clients **must ignore** unknown keys so the protocol can evolve.

---

## Overall flow

```text
            Client                                  Server
             │                                         │
             ├─▶  CONNECT                              │
             │     (nick)                              │
             │                                         │
             │   ASSIGN_ID  ◀──────────────────────────┤
             │                                         │
      repeat │                                         │
      until  │  TURN_BEGIN ───────────────────────────▶│
   GAME_OVER │  ACTION       ◀──────────────────────── │
             │  TURN_RESULT ─────────────────────────▶ │
             │  TURN_END    ─────────────────────────▶ │  ← Now sent after each turn
             │                                         │
             └─▶ GAME_OVER   ◀─────────────────────────┤
````

---

## 1  Messages

### 1.1 CONNECT  (client → server)

| Field       | Type    | Required | Description                       |
| ----------- | ------- | -------- | --------------------------------- |
| `type`      | string  | ✓        | `"CONNECT"`                       |
| `nick`      | string  | ✗        | Human nickname (for logs / UI)    |
| `spectator` | boolean | ✗        | `true` to observe without playing |

```json
{ "type": "CONNECT", "nick": "bot-42" }
```

---

### 1.2 ASSIGN\_ID  (server → client)

| Field       | Type    | Required | Description                                     |
| ----------- | ------- | -------- | ----------------------------------------------- |
| `type`      | string  | ✓        | `"ASSIGN_ID"`                                   |
| `player_id` | integer | ✗        | Omitted for spectators; starts at 1 for players |

```json
{ "type": "ASSIGN_ID", "player_id": 2 }
```

---

### 1.3 TURN\_BEGIN  (server → all)

| Field           | Type       | Required | Description                        |
| --------------- | ---------- | -------- | ---------------------------------- |
| `type`          | string     | ✓        | `"TURN_BEGIN"`                     |
| `turn_index`    | integer    | ✓        | 0-based global turn counter        |
| `player_id`     | integer    | ✓        | ID whose turn it is                |
| `state`         | Game State | ✓        | Full snapshot                      |
| `time_limit_ms` | integer    | ✓        | How long that player has to answer |

---

### 1.4 ACTION  (client → server)

Sent **only** by the `player_id` that just received `TURN_BEGIN`.

| Field       | Type   | Required | Description               |
| ----------- | ------ | -------- | ------------------------- |
| `type`      | string | ✓        | `"ACTION"`                |
| `player_id` | int    | ✓        | Must match sender’s ID    |
| `action`    | object | ✓        | One of the variants below |

#### `action` variants

| Variant           | Required keys                                               | Notes                                                |
| ----------------- | ----------------------------------------------------------- | ---------------------------------------------------- |
| stand             | `{ "action": "stand" }`                                     | –                                                    |
| walk              | `{ "action": "walk", "dx": float }`                         | `dx` in world units (+ → right)                      |
| attack:kick       | `{ "action": "attack", "weapon": "kick" }`                  | Flat 80 damage to the first living worm within 1 unit |
| attack:bazooka    | `{ "action": "attack", "weapon": "bazooka", "angle_deg": float }` | 0° = right, CCW positive                             |
| attack:grenade    | `{ "action": "attack", "weapon": "grenade", "dx": float }`  | Single `dx` (max |dx| = 3); horizontal distance only |

Gravity is applied **instantly** only after a **walk** action: the worm falls in the same column until it lands on the first solid tile (`map[row][col] == 1`) or exits below the last row (water), which sets its `health` to `0`.

---

### 1.5 TURN\_RESULT  (server → all)

| Field        | Type       | Required | Description                       |
| ------------ | ---------- | -------- | --------------------------------- |
| `type`       | string     | ✓        | `"TURN_RESULT"`                   |
| `turn_index` | integer    | ✓        | Same index as the triggering turn |
| `player_id`  | integer    | ✓        | The acting player                 |
| `state`      | Game State | ✓        | Resulting state                   |
| `reward`     | number     | ✗        | Optional per-turn reward          |
| `effects`    | object     | ✗        | Optional visual‐only data describing the projectile trajectory and impact:  |
|              |            |          | • `weapon` (string): `"bazooka"` or `"grenade"`                             |
|              |            |          | • `trajectory` (array of `{ x: number, y: number }`): points along the shot | 
|              |            |          | • `impact` ({ x: number, y: number }): the final hit location               |

---

### 1.6 TURN\_END  (server → all)

| Field            | Type    | Required | Description          |
| ---------------- | ------- | -------- | -------------------- |
| `type`           | string  | ✓        | `"TURN_END"`         |
| `next_player_id` | integer | ✓        | ID who will act next |

---

### 1.7 PLAYER\_ELIMINATED  (server → all)

| Field       | Type    | Required | Description                 |
| ----------- | ------- | -------- | --------------------------- |
| `type`      | string  | ✓        | `"PLAYER_ELIMINATED"`       |
| `player_id` | integer | ✓        | ID of the eliminated player |

Clients should disable inputs for that player upon receiving this.

---

### 1.8 GAME\_OVER  (server → all)

| Field         | Type       | Required | Description        |
| ------------- | ---------- | -------- | ------------------ |
| `type`        | string     | ✓        | `"GAME_OVER"`      |
| `winner_id`   | integer    | ✗        | Winner’s ID if any |
| `final_state` | Game State | ✓        | Final snapshot     |

---

### 1.9 ERROR  (server → client)

| Field  | Type   | Required | Description         |
| ------ | ------ | -------- | ------------------- |
| `type` | string | ✓        | `"ERROR"`           |
| `msg`  | string | ✓        | Human-readable text |

---

## 2  Game State

```jsonc
{
  "worms": [
    { "id": 0, "nick": "Alice", "health": 100, "x": 3.20, "y": 1.75 },
    { "id": 1, "nick": "Bot-42", "health":  90, "x": 6.00, "y": 2.00 }
  ],
  "map": [
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,1,1,0,0,0,1,0],
    [1,1,1,0,0,1,1,1]
  ]
}
```

| Key     | Type          | Description                                                                                                        |
| ------- | ------------- | ------------------------------------------------------------------------------------------------------------------ |
| `worms` | array         | Each worm’s **`id`**, **`nick`**, **`health`**, and **position** in world units (`x`, `y` floats; 1 unit = 1 tile) |
| `map`   | 2-D int array | `1` = solid terrain, `0` = empty air. Values below the last row (water) kill a worm instantly when fallen into.    |

Clients **must ignore** any extra keys they do not understand.

---

## 3  Rendering rule (client hint)

To draw the map centred in the canvas:

```
tile = min(canvasW / cols, canvasH / rows)
xMargin = (canvasW − tile*cols)/2
yMargin = (canvasH − tile*rows)/2
tile(i,j) at (xMargin + i*tile , yMargin + j*tile)
```
