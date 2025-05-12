# WebSocket JSON Protocol

This document defines every JSON message exchanged between **clients** (bots or browsers) and the **Worms Server**.  All messages are single JSON objects sent over a WebSocket connection.  Each object **must** contain a `type` field that determines its structure.

*Numbers below use JavaScript / Python conventions — i.e. IEEE‑754 double precision floats for real numbers and signed 32‑bit integers for IDs unless stated otherwise.*

---

## 1  Overall flow

```text
          (connect)                    ┌──────────────┐
 Client ──────────────────────────────►│  Server      │
  ▲     ① CONNECT                      │              │
  │     ② ASSIGN_ID                    │   Turn loop  │
  │                                    │     …        │
  │  ┌─────────────────────────────────┘
  │  │      (repeat until GAME_OVER)
  │  │
  │  │ ③ TURN_BEGIN        ─→  only the player whose turn it is
  │  │ ④ ACTION            ←─  that same player (within 15 s)
  │  │ ⑤ TURN_RESULT       ─→  broadcast to *all* players & spectators
  │  │ ⑥ TURN_END          ─→  broadcast; announces next player
  │  │           (back to ③ for next player)
  │  │
  │  └───▶ GAME_OVER        ─→  broadcast once `game_over()` becomes true
  │
  └──────── connection closes or stays open for post‑game chat
```

Client‑to‑server messages are **bold**, server‑to‑client are *italic*, and broadcasts are sent to everyone (players **and** spectators).

---

## 2  Message catalog

### 2.1  CONNECT  (client → server)

| Field       | Type    | Required | Description                                                     |
| ----------- | ------- | -------- | --------------------------------------------------------------- |
| `type`      | string  | ✓        | Always the literal string `"CONNECT"`.                          |
| `nick`      | string  | ✗        | Human‑readable nickname (shown in logs/UI).                     |
| `spectator` | boolean | ✗        | `true` to join as a non‑playing spectator. Defaults to `false`. |

#### Example

```json
{ "type": "CONNECT", "nick": "bot‑42" }
```

---

### 2.2  ASSIGN\_ID  (*server → client*)

| Field       | Type    | Required | Description                                            |
| ----------- | ------- | -------- | ------------------------------------------------------ |
| `type`      | string  | ✓        | `"ASSIGN_ID"`                                          |
| `player_id` | integer | ✗        | Positive integer starting at 1. Absent for spectators. |

```json
{ "type": "ASSIGN_ID", "player_id": 2 }
```

---

### 2.3  TURN\_BEGIN  (*server → client*)

Sent to **all** sockets at the beginning of each turn.  Only the targeted player is allowed to act.

| Field           | Type                      | Required | Description                                       |
| --------------- | ------------------------- | -------- | ------------------------------------------------- |
| `type`          | string                    | ✓        | `"TURN_BEGIN"`                                    |
| `player_id`     | integer                   | ✓        | ID whose turn it is                               |
| `state`         | [Game State](#game-state) | ✓        | Full current state snapshot                       |
| `time_limit_ms` | integer                   | ✓        | How long (milliseconds) the player has to respond |

Example (truncated):

```json
{
  "type": "TURN_BEGIN",
  "player_id": 1,
  "state": { «…see below…» },
  "time_limit_ms": 15000
}
```

---

### 2.4  ACTION  (client → server)

Must be sent **only by the player whose turn it is** within `time_limit_ms`.

| Field       | Type    | Required | Description                     |
| ----------- | ------- | -------- | ------------------------------- |
| `type`      | string  | ✓        | `"ACTION"`                      |
| `player_id` | integer | ✓        | Must match sender’s assigned ID |
| `action`    | object  | ✓        | One of the structures below     |

#### `action` object variants

| Variant  | Required keys                         | Extra keys                                                                                                                 |
| -------- | ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `stand`  | `{"action":"stand"}`                  | none                                                                                                                       |
| `walk`   | `{"action":"walk","amount-x":float}`  | none                                                                                                                       |
| `attack` | `{"action":"attack","weapon":string}` | Depends on weapon:<br/>• `kick` → `force` (float 0‑100)<br/>• `bazooka` → `angle` (deg)<br/>• `grenade` → `angle`, `force` |

Example:

```json
{
  "type": "ACTION",
  "player_id": 2,
  "action": { "action": "attack", "weapon": "bazooka", "angle": 45.0 }
}
```

---

### 2.5  TURN\_RESULT  (*server → all*)

Broadcast immediately after an `ACTION` is processed.

| Field       | Type                      | Required | Description                                |
| ----------- | ------------------------- | -------- | ------------------------------------------ |
| `type`      | string                    | ✓        | `"TURN_RESULT"`                            |
| `player_id` | integer                   | ✓        | The acting player                          |
| `state`     | [Game State](#game-state) | ✓        | Updated state after action                 |
| `reward`    | number                    | ✗        | Per‑turn reward (used for RL); 0 if unused |

---

### 2.6  TURN\_END  (*server → all*)

Announces whose turn is next.

| Field            | Type    | Required | Description                               |
| ---------------- | ------- | -------- | ----------------------------------------- |
| `type`           | string  | ✓        | `"TURN_END"`                              |
| `next_player_id` | integer | ✓        | ID who will receive the next `TURN_BEGIN` |

---

### 2.7  GAME\_OVER  (*server → all*)

Sent exactly once when `GameCore.game_over()` becomes *True*.

| Field         | Type                      | Required | Description                       |
| ------------- | ------------------------- | -------- | --------------------------------- |
| `type`        | string                    | ✓        | `"GAME_OVER"`                     |
| `winner_id`   | integer                   | ✗        | ID of the winning player (if any) |
| `final_state` | [Game State](#game-state) | ✓        | Final snapshot                    |

---

### 2.8  ERROR  (*server → client*)

Sent to a player when their action is invalid or late.

| Field  | Type   | Required | Description                      |
| ------ | ------ | -------- | -------------------------------- |
| `type` | string | ✓        | `"ERROR"`                        |
| `msg`  | string | ✓        | Human‑readable error description |

---

## 3  <code id="game-state">Game State</code>

```jsonc
{
  "worms": [
    { "id": 0, "health": 100, "x": 1.0, "y": 1.0 },
    { "id": 1, "health":  90, "x": 6.0, "y": 2.0 }
  ],
  "map": [        // 2‑D array of ints
    [1,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0],
    [1,1,1,0,0,0,1,0],
    [1,1,1,0,0,1,1,1]
  ]
}
```

| Key     | Type          | Description                                                                                        |
| ------- | ------------- | -------------------------------------------------------------------------------------------------- |
| `worms` | array         | Each worm’s *id*, *health* (0‑100), and *position* in **world units** (floats; 1 unit = one tile). |
| `map`   | 2‑D int array | `1` = terrain, `0` = empty/water.  Index 0,0 is top‑left.                                          |

Future versions may extend the state object (e.g. wind, pickups).  Clients **must ignore** unknown keys.
