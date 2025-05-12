$'

I am in the process of creating a reinforcement learning project where agents play the turn-based game W.O.R.M.S.. It consists of three types of components:
- server / environment
- cli client / ai agent
- web client / player interface for testing

There can be multiple clients, all commuicates with the server through network ports, sending json documents on websockets.

## structure
```
(tree -I node_modules -I __pycache__ -I plan.md -I prompt.nu)
```

## server
This is the component that handles actions sent from different agents, simulates the environments, and sends back the new state.

### `./environment/server.py`
```py
(open ./environment/server.py)
```

### `./environment/game_core.py`
```py
(open ./environment/game_core.py)
```

## web client
This is a type of agent that will be used to visualize and test the system. The user can play a game instead of having the AI decide actions.

### `./frontend/src/script.js`
```js
(open ./frontend/src/script.js)
```

### `./frontend/public/index.html`
```html
(open ./frontend/public/index.html)
```

## reinforcement client

### `./agents/client.py`
This is a type of agent that decides actions based on a reinforcement model, and trains during games. 

```py
(open ./agents/client.py)
```

## json documentation
(open ./json-docs.md)
'
