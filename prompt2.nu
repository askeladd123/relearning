$'
This is an overview of my ongoing project using reinfocement learning to play the 2D, turn-based game W.O.R.M.S..
- [[#description]]
- [[#structure]]
- [[#server]]
- [[#reinforcement client]]
- [[#json documentation]]

# instructions
Please tell me if you think there are inconsistancies with my descriptions and the code. Also tell me if you have clarifying questions before finishing my request.

If you made changes or created a new file, provide the path and the full code, not just snippets of what you changed. 

Send me the files in code blocks in chat.

# description
(open description.md)

---

# structure
```
(tree -I node_modules -I __pycache__ -I plan.md -I prompt.nu)
```

---

# server
This is the component that handles actions sent from different agents, simulates the environments, and sends back the new state.

## `./environment/server.py`
```py
(open ./environment/server.py)
```

## `./environment/game_core.py`
```py
(open ./environment/game_core.py)
```

---

# reinforcement client

## `./agents/a2c_manual/client.py`
This is a type of agent that decides actions based on a reinforcement model, and trains during games. 

```py
(open ./agents/client.py)
```

---

# json documentation

## `./json-docs.md`:
(open ./json-docs.md)
'
