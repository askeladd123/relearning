import { Application, Container, Graphics } from 'pixi.js';

document.addEventListener('DOMContentLoaded', async () => {
  // Initialize Pixi Application
  const app = new Application();
  await app.init({ width: window.innerWidth, height: 400, backgroundColor: '#1099bb' });
  document.getElementById('game-container').appendChild(app.canvas);

  // Container for the map
  const tileSize = 20;
  const mapContainer = new Container();
  app.stage.addChild(mapContainer);

  // Define possible actions
  const ACTIONS = [
    { action: 'stand' },
    { action: 'walk',   'amount-x': 12.0 },
    { action: 'attack', 'weapon': 'kick',    'force': 70.0 },
    { action: 'attack', 'weapon': 'bazooka', 'angle': 120.0 },
    { action: 'attack', 'weapon': 'grenade', 'angle': 15.0,  'force': 50.0 }
  ];

  // Setup WebSocket
  const socket = new WebSocket('ws://127.0.0.1:8765');
  socket.addEventListener('open', () => {
    console.log('WebSocket connected');
  });
  socket.addEventListener('message', (evt) => {
    try {
      const data = JSON.parse(evt.data);
      drawMap(data['new-environment'].map);
    } catch (e) {
      console.error('Invalid JSON:', e);
    }
  });

  // UI elements
  const actionSelect = document.getElementById('action-select');
  const actionParams = document.getElementById('action-params');
  const sendBtn = document.getElementById('send-action');

  // Populate dropdown
  ACTIONS.forEach((a, i) => {
    const opt = document.createElement('option');
    let label = a.action;
    if (a.weapon) label += ` (${a.weapon})`;
    opt.value = i;
    opt.textContent = label;
    actionSelect.appendChild(opt);
  });

  // Update params UI based on selected action
  function updateParams() {
    actionParams.innerHTML = '';
    const template = ACTIONS[actionSelect.value];
    Object.keys(template).forEach((key) => {
      if (key === 'action') return;
      const group = document.createElement('div');
      group.className = 'field-group';
      const lbl = document.createElement('label');
      lbl.htmlFor = `param-${key}`;
      lbl.textContent = key + ':';
      const input = document.createElement('input');
      input.id = `param-${key}`;
      input.name = key;
      input.type = typeof template[key] === 'number' ? 'number' : 'text';
      input.value = template[key];
      group.appendChild(lbl);
      group.appendChild(input);
      actionParams.appendChild(group);
    });
  }

  actionSelect.addEventListener('change', updateParams);
  updateParams();

  // Send action on button click
  sendBtn.addEventListener('click', () => {
    const idx = actionSelect.value;
    const template = ACTIONS[idx];
    const payload = { action: template.action };
    Object.keys(template).forEach((key) => {
      if (key === 'action') return;
      const input = document.getElementById(`param-${key}`);
      let val = input.value;
      if (typeof template[key] === 'number') {
        val = parseFloat(val);
      }
      payload[key] = val;
    });
    socket.send(JSON.stringify({ actions: payload }));
  });

  // Function to draw map
  function drawMap(map) {
    mapContainer.removeChildren();
    map.forEach((row, y) => {
      row.forEach((cell, x) => {
        const g = new Graphics();
        g.beginFill(cell === 1 ? 0x000000 : 0xffffff);
        g.drawRect(x * tileSize, y * tileSize, tileSize, tileSize);
        g.endFill();
        mapContainer.addChild(g);
      });
    });
  }
});

