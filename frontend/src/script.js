import { Application, Container, Graphics } from 'pixi.js';

// Store the last received environment so we can redraw on resize
let lastEnv = null;

document.addEventListener('DOMContentLoaded', async () => {
  // Initialize Pixi Application
  const app = new Application();
  const canvasHeight = 400;
  await app.init({ width: window.innerWidth, height: canvasHeight, backgroundColor: '#1099bb' });
  document.getElementById('game-container').appendChild(app.canvas);

  // Container for map and worms
  const contentContainer = new Container();
  app.stage.addChild(contentContainer);

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
  socket.addEventListener('open', () => console.log('WebSocket connected'));
  socket.addEventListener('message', (evt) => {
    try {
      const data = JSON.parse(evt.data)['new-environment'];
      lastEnv = data;
      drawEnvironment(data, contentContainer, app);
    } catch (e) {
      console.error('Invalid JSON:', e);
    }
  });

  // UI elements
  const actionSelect = document.getElementById('action-select');
  const actionParams = document.getElementById('action-params');
  const sendBtn      = document.getElementById('send-action');

  // Populate dropdown
  ACTIONS.forEach((a, i) => {
    const opt = document.createElement('option');
    let label = a.action;
    if (a.weapon) label += ` (${a.weapon})`;
    opt.value       = i;
    opt.textContent = label;
    actionSelect.appendChild(opt);
  });

  // Build parameter inputs for each action
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
      input.id    = `param-${key}`;
      input.name  = key;
      input.type  = typeof template[key] === 'number' ? 'number' : 'text';
      input.value = template[key];
      group.appendChild(lbl);
      group.appendChild(input);
      actionParams.appendChild(group);
    });
  }

  actionSelect.addEventListener('change', updateParams);
  updateParams();

  // Send the chosen action over WS
  sendBtn.addEventListener('click', () => {
    const idx      = actionSelect.value;
    const template = ACTIONS[idx];
    const payload  = { action: template.action };

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

  // Handle window resize: adjust renderer width and redraw
  window.addEventListener('resize', () => {
    app.renderer.resize(window.innerWidth, canvasHeight);
    if (lastEnv) drawEnvironment(lastEnv, contentContainer, app);
  });
});

/**
 * Draw both the map and the worms into the given container.
 */
function drawEnvironment(env, container, app) {
  container.removeChildren();
  const map  = env.map;
  const worms = env.worms || [];

  const rows = map.length;
  const cols = map[0].length;
  const W    = app.renderer.screen.width;
  const H    = app.renderer.screen.height;
  const size = Math.min(W / cols, H / rows);
  const xMargin = (W - size * cols) / 2;
  const yMargin = (H - size * rows) / 2;

  // Draw tiles
  map.forEach((row, y) => {
    row.forEach((cell, x) => {
      const g = new Graphics();
      g.beginFill(cell === 1 ? 0x000000 : 0x1099bb); // water bg vs terrain
      g.drawRect(xMargin + x * size, yMargin + y * size, size, size);
      g.endFill();
      container.addChild(g);
    });
  });

  // Draw worms as circles
  worms.forEach((w) => {
    const cx = xMargin + w.x * size;
    const cy = yMargin + w.y * size;
    const radius = size * 0.4;
    const c = new Graphics();
    c.beginFill(0xff0000);
    c.drawCircle(cx, cy, radius);
    c.endFill();
    container.addChild(c);
  });
}
