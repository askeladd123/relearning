import { Application, Container, Graphics } from 'pixi.js';

let lastEnv = null;
let playerId = null;
let currentPlayer = null;
let myTurn = false;

document.addEventListener('DOMContentLoaded', async () => {
  // ---------- PIXI ----------------------------------------------------
  const app = new Application();
  const canvasHeight = 400;
  await app.init({
    width: window.innerWidth,
    height: canvasHeight,
    backgroundColor: '#1099bb',
  });
  document.getElementById('game-container').appendChild(app.canvas);

  const content = new Container();
  app.stage.addChild(content);

  // ---------- UI ELEMENTS ---------------------------------------------
  const statusBar = document.createElement('div');
  statusBar.id = 'status-bar';
  statusBar.style.cssText = 'padding:0.5em; text-align:center; font-family:sans-serif;';
  document.body.insertBefore(statusBar, document.getElementById('game-container'));

  const overlay = document.createElement('div');
  overlay.id = 'waiting-overlay';
  overlay.style.cssText = [
    'position:absolute',
    'top:0',
    'left:0',
    'width:100%',
    'height:100%',
    'background:rgba(0,0,0,0.5)',
    'display:flex',
    'align-items:center',
    'justify-content:center',
    'color:white',
    'font-size:2em',
    'pointer-events:none',
    'z-index:10',
    'visibility:hidden'
  ].join(';');
  overlay.textContent = 'Waiting for opponent...';
  document.body.appendChild(overlay);

  // ---------- Action presets ------------------------------------------
  const ACTIONS = [
    { action: 'stand' },
    { action: 'walk', dx: 1.0 },
    { action: 'attack', weapon: 'kick', force: 70.0 },
    { action: 'attack', weapon: 'bazooka', angle_deg: 120.0 },
    { action: 'attack', weapon: 'grenade', angle_deg: 15.0, force: 50.0 },
  ];

  // ---------- WebSocket -----------------------------------------------
  const socket = new WebSocket('ws://127.0.0.1:8765');

  socket.addEventListener('open', () => {
    console.log('WebSocket connected');
    socket.send(JSON.stringify({ type: 'CONNECT', nick: 'browser' }));
  });

  socket.addEventListener('message', (evt) => {
    const msg = JSON.parse(evt.data);
    switch (msg.type) {
      case 'ASSIGN_ID':
        playerId = msg.player_id;
        statusBar.textContent = `You are Player ${playerId}`;
        break;
      case 'TURN_BEGIN':
        currentPlayer = msg.player_id;
        myTurn = currentPlayer === playerId;
        lastEnv = msg.state;
        drawEnvironment(lastEnv);
        updateUI();
        break;
      case 'TURN_RESULT':
        lastEnv = msg.state;
        drawEnvironment(lastEnv);
        break;
      case 'TURN_END':
        currentPlayer = msg.next_player_id;
        myTurn = currentPlayer === playerId;
        updateUI();
        break;
      case 'GAME_OVER':
        alert(`Game over! Winner: ${msg.winner_id}`);
        myTurn = false;
        updateUI();
        break;
      case 'ERROR':
        console.error(msg.msg);
        break;
    }
  });

  // ---------- UI CONTROLS ---------------------------------------------
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

  function buildParams() {
    actionParams.innerHTML = '';
    const tmpl = ACTIONS[actionSelect.value];
    Object.keys(tmpl).forEach((k) => {
      if (k === 'action') return;
      const group = document.createElement('div');
      group.className = 'field-group';
      const lbl = document.createElement('label');
      lbl.htmlFor = `param-${k}`;
      lbl.textContent = `${k}:`;
      const inp = document.createElement('input');
      inp.id = `param-${k}`;
      inp.type = typeof tmpl[k] === 'number' ? 'number' : 'text';
      inp.value = tmpl[k];
      group.appendChild(lbl);
      group.appendChild(inp);
      actionParams.appendChild(group);
    });
    updateUI();
  }
  actionSelect.addEventListener('change', buildParams);
  buildParams();

  function updateUI() {
    // Set status text
    if (playerId != null) {
      statusBar.textContent = myTurn
        ? `Your turn (Player ${playerId})`
        : `Waiting for Player ${currentPlayer}`;
    }
    // Enable/disable controls
    sendBtn.disabled = !myTurn;
    actionSelect.disabled = !myTurn;
    actionParams.querySelectorAll('input').forEach(i => { i.disabled = !myTurn; });
    // Show or hide overlay
    overlay.style.visibility = myTurn ? 'hidden' : 'visible';
  }

  sendBtn.addEventListener('click', () => {
    if (!myTurn) return;
    const tmpl = ACTIONS[actionSelect.value];
    const payload = { action: tmpl.action };
    Object.keys(tmpl).forEach((k) => {
      if (k === 'action') return;
      const v = document.getElementById(`param-${k}`).value;
      payload[k] = typeof tmpl[k] === 'number' ? parseFloat(v) : v;
    });
    socket.send(JSON.stringify({ type: 'ACTION', player_id: playerId, action: payload }));
    myTurn = false;
    updateUI();
  });

  window.addEventListener('resize', () => {
    app.renderer.resize(window.innerWidth, canvasHeight);
    if (lastEnv) drawEnvironment(lastEnv);
  });

  // ---------- Rendering -----------------------------------------------
  function drawEnvironment(env) {
    content.removeChildren();
    const { map, worms = [] } = env;
    const rows = map.length;
    const cols = map[0].length;
    const W = app.renderer.screen.width;
    const H = app.renderer.screen.height;
    const tileSize = Math.min(W / cols, H / rows);
    const xMargin = (W - tileSize * cols) / 2;
    const yMargin = (H - tileSize * rows) / 2;

    // Draw terrain
    map.forEach((row, y) => {
      row.forEach((cell, x) => {
        const g = new Graphics();
        g.beginFill(cell === 1 ? 0x000000 : 0x1099bb);
        g.drawRect(
          xMargin + x * tileSize,
          yMargin + y * tileSize,
          tileSize,
          tileSize,
        );
        g.endFill();
        content.addChild(g);
      });
    });

    // Draw worms (world‑unit coords)
    worms.forEach((w) => {
      const cx = xMargin + w.x * tileSize;
      const cy = yMargin + w.y * tileSize;
      const r = tileSize * 0.4;
      const c = new Graphics();
      c.beginFill(0xff0000);
      c.drawCircle(cx, cy, r);
      c.endFill();
      content.addChild(c);
    });
  }
});
