import { Application, Container, Graphics, Text } from 'pixi.js';

let lastEnv = null;
let playerId = null;
let currentPlayer = null;
let myTurn = false;
let eliminated = false;

document.addEventListener('DOMContentLoaded', async () => {
  const canvasHeight = 400;

  // Create and initialize PIXI Application
  const app = new Application();
  await app.init({
    width: window.innerWidth,
    height: canvasHeight,
    backgroundColor: 0x1099bb
  });
  document.getElementById('game-container').appendChild(app.view);

  const content = new Container();
  app.stage.addChild(content);

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

  const ACTIONS = [
    { action: 'stand' },
    { action: 'walk', dx: 1.0 },
    { action: 'attack', weapon: 'kick', force: 70.0 },
    { action: 'attack', weapon: 'bazooka', angle_deg: 120.0 },
    { action: 'attack', weapon: 'grenade', angle_deg: 15.0, force: 50.0 },
  ];

  const socket = new WebSocket('ws://127.0.0.1:8765');
  socket.addEventListener('open', () => {
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
        if (eliminated) break;
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
      case 'PLAYER_ELIMINATED':
        if (msg.player_id === playerId) {
          eliminated = true;
          myTurn = false;
          statusBar.textContent = 'You have been eliminated';
          sendBtn.disabled = true;
          actionSelect.disabled = true;
          actionParams.querySelectorAll('input').forEach(i => i.disabled = true);
          overlay.textContent = 'Eliminated';
          overlay.style.visibility = 'visible';
        }
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

  const actionSelect = document.getElementById('action-select');
  const actionParams = document.getElementById('action-params');
  const sendBtn = document.getElementById('send-action');

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
    if (playerId != null && !eliminated) {
      statusBar.textContent = myTurn
        ? `Your turn (Player ${playerId})`
        : `Waiting for Player ${currentPlayer}`;
      sendBtn.disabled = !myTurn;
      actionSelect.disabled = !myTurn;
      actionParams.querySelectorAll('input').forEach(i => i.disabled = !myTurn);
      overlay.style.visibility = myTurn ? 'hidden' : 'visible';
    }
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
    // now safe: app.renderer has been initialized by init()
    app.renderer.resize(window.innerWidth, canvasHeight);
    if (lastEnv) drawEnvironment(lastEnv);
  });

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

    map.forEach((row, y) => {
      row.forEach((cell, x) => {
        const g = new Graphics();
        g.beginFill(cell === 1 ? 0x000000 : 0x1099bb);
        g.drawRect(xMargin + x * tileSize, yMargin + y * tileSize, tileSize, tileSize);
        g.endFill();
        content.addChild(g);
      });
    });

    worms.forEach((w) => {
      const cx = xMargin + w.x * tileSize;
      const cy = yMargin + w.y * tileSize;
      const r = tileSize * 0.4;

      const c = new Graphics();
      c.beginFill(0xff0000);
      c.drawCircle(cx, cy, r);
      c.endFill();
      content.addChild(c);

      const barWidth = tileSize * 0.8;
      const barHeight = tileSize * 0.1;
      const barX = cx - barWidth / 2;
      const barY = cy - r - barHeight - 2;
      const bg = new Graphics();
      bg.beginFill(0x555555);
      bg.drawRect(barX, barY, barWidth, barHeight);
      bg.endFill();
      content.addChild(bg);

      const fillRatio = Math.max(0, Math.min(w.health / 100, 1));
      const fg = new Graphics();
      const fillColor = w.health > 50
        ? 0x00ff00
        : w.health > 20
        ? 0xffff00
        : 0xff0000;
      fg.beginFill(fillColor);
      fg.drawRect(barX, barY, barWidth * fillRatio, barHeight);
      fg.endFill();
      content.addChild(fg);

      const name = w.nick || `Player ${w.id}`;
      const txt = new Text(name, {
        fontFamily: 'Arial',
        fontSize: tileSize * 0.15,
        fill: 0xffffff,
      });
      txt.anchor.set(0.5, 1);
      txt.x = cx;
      txt.y = barY - 2;
      content.addChild(txt);
    });
  }
});
