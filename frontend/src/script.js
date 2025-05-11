import { Application, Container, Graphics } from 'pixi.js';

let lastEnv = null;
let playerId = null;
let myTurn = false;

document.addEventListener('DOMContentLoaded', async () => {
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

  const ACTIONS = [
    { action: 'stand' },
    { action: 'walk', 'amount-x': 12.0 },
    { action: 'attack', weapon: 'kick', force: 70.0 },
    { action: 'attack', weapon: 'bazooka', angle: 120.0 },
    { action: 'attack', weapon: 'grenade', angle: 15.0, force: 50.0 },
  ];

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
        console.log('Assigned ID', playerId);
        break;
      case 'TURN_BEGIN':
        myTurn = msg.player_id === playerId;
        lastEnv = msg.state;
        drawEnvironment(lastEnv);
        break;
      case 'TURN_RESULT':
        lastEnv = msg.state;
        drawEnvironment(lastEnv);
        break;
      case 'TURN_END':
        myTurn = msg.next_player_id === playerId;
        break;
      case 'GAME_OVER':
        alert(`Game over! Winner: ${msg.winner_id}`);
        myTurn = false;
        break;
      case 'ERROR':
        console.error(msg.msg);
        myTurn = false;
        break;
    }
    updateSendButton();
  });

  /* ---------- UI ---------- */
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

  const buildParams = () => {
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
  };

  actionSelect.addEventListener('change', buildParams);
  buildParams();

  const updateSendButton = () => {
    sendBtn.disabled = !myTurn;
  };

  sendBtn.addEventListener('click', () => {
    if (!myTurn) return;
    const tmpl = ACTIONS[actionSelect.value];
    const payload = { action: tmpl.action };
    Object.keys(tmpl).forEach((k) => {
      if (k === 'action') return;
      const v = document.getElementById(`param-${k}`).value;
      payload[k] = typeof tmpl[k] === 'number' ? parseFloat(v) : v;
    });
    socket.send(
      JSON.stringify({
        type: 'ACTION',
        player_id: playerId,
        action: payload,
      }),
    );
    myTurn = false;
    updateSendButton();
  });

  window.addEventListener('resize', () => {
    app.renderer.resize(window.innerWidth, canvasHeight);
    if (lastEnv) drawEnvironment(lastEnv);
  });

  function drawEnvironment(env) {
    content.removeChildren();
    const map = env.map;
    const worms = env.worms || [];

    const rows = map.length;
    const cols = map[0].length;
    const W = app.renderer.screen.width;
    const H = app.renderer.screen.height;
    const size = Math.min(W / cols, H / rows);
    const xMargin = (W - size * cols) / 2;
    const yMargin = (H - size * rows) / 2;

    map.forEach((row, y) => {
      row.forEach((cell, x) => {
        const g = new Graphics();
        g.beginFill(cell === 1 ? 0x000000 : 0x1099bb);
        g.drawRect(xMargin + x * size, yMargin + y * size, size, size);
        g.endFill();
        content.addChild(g);
      });
    });

    worms.forEach((w) => {
      const cx = xMargin + w.x * size;
      const cy = yMargin + w.y * size;
      const r = size * 0.4;
      const c = new Graphics();
      c.beginFill(0xff0000);
      c.drawCircle(cx, cy, r);
      c.endFill();
      content.addChild(c);
    });
  }
});
