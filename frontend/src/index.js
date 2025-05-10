import * as PIXI from 'pixi.js';

const app = new PIXI.Application({
    width: 800,
    height: 600,
    backgroundColor: 0x1099bb
});

document.body.appendChild(app.view);

const text = new PIXI.Text('Hello, Pixi.js!', {
    fontFamily: 'Arial',
    fontSize: 48,
    fill: 0xffffff,
    align: 'center'
});

text.anchor.set(0.5);
text.x = app.screen.width / 2;
text.y = app.screen.height / 2;

app.stage.addChild(text);
