import { World3D } from './world3d.js';
import { MonteCarloLab } from './mclab.js';
import { LineChart, buildTuning, drawSpeedGauge, drawWheelGauge, renderEvents, renderIntent, renderResults } from './ui.js';

const $ = (id) => document.getElementById(id);
const world = new World3D($('gl'));

const state = { camTimer: 0, snap: null, ws: null, keys: new Set(), steer: 0, mode: 'active', worldDesc: null, lastSend: 0,
  frames: 0, camFrame: 0 };

// ---------------------------------------------------------------- charts
const chSpeed = new LineChart($('chSpeed'), [
  { color: '#2dd4bf', label: 'speed', fill: true }, { color: '#fb923c', label: 'safe speed cap', width: 1.5 }], { min: 0, max: 1.4 });
const chDist = new LineChart($('chDist'), [
  { color: '#60a5fa', label: 'ahead (static)', fill: true }, { color: '#facc15', label: 'moving object', width: 1.5 },
  { color: '#f472b6', label: 'awareness zone', width: 1.2 }], { min: 0, max: 2.0 });
const chRisk = new LineChart($('chRisk'), [
  { color: '#94a3b8', label: 'physics-only' }, { color: '#facc15', label: 'intent blend', width: 1.6 },
  { color: '#2dd4bf', label: 'adaptive (learned)', width: 2.4 }], { min: 0, max: 1 });

// ---------------------------------------------------------------- network
function connect() {
  const ws = new WebSocket(`ws://${location.hostname}:${+location.port + 1}`);
  state.ws = ws;
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'world') onWorld(msg);
    else if (msg.type === 'tunables') buildTuning($('tuneBox'), msg.spec, (id, value) => send({ type: 'cmd', cmd: 'param', id, value }));
    else if (msg.type === 'state') onState(msg);
    else if (msg.type === 'mc') lab.load(msg.cases);
    else if (msg.type === 'mc_status') lab.setStatus(msg.text);
    else if (msg.type === 'export') download('tuning.json', JSON.stringify(msg.data, null, 2));
  };
  ws.onclose = () => setTimeout(connect, 1000);
}
const lab = new MonteCarloLab({ root: $('mcLab'), canvas: $('mcCanvas'), stats: $('mcStats'), status: $('mcStatus') }, (o) => send(o));
$('btnMc').onclick = () => lab.open();
$('mcClose').onclick = () => lab.close();
$('mcReplay').onclick = () => lab.play();
$('mcLab').querySelectorAll('[data-n]').forEach((b) => (b.onclick = () => lab.run(+b.dataset.n)));
$('mcSpeed').onclick = (e) => { const v = e.target.dataset.v; if (!v) return; lab.setSpeed(+v); setActive('mcSpeed', 'v', v); };
const send = (o) => { if (state.ws && state.ws.readyState === 1) state.ws.send(JSON.stringify(o)); };

function download(name, text) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([text], { type: 'application/json' }));
  a.download = name; a.click();
}

const scenarioNames = {};
function onWorld(msg) {
  state.worldDesc = msg;
  world.setWorld(msg);
  world.snapCamera();
  [chSpeed, chDist, chRisk].forEach((c) => c.clear());
  $('scenText').innerHTML = `<b>${msg.name}</b>${msg.text}`;
  const sel = $('scenario');
  if (sel.value !== msg.scenario) sel.value = msg.scenario;
}

const LEVELS = [['clear', 'CLEAR'], ['caution', 'CAUTION'], ['warning', 'WARNING'], ['braking', 'BRAKING']];
function onState(s) {
  state.snap = s;
  const a = s.adas;
  const banner = $('banner');
  if (a.mode === 'off') { banner.className = 'banner off'; $('bannerText').textContent = 'ADAS OFF'; }
  else {
    banner.className = 'banner ' + LEVELS[a.level][0];
    $('bannerText').textContent = s.features.fault ? s.features.fault.toUpperCase() : (s.features.lane.level >= 1 && a.level <= s.features.lane.level ? 'LANE DEPARTURE' : (a.mode === 'advisory' && a.level === 3 ? 'WOULD BRAKE' : LEVELS[a.level][1]));
  }
  const fmt = (x, u = 'm', d = 2) => (x >= 0 ? x.toFixed(d) + ' ' + u : '--');
  $('sD').textContent = fmt(a.D);
  $('sTtc').textContent = fmt(a.ttc, 's', 1);
  $('sVs').textContent = fmt(a.v_safe, 'm/s');
  $('sGap').textContent = s.metrics.min_clearance >= 0 ? s.metrics.min_clearance.toFixed(0) + ' cm' : '--';
  drawSpeedGauge($('speedGauge'), s.car.v, 1.2, a.level);
  drawWheelGauge($('wheelGauge'), s.cmd.steer, s.car.wl, s.car.wr);
  $('pDriver').style.height = Math.abs(s.cmd.pwm) / 255 * 50 + '%';
  $('pDriver').style.transform = s.cmd.pwm < 0 ? 'translateY(100%)' : 'none';
  $('pAdas').style.height = Math.abs(s.cmd.out) / 255 * 50 + '%';
  $('pAdas').style.transform = s.cmd.out < 0 ? 'translateY(100%)' : 'none';

  chSpeed.push(s.t, [Math.abs(s.car.v), a.v_safe >= 0 ? Math.min(a.v_safe, 1.4) : NaN]);
  chDist.push(s.t, [a.D_static >= 0 ? Math.min(a.D_static, 2) : 2, a.D_moving >= 0 ? Math.min(a.D_moving, 2) : NaN,
    a.D_aware >= 0 ? Math.min(a.D_aware, 2) : NaN]);
  chRisk.push(s.t, [s.risk.base, s.risk.blend, s.risk.adaptive]);
  renderIntent($('intentBars'), s.intent);
  renderEvents($('events'), s.events);

  $('btnPause').textContent = s.paused ? 'Resume' : 'Pause';
  $('btnRec').classList.toggle('on', s.recording);
  $('crash').classList.toggle('show', s.car.collided);
  if (a.mode !== state.mode) setActive('modeSeg', 'mode', a.mode);
  state.mode = a.mode;
  warnFeedback(a);
  updateTags(s);
  updateAssists(s);
  drawPipOverlay(s);
  updateCamTab(s);
  world.update(s, Math.min(1 / 20, (performance.now() - (state.lastFrame || performance.now())) / 1000 || 0.033));
}

// ---------------------------------------------------------------- controls
function setActive(segId, attr, value) {
  document.querySelectorAll(`#${segId} button`).forEach((b) => b.classList.toggle('on', b.dataset[attr] === String(value)));
}
$('modeSeg').onclick = (e) => { const m = e.target.dataset.mode; if (m) send({ type: 'cmd', cmd: 'mode', mode: m }); };
$('camSeg').onclick = (e) => { const c = e.target.dataset.cam; if (c) setCam(c); };
$('speedSeg').onclick = (e) => {
  const v = e.target.dataset.speed; if (!v) return;
  send({ type: 'cmd', cmd: 'speed', value: +v }); setActive('speedSeg', 'speed', v);
};
$('btnPause').onclick = () => send({ type: 'cmd', cmd: 'pause' });
$('btnReset').onclick = () => send({ type: 'cmd', cmd: 'reset' });
$('btnRec').onclick = () => send({ type: 'cmd', cmd: 'record' });
$('btnExport').onclick = () => send({ type: 'cmd', cmd: 'export' });
$('btnPark').onclick = () => send({ type: 'cmd', cmd: 'park' });
$('btnBypass').onclick = () => send({ type: 'cmd', cmd: 'bypass' });
$('btnProfile').onclick = () => send({ type: 'cmd', cmd: 'profile', name: state.profile === 'real' ? 'sim' : 'real' });
$('btnAcc').onclick = () => send({ type: 'cmd', cmd: 'acc' });
$('btnIsa').onclick = () => send({ type: 'cmd', cmd: 'isa' });
$('btnLane').onclick = () => send({ type: 'cmd', cmd: 'lane' });
$('scenario').onchange = (e) => send({ type: 'cmd', cmd: 'scenario', id: e.target.value });
$('tabs').onclick = (e) => {
  const t = e.target.dataset.tab; if (!t) return;
  document.querySelectorAll('#tabs button').forEach((b) => b.classList.toggle('on', b.dataset.tab === t));
  document.querySelectorAll('.tab').forEach((p) => p.classList.toggle('on', p.id === 'tab-' + t));
  if (t === 'results') renderResults($('resultsBox'));
};

function setCam(c) {
  world.setCamMode(c);
  setActive('camSeg', 'cam', c);
}

const scenarios = [['playground', 'Playground'], ['wall', 'Emergency stop'], ['pedestrian', 'Pedestrian crossing'],
  ['emerging', 'Emerging pedestrian'], ['slalom', 'Slalom'], ['corridor', 'Narrow corridor'],
  ['virtual', 'Virtual driver (lapses)'], ['leader', 'Follow the leader'], ['cutin', 'Car cuts in'], ['lane', 'Lane keeping'], ['parking', 'Parking bay (ArUco)'],
  ['signs', 'Traffic signs (ISA)'], ['bypass', 'Obstacle bypass (arc around)']];
$('scenario').innerHTML = scenarios.map(([id, n]) => `<option value="${id}">${n}</option>`).join('');

window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  state.keys.add(k);
  if (k === 'r') send({ type: 'cmd', cmd: 'reset' });
  else if (k === 'x') { const order = ['off', 'advisory', 'active']; send({ type: 'cmd', cmd: 'mode', mode: order[(order.indexOf(state.mode) + 1) % 3] }); }
  else if (k === 'p') send({ type: 'cmd', cmd: 'pause' });
  else if (k === 'g') send({ type: 'cmd', cmd: 'park' });
  else if (k === 'b') send({ type: 'cmd', cmd: 'bypass' });
  else if (k === 'k') send({ type: 'cmd', cmd: 'profile', name: state.profile === 'real' ? 'sim' : 'real' });
  else if (k === 'c') send({ type: 'cmd', cmd: 'acc' });
  else if (k === 'i') send({ type: 'cmd', cmd: 'isa' });
  else if (k === 'l') send({ type: 'cmd', cmd: 'lane' });
  else if (['1', '2', '3', '4'].includes(k)) setCam(['chase', 'top', 'cockpit', 'orbit'][+k - 1]);
  if (['arrowup', 'arrowdown', 'arrowleft', 'arrowright', ' '].includes(k)) e.preventDefault();
});
window.addEventListener('keyup', (e) => state.keys.delete(e.key.toLowerCase()));

function readInput(dt) {
  const K = state.keys;
  let pwm = 0, target = 0;
  if (K.has('w') || K.has('arrowup')) pwm = K.has('shift') ? 255 : 200;
  if (K.has('s') || K.has('arrowdown')) pwm = -200;
  if (K.has('a') || K.has('arrowleft')) target = 1;
  if (K.has('d') || K.has('arrowright')) target = -1;
  if (K.has(' ')) pwm = 0;

  const pads = navigator.getGamepads ? navigator.getGamepads() : [];
  for (const p of pads) {
    if (!p || !p.connected) continue;
    const dz = (v) => (Math.abs(v) < 0.08 ? 0 : v);
    const x = dz(p.axes[0] || 0);
    if (x !== 0) target = -x;                                    // stick right = steer right (negative)
    const rt = p.buttons[7] ? p.buttons[7].value : 0, lt = p.buttons[6] ? p.buttons[6].value : 0;
    if (rt > 0.05 || lt > 0.05) pwm = (rt - lt) * 255;
  }
  const rate = target === 0 ? 5.0 : 3.2;
  state.steer += Math.max(-rate * dt, Math.min(rate * dt, target - state.steer));
  return [state.steer, pwm];
}

// ---------------------------------------------------------------- main loop
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  state.lastFrame = now;

  if (now - state.lastSend > 33) {
    state.lastSend = now;
    const [steer, pwm] = readInput(0.033);
    send({ type: 'input', steer, pwm });
  }

  world.render();

  // car camera picture-in-picture, placed where the #pip element sits
  const r = $('pip').getBoundingClientRect();
  const H = window.innerHeight;
  world.renderCarCamera(Math.round(r.left), Math.round(H - r.bottom), Math.round(r.width), Math.round(r.height));

  if ((state.frames++ % 3) === 0) { chSpeed.draw(); chDist.draw(); chRisk.draw(); }
  requestAnimationFrame(frame);
}

connect();
requestAnimationFrame(frame);
window.__send = send;
window.__world = world;       // handy for debugging in the console


// ---------------------------------------------------------------- assists + camera overlays
function updateAssists(s) {
  const f = s.features;
  state.profile = f.profile;
  $('btnProfile').classList.toggle('on', f.profile === 'real');
  $('profileLabel').textContent = f.profile === 'real' ? 'Car: REAL (measured)' : 'Car: default';
  $('btnBypass').classList.toggle('on', !!(f.bypass && f.bypass.on));
  $('btnPark').classList.toggle('on', f.park.state === 'approach');
  $('btnAcc').classList.toggle('on', f.acc.on);
  $('btnIsa').classList.toggle('on', f.isa.on);
  $('btnLane').classList.toggle('on', f.lane.mode !== 'off');
  $('laneLabel').textContent = 'Lane: ' + f.lane.mode;
  const lines = [];
  if (f.park.state !== 'idle') lines.push(`<b>Park:</b> ${f.park.msg}` + (f.park.error ? `<br>offset ${f.park.error.lateral_cm} cm, yaw ${f.park.error.yaw_deg}&deg;` : ''));
  if (f.bypass) lines.push(`<b>Bypass:</b> ${f.bypass.state} - ${f.bypass.msg}<br>lateral ${(f.bypass.y * 100).toFixed(0)} cm (target ${(f.bypass.y_ref * 100).toFixed(0)}), heading ${f.bypass.th}&deg;`);
  if (f.acc.on && f.acc.lead) lines.push(`<b>Following:</b> gap ${(f.acc.lead[0] * 100).toFixed(0)} cm`);
  if (f.lane.mode !== 'off' && f.lane.valid) lines.push(`<b>Lane:</b> ${(f.lane.offset * 100).toFixed(0)} cm off centre` + (f.lane.assisting ? ' (steering)' : ''));
  if (f.isa.on && f.isa.active) lines.push(`<b>Sign:</b> ${f.isa.active}` + (f.isa.cap >= 0 ? ` (max ${f.isa.cap.toFixed(2)} m/s)` : ''));
  $('assistInfo').innerHTML = lines.join('<br>');
}

function drawPipOverlay(s) {
  const c = $('pipOverlay'), dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
  const g = c.getContext('2d');
  g.setTransform(dpr * w / 640, 0, 0, dpr * h / 480, 0, 0);
  g.clearRect(0, 0, 640, 480);
  g.font = '600 26px system-ui, sans-serif';
  g.lineJoin = 'round';
  const draw = (m, colour, dash, label) => {
    g.strokeStyle = colour; g.fillStyle = colour; g.lineWidth = 4; g.setLineDash(dash);
    g.beginPath();
    m.corners.forEach(([x, y], i) => (i ? g.lineTo(x, y) : g.moveTo(x, y)));
    g.closePath(); g.stroke();
    g.setLineDash([]);
    const [x0, y0] = m.corners[0];
    g.fillText(label, Math.max(4, x0), Math.max(28, y0 - 8));
  };
  for (const m of s.markers) draw(m, '#2dd4bf', [], `#${m.id}  ${m.dist.toFixed(2)} m`);
  for (const m of s.cv) draw(m, '#facc15', [10, 8], '');
  const det = $('pipDet');
  det.textContent = s.markers.length ? `${s.markers.length} marker${s.markers.length > 1 ? 's' : ''} detected` : 'no markers in view';
}

function updateCamTab(s) {
  const box = $('camInfo');
  if (!box.offsetParent) return;           // tab not visible
  const rows = s.markers.map((m) => {
    const cv = s.cv.find((c) => c.id === m.id);
    return `<tr><td>#${m.id}</td><td>${m.dist.toFixed(2)} m</td><td>${cv ? cv.dist.toFixed(2) + ' m' : '--'}</td></tr>`;
  }).join('');
  box.innerHTML = `<table class="det"><tr><th>Marker</th><th>Simulated sensor</th><th>OpenCV on rendered frame</th></tr>${rows || '<tr><td colspan="3" class="hint">No markers in view. Try the "Parking bay" or "Traffic signs" scenario.</td></tr>'}</table>
  <p class="hint" style="margin-top:10px">Teal outlines = geometric sensor used by the controllers. Dashed yellow = corners OpenCV found in the actual rendered image. Agreement between them validates the whole camera pipeline.</p>`;
}

async function sendCameraFrame() {
  if (!$('cvOn').checked || !state.ws || state.ws.readyState !== 1) return;
  const img = world.captureCarCamera();
  if (!img) return;
  const c = document.createElement('canvas');
  c.width = 640; c.height = 480;
  c.getContext('2d').putImageData(img, 0, 0);
  c.toBlob((b) => { if (b && state.ws.readyState === 1) state.ws.send(b); }, 'image/jpeg', 0.88);
}
setInterval(sendCameraFrame, 250);


// ---------------------------------------------------------------- warnings you can hear and feel
let audio = null, soundOn = false, nextBeep = 0;
$('btnSound').onclick = () => {
  soundOn = !soundOn;
  $('btnSound').textContent = soundOn ? 'Sound on' : 'Sound off';
  $('btnSound').classList.toggle('on', soundOn);
  if (soundOn && !audio) audio = new (window.AudioContext || window.webkitAudioContext)();
};

function beep(freq, ms) {
  if (!audio) return;
  const o = audio.createOscillator(), g = audio.createGain();
  o.type = 'sine'; o.frequency.value = freq;
  g.gain.setValueAtTime(0.0001, audio.currentTime);
  g.gain.exponentialRampToValueAtTime(0.12, audio.currentTime + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + ms / 1000);
  o.connect(g).connect(audio.destination);
  o.start(); o.stop(audio.currentTime + ms / 1000 + 0.02);
}

function warnFeedback(a) {
  const level = a.mode === 'off' ? 0 : a.level;
  const now = performance.now();
  if (soundOn && level >= 1 && now > nextBeep) {
    // caution: slow low beep; warning: faster; braking: fast, high
    const gap = [0, 900, 420, 180][level];
    beep([0, 520, 760, 1100][level], level === 3 ? 110 : 140);
    nextBeep = now + gap;
  }
  // controller rumble
  for (const p of (navigator.getGamepads ? navigator.getGamepads() : [])) {
    if (!p || !p.vibrationActuator || !p.vibrationActuator.playEffect) continue;
    if (level >= 2 && now - (state.lastRumble || 0) > 250) {
      state.lastRumble = now;
      p.vibrationActuator.playEffect('dual-rumble', { duration: 260, weakMagnitude: level === 3 ? 1.0 : 0.5,
        strongMagnitude: level === 3 ? 0.9 : 0.2 });
    }
  }
}


// ---------------------------------------------------------------- floating labels over tracked movers
const tagPool = [];
function updateTags(s) {
  let n = 0;
  for (const t of s.tracks) {
    if (!t.moving) continue;
    const p = world.toScreen(t.x, t.y, 0.3);
    if (!p) continue;
    let el = tagPool[n];
    if (!el) {
      el = document.createElement('div');
      el.className = 'tag';
      document.body.appendChild(el);
      tagPool.push(el);
    }
    const sp = Math.hypot(t.vx, t.vy);
    el.textContent = `MOVING  ${sp.toFixed(2)} m/s`;
    el.style.transform = `translate(${p[0]}px, ${p[1]}px) translate(-50%, -100%)`;
    el.style.display = 'block';
    n++;
  }
  for (let i = n; i < tagPool.length; i++) tagPool[i].style.display = 'none';
}
