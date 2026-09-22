// Monte Carlo lab: many random drives replayed side by side, sped up.
// In every cell the RED car has the ADAS off and the TEAL car has it on; both get the identical driver input.

const COLORS = ['#34d399', '#facc15', '#fb923c', '#ef4444'];
const X0 = -1.2, X1 = 6.2, Y0 = -2.2, Y1 = 2.2;

export class MonteCarloLab {
  constructor(els, send) {
    this.el = els.root;
    this.canvas = els.canvas;
    this.stats = els.stats;
    this.status = els.status;
    this.send = send;
    this.cases = [];
    this.speed = 6;
    this.t0 = 0;
    this.playing = false;
    this.raf = 0;
    this.finished = false;
  }

  open() { this.el.classList.add('show'); this.resize(); if (!this.cases.length) this.run(24); else this.play(); }
  close() { this.el.classList.remove('show'); this.playing = false; cancelAnimationFrame(this.raf); }
  run(n) {
    this.playing = false;
    this.cases = [];
    this.status.textContent = `simulating ${n} random drives, each twice (ADAS off and on) ...`;
    this.send({ type: 'cmd', cmd: 'mc', n });
  }
  setStatus(t) { this.status.textContent = t; }
  setSpeed(v) { this.speed = v; }

  load(cases) {
    this.cases = cases;
    this.status.textContent = `${cases.length} drives`;
    this.play();
  }

  play() {
    this.t0 = performance.now();
    this.playing = true;
    this.finished = false;
    cancelAnimationFrame(this.raf);
    const loop = () => { if (!this.playing) return; this.draw(); this.raf = requestAnimationFrame(loop); };
    loop();
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
  }

  layout(n, w, h) {
    let best = null;
    for (let cols = 1; cols <= n; cols++) {
      const rows = Math.ceil(n / cols);
      const cw = w / cols, ch = h / rows;
      const s = Math.min(cw / 1.7, ch);
      if (!best || s > best.s) best = { cols, rows, s, cw, ch };
    }
    return best;
  }

  at(traj, t) {
    // trajectory sample nearest to (and not after) time t
    let lo = 0, hi = traj.length - 1;
    while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (traj[mid][0] <= t) lo = mid; else hi = mid - 1; }
    return lo;
  }

  draw() {
    const c = this.canvas, g = c.getContext('2d'), dpr = window.devicePixelRatio || 1;
    if (c.width !== Math.round(c.clientWidth * dpr) || c.height !== Math.round(c.clientHeight * dpr)) this.resize();
    const W = c.clientWidth, H = c.clientHeight;
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, c.width, c.height);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = this.cases.length;
    if (!n) return;
    const L = this.layout(n, W, H);
    const t = ((performance.now() - this.t0) / 1000) * this.speed;

    let offCrash = 0, onCrash = 0, done = 0;
    this.cases.forEach((cs, i) => {
      const col = i % L.cols, row = Math.floor(i / L.cols);
      const cellW = L.cw - 10, cellH = L.ch - 10;
      const k = Math.min(cellW / (X1 - X0), cellH / (Y1 - Y0));
      const ox = col * L.cw + 5 + (cellW - k * (X1 - X0)) / 2;
      const oy = row * L.ch + 5 + (cellH - k * (Y1 - Y0)) / 2;
      const P = (x, y) => [ox + (x - X0) * k, oy + (Y1 - y) * k];

      // cell background
      g.fillStyle = 'rgba(255,255,255,0.035)';
      g.strokeStyle = 'rgba(148,178,255,0.16)';
      g.lineWidth = 1;
      g.beginPath(); g.roundRect(ox - 3, oy - 3, k * (X1 - X0) + 6, k * (Y1 - Y0) + 6, 8); g.fill(); g.stroke();

      // static objects
      for (const o of cs.static) {
        if (o.k === 'wall') {
          const [a, b] = P(o.x1, o.y1), [c2, d] = P(o.x2, o.y2);
          g.strokeStyle = 'rgba(203,213,225,0.8)'; g.lineWidth = Math.max(1, k * 0.03);
          g.beginPath(); g.moveTo(a, b); g.lineTo(c2, d); g.stroke();
        } else if (o.k === 'box') {
          const [px, py] = P(o.cx, o.cy);
          g.save(); g.translate(px, py); g.rotate(-o.hd);
          g.fillStyle = '#b98555'; g.fillRect(-o.l * k / 2, -o.w * k / 2, o.l * k, o.w * k);
          g.restore();
        } else if (o.k === 'cone') {
          const [px, py] = P(o.x, o.y);
          g.fillStyle = '#ff6a1a'; g.beginPath(); g.arc(px, py, Math.max(2, o.r * k * 1.5), 0, 6.283); g.fill();
        }
      }

      const tEnd = Math.max(cs.off[cs.off.length - 1][0], cs.on[cs.on.length - 1][0]);
      const tt = Math.min(t, tEnd + 0.05);
      const iOn = this.at(cs.on, tt), iOff = this.at(cs.off, tt);

      // moving objects (pedestrians) follow the ADAS-on run's timeline
      const dyn = cs.dyn[Math.min(iOn, cs.dyn.length - 1)] || [];
      for (const [dx, dy] of dyn) {
        const [px, py] = P(dx, dy);
        g.fillStyle = '#60a5fa'; g.beginPath(); g.arc(px, py, Math.max(2.5, 0.06 * k), 0, 6.283); g.fill();
      }

      const trail = (traj, idx, colour) => {
        g.strokeStyle = colour; g.lineWidth = Math.max(1.4, k * 0.03); g.lineJoin = 'round';
        g.beginPath();
        for (let j = 0; j <= idx; j++) { const [px, py] = P(traj[j][1], traj[j][2]); j ? g.lineTo(px, py) : g.moveTo(px, py); }
        g.stroke();
      };
      const car = (traj, idx, colour) => {
        const s = traj[idx];
        const [px, py] = P(s[1], s[2]);
        g.save(); g.translate(px, py); g.rotate(-s[3]);
        g.fillStyle = colour; g.strokeStyle = 'rgba(0,0,0,.6)'; g.lineWidth = 1;
        g.beginPath(); g.roundRect(-0.05 * k, -0.07 * k, 0.31 * k, 0.14 * k, 2); g.fill(); g.stroke();
        g.restore();
      };
      trail(cs.off, iOff, 'rgba(239,68,68,0.55)');
      trail(cs.on, iOn, 'rgba(45,212,191,0.75)');
      car(cs.off, iOff, '#ef4444');
      car(cs.on, iOn, COLORS[Math.max(cs.on[iOn][4], 0)] === '#34d399' ? '#2dd4bf' : COLORS[cs.on[iOn][4]]);

      // crash markers and counters
      const offEnd = cs.off[cs.off.length - 1][0], onEnd = cs.on[cs.on.length - 1][0];
      if (cs.off_crash && t >= offEnd) {
        offCrash++;
        const s = cs.off[cs.off.length - 1]; const [px, py] = P(s[1], s[2]);
        g.strokeStyle = '#ff3b3b'; g.lineWidth = 2.5;
        const r = Math.max(6, k * 0.22);
        g.beginPath(); g.moveTo(px - r, py - r); g.lineTo(px + r, py + r); g.moveTo(px + r, py - r); g.lineTo(px - r, py + r); g.stroke();
      }
      if (cs.on_crash && t >= onEnd) onCrash++;
      if (t >= tEnd) done++;

      g.fillStyle = 'rgba(203,213,225,0.55)'; g.font = '10px system-ui';
      g.fillText(`#${cs.seed}  PWM ${cs.pwm}`, ox + 4, oy + 12);
    });

    const pct = (a, b) => (b ? Math.round(100 * a / b) : 0);
    this.stats.innerHTML = `
      <div class="mcstat"><small>Drives finished</small><b>${done}/${n}</b></div>
      <div class="mcstat off"><small>Crashes, ADAS off</small><b>${offCrash}</b><em>${pct(offCrash, n)}%</em></div>
      <div class="mcstat on"><small>Crashes, ADAS on</small><b>${onCrash}</b><em>${pct(onCrash, n)}%</em></div>
      <div class="mcstat"><small>Crash reduction</small><b>${offCrash ? Math.round(100 * (offCrash - onCrash) / offCrash) : 0}%</b></div>`;
    if (done >= n && !this.finished) { this.finished = true; }
  }
}
