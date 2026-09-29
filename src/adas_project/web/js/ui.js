// UI helpers: rolling charts, gauges, tuning sliders, results tab.

export class LineChart {
  constructor(canvas, series, opts = {}) {
    this.canvas = canvas;
    this.series = series;                    // [{key, color, label, width}]
    this.window = opts.window ?? 12;         // seconds
    this.min = opts.min ?? 0;
    this.max = opts.max ?? 1;
    this.auto = opts.auto ?? false;
    this.data = series.map(() => []);
    this.times = [];
    this.legend = opts.legend !== false;
    if (this.legend) {
      const l = document.createElement('div');
      l.className = 'legend';
      l.innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.label}</span>`).join('');
      canvas.parentElement.insertBefore(l, canvas);
    }
  }

  push(t, values) {
    this.times.push(t);
    values.forEach((v, i) => this.data[i].push(v));
    while (this.times.length && t - this.times[0] > this.window) {
      this.times.shift();
      this.data.forEach((d) => d.shift());
    }
  }

  clear() { this.times = []; this.data = this.series.map(() => []); }

  draw() {
    const c = this.canvas, dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.round(w * dpr)) { c.width = Math.round(w * dpr); c.height = Math.round(h * dpr); }
    const g = c.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const pad = { l: 26, r: 4, t: 4, b: 14 };
    const pw = w - pad.l - pad.r, ph = h - pad.t - pad.b;

    let lo = this.min, hi = this.max;
    if (this.auto) {
      hi = Math.max(this.max, ...this.data.flat().filter(Number.isFinite));
    }
    g.font = '10px system-ui, sans-serif';
    g.fillStyle = 'rgba(138,155,194,0.9)';
    g.strokeStyle = 'rgba(148,178,255,0.10)';
    g.lineWidth = 1;
    for (let i = 0; i <= 3; i++) {
      const y = pad.t + ph * (i / 3);
      g.beginPath(); g.moveTo(pad.l, y); g.lineTo(w - pad.r, y); g.stroke();
      const val = hi - (hi - lo) * (i / 3);
      g.fillText(val.toFixed(hi > 3 ? 0 : 1), 1, y + 3);
    }
    if (!this.times.length) return;
    const t1 = this.times[this.times.length - 1], t0 = t1 - this.window;
    const X = (t) => pad.l + ((t - t0) / this.window) * pw;
    const Y = (v) => pad.t + ph * (1 - (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo || 1));

    this.series.forEach((s, si) => {
      const d = this.data[si];
      g.beginPath();
      let started = false;
      for (let i = 0; i < d.length; i++) {
        if (!Number.isFinite(d[i])) { started = false; continue; }
        const x = X(this.times[i]), y = Y(d[i]);
        if (!started) { g.moveTo(x, y); started = true; } else g.lineTo(x, y);
      }
      g.strokeStyle = s.color;
      g.lineWidth = s.width ?? 2;
      g.lineJoin = 'round';
      g.stroke();
      if (s.fill && d.length) {
        g.lineTo(X(this.times[d.length - 1]), Y(lo));
        g.lineTo(X(this.times[0]), Y(lo));
        g.closePath();
        const grad = g.createLinearGradient(0, pad.t, 0, pad.t + ph);
        grad.addColorStop(0, s.color + '55'); grad.addColorStop(1, s.color + '00');
        g.fillStyle = grad; g.fill();
      }
    });
  }
}

export function drawSpeedGauge(el, v, vmax = 1.2, level = 0) {
  const colours = ['#34d399', '#facc15', '#fb923c', '#ef4444'];
  const r = 44, cx = 66, cy = 62;
  const a0 = Math.PI * 0.8, a1 = Math.PI * 2.2;
  const frac = Math.min(Math.max(Math.abs(v) / vmax, 0), 1);
  const p = (a, rr) => `${cx + rr * Math.cos(a)},${cy + rr * Math.sin(a)}`;
  const arc = (from, to) => `M ${p(from, r)} A ${r} ${r} 0 ${to - from > Math.PI ? 1 : 0} 1 ${p(to, r)}`;
  let ticks = '';
  for (let i = 0; i <= 12; i++) {
    const a = a0 + (a1 - a0) * (i / 12);
    ticks += `<line x1="${cx + (r - 9) * Math.cos(a)}" y1="${cy + (r - 9) * Math.sin(a)}" x2="${cx + (r - 4) * Math.cos(a)}" y2="${cy + (r - 4) * Math.sin(a)}" stroke="rgba(148,178,255,.35)" stroke-width="1.5"/>`;
  }
  el.innerHTML = `<svg viewBox="0 0 132 96" width="132" height="96">
    <path d="${arc(a0, a1)}" stroke="rgba(255,255,255,.08)" stroke-width="8" fill="none" stroke-linecap="round"/>
    <path d="${arc(a0, a0 + (a1 - a0) * Math.max(frac, 0.005))}" stroke="${colours[level]}" stroke-width="8" fill="none" stroke-linecap="round"/>
    ${ticks}
    <text x="${cx}" y="${cy + 4}" text-anchor="middle" font-size="24" font-weight="700" fill="#e5ecff">${Math.abs(v).toFixed(2)}</text>
    <text x="${cx}" y="${cy + 19}" text-anchor="middle" font-size="9.5" fill="#8a9bc2" letter-spacing="1.5">M/S</text>
  </svg>`;
}

export function drawWheelGauge(el, steer, wl, wr) {
  const ang = -steer * 100;                 // + steer = left = counter-clockwise on screen
  el.innerHTML = `<svg viewBox="0 0 132 96" width="132" height="96">
    <g transform="translate(66 46)">
      <g transform="rotate(${ang})">
        <circle r="34" fill="none" stroke="rgba(255,255,255,.16)" stroke-width="7"/>
        <circle r="34" fill="none" stroke="#3b82f6" stroke-width="7" stroke-dasharray="26 190" stroke-dashoffset="-13" opacity=".9"/>
        <line x1="-34" y1="0" x2="34" y2="0" stroke="rgba(255,255,255,.28)" stroke-width="6" stroke-linecap="round"/>
        <line x1="0" y1="0" x2="0" y2="34" stroke="rgba(255,255,255,.28)" stroke-width="6" stroke-linecap="round"/>
        <circle r="7" fill="#1e293b" stroke="#60a5fa" stroke-width="2"/>
      </g>
    </g>
    <text x="66" y="93" text-anchor="middle" font-size="10" fill="#8a9bc2">L ${(wl * 57.3).toFixed(0)}°   R ${(wr * 57.3).toFixed(0)}°</text>
  </svg>`;
}

export function buildTuning(box, spec, onChange) {
  const groups = {};
  spec.forEach((s) => (groups[s.group] ||= []).push(s));
  box.innerHTML = '';
  for (const [name, items] of Object.entries(groups)) {
    const g = document.createElement('div');
    g.className = 'group';
    const sim = /Reality|Virtual/.test(name);
    g.innerHTML = `<h5 class="${sim ? 'sim' : ''}">${name}</h5>`;
    for (const s of items) {
      const row = document.createElement('div');
      row.className = 'slider';
      const dec = s.step < 0.01 ? 3 : s.step < 0.1 ? 2 : s.step < 1 ? 1 : 0;
      row.innerHTML = `<div class="top"><span>${s.label}</span><b>${s.value.toFixed(dec)} ${s.unit}</b></div>
        <input type="range" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.value}">
        ${s.help ? `<small>${s.help}</small>` : ''}`;
      const input = row.querySelector('input'), label = row.querySelector('b');
      input.addEventListener('input', () => {
        label.textContent = `${(+input.value).toFixed(dec)} ${s.unit}`;
        onChange(s.id, +input.value);
      });
      g.appendChild(row);
    }
    box.appendChild(g);
  }
}

export function renderIntent(el, probs) {
  const names = { straight: 'Straight', left: 'Left', right: 'Right', brake: 'Ease off' };
  el.innerHTML = Object.keys(names).map((k) => {
    const p = probs[k] ?? 0;
    return `<div class="bar"><span>${names[k]}</span><div class="track"><div class="fill" style="width:${(p * 100).toFixed(0)}%"></div></div><b>${(p * 100).toFixed(0)}%</b></div>`;
  }).join('');
}

export function renderEvents(el, events) {
  el.innerHTML = [...events].reverse().map((e) =>
    `<div class="ev ${e.kind}"><time>${e.t.toFixed(1)}s</time><span>${e.text}</span></div>`).join('');
}

// ---------------------------------------------------------------- results tab
function bars(rows, colours, w = 330, h = 120, labels = []) {
  const n = rows[0].length, groups = rows.length;
  const max = Math.max(1, ...rows.flat());
  const gw = (w - 30) / n, bw = Math.min(26, gw / (groups + 0.6));
  let svg = `<svg viewBox="0 0 ${w} ${h + 18}" width="100%">`;
  rows.forEach((row, gi) => {
    row.forEach((v, i) => {
      const x = 28 + i * gw + gi * (bw + 2), bh = (v / max) * (h - 12);
      svg += `<rect x="${x}" y="${h - bh}" width="${bw}" height="${bh}" rx="3" fill="${colours[gi]}"/>`;
      svg += `<text x="${x + bw / 2}" y="${h - bh - 3}" text-anchor="middle" font-size="9" fill="#cbd5e1">${v}</text>`;
    });
  });
  labels.forEach((l, i) => {
    svg += `<text x="${28 + i * gw + gw / 2 - 6}" y="${h + 13}" text-anchor="middle" font-size="9.5" fill="#8a9bc2">${l}</text>`;
  });
  return svg + '</svg>';
}

function curveSvg(curves, w = 330, h = 170) {
  const cols = { physics: '#94a3b8', blend: '#facc15', adaptive: '#2dd4bf' };
  const names = { physics: 'physics-only', blend: 'intent blend', adaptive: 'adaptive (learned)' };
  const maxFa = Math.max(2, ...Object.values(curves).flatMap((c) => c.map((r) => r.false_alarms_per_min)));
  const X = (v) => 34 + (v / maxFa) * (w - 44), Y = (v) => 8 + (1 - v) * (h - 30);
  let s = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
  for (let i = 0; i <= 4; i++) {
    s += `<line x1="34" x2="${w - 10}" y1="${Y(i / 4)}" y2="${Y(i / 4)}" stroke="rgba(148,178,255,.12)"/>`;
    s += `<text x="4" y="${Y(i / 4) + 3}" font-size="9" fill="#8a9bc2">${(i * 25)}%</text>`;
  }
  for (const [name, rows] of Object.entries(curves)) {
    const pts = rows.slice().sort((a, b) => a.false_alarms_per_min - b.false_alarms_per_min)
      .map((r) => `${X(r.false_alarms_per_min)},${Y(r.detection)}`).join(' ');
    s += `<polyline points="${pts}" fill="none" stroke="${cols[name]}" stroke-width="2.4" stroke-linejoin="round"/>`;
  }
  s += `<text x="${w / 2}" y="${h - 4}" text-anchor="middle" font-size="9.5" fill="#8a9bc2">false alarms per minute →</text></svg>`;
  s += `<div class="legend">${Object.keys(curves).map((k) => `<span><i style="background:${cols[k]}"></i>${names[k]}</span>`).join('')}</div>`;
  return s;
}


function lineSvg(series, w = 330, h = 120, xLabel = '', yMax = null) {
  const xs = series.flatMap((s) => s.pts.map((p) => p[0])), ys = series.flatMap((s) => s.pts.map((p) => p[1]));
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = yMax ?? Math.max(1, ...ys), y0 = 0;
  const X = (v) => 34 + ((v - x0) / (x1 - x0 || 1)) * (w - 44), Y = (v) => 8 + (1 - (v - y0) / (y1 - y0 || 1)) * (h - 30);
  let svg = `<svg viewBox="0 0 ${w} ${h}" width="100%">`;
  for (let i = 0; i <= 3; i++) {
    const v = y0 + (y1 - y0) * (i / 3);
    svg += `<line x1="34" x2="${w - 10}" y1="${Y(v)}" y2="${Y(v)}" stroke="rgba(148,178,255,.12)"/><text x="4" y="${Y(v) + 3}" font-size="9" fill="#8a9bc2">${v.toFixed(v < 10 ? 1 : 0)}</text>`;
  }
  for (const s of series) {
    svg += `<polyline points="${s.pts.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')}" fill="none" stroke="${s.color}" stroke-width="2.4" stroke-linejoin="round"/>`;
    for (const p of s.pts) svg += `<circle cx="${X(p[0])}" cy="${Y(p[1])}" r="2.4" fill="${s.color}"/>`;
  }
  svg += `<text x="${w / 2}" y="${h - 4}" text-anchor="middle" font-size="9.5" fill="#8a9bc2">${xLabel}</text></svg>`;
  return svg;
}

export async function renderResults(box) {
  let data;
  try {
    const r = await fetch('data/results.json', { cache: 'no-store' });
    data = await r.json();
  } catch (e) {
    box.innerHTML = '<p class="hint">No results yet. Run <code>python -m sim.report</code> to generate them.</p>';
    return;
  }
  let html = '';
  if (data.scenarios) {
    html += `<div class="chart-card"><h4>Scenario suite <em>${data.scenarios.passed}/${data.scenarios.total} passed</em></h4>
      <table class="res"><tr><th>Scenario</th><th>ADAS off</th><th>ADAS on</th></tr>` +
      data.scenarios.rows.map((r) => `<tr><td>${r.name}</td><td class="${r.off_crash ? 'bad' : 'ok'}">${r.off}</td><td class="${r.pass ? 'ok' : 'bad'}">${r.on}</td></tr>`).join('') +
      `</table></div>`;
  }
  if (data.monte_carlo) {
    const m = data.monte_carlo;
    html += `<div class="chart-card"><h4>Random stress test <em>${m.runs} runs</em></h4>
      <div class="legend"><span><i style="background:#ef4444"></i>ADAS off</span><span><i style="background:#2dd4bf"></i>ADAS on</span></div>
      ${bars([m.bins.map((b) => b.off), m.bins.map((b) => b.on)], ['#ef4444', '#2dd4bf'], 330, 110, m.bins.map((b) => b.label))}
      <div class="hint" style="margin:4px 0 0">Crashes by throttle level (PWM). Total: ${m.total_off} without ADAS, ${m.total_on} with.</div></div>`;
  }
  if (data.warning) {
    const wv = data.warning;
    html += `<div class="chart-card"><h4>Does intent help the warnings? <em>held-out drives</em></h4>${curveSvg(wv.curves)}
      <div class="hint" style="margin:4px 0 0">${wv.summary || ''}</div></div>`;
  }
  if (data.intent) {
    const i = data.intent;
    html += `<div class="chart-card"><h4>Intent model <em>held-out drives</em></h4>
      <table class="res"><tr><th></th><th>Accuracy</th><th>Macro F1</th></tr>
      <tr><td>Keep-doing-the-same baseline</td><td>${(i.baseline.accuracy * 100).toFixed(1)}%</td><td>${i.baseline.macro_f1.toFixed(2)}</td></tr>
      <tr><td>Learned model</td><td class="ok">${(i.learned.accuracy * 100).toFixed(1)}%</td><td class="ok">${i.learned.macro_f1.toFixed(2)}</td></tr></table></div>`;
  }

  if (data.lane) {
    const names = { off: 'No assist', warn: 'Warning only', assist: 'Lane-keeping assist' };
    html += `<div class="chart-card"><h4>Lane keeping <em>curving road, wheel left alone</em></h4><table class="res">
      <tr><th></th><th>Worst drift</th><th>Warned</th></tr>` +
      data.lane.map((r) => `<tr><td>${names[r.mode]}</td><td class="${r.worst_offset_cm < 12 ? 'ok' : 'bad'}">${r.worst_offset_cm.toFixed(0)} cm</td><td>${(r.warning_share * 100).toFixed(0)}%</td></tr>`).join('') +
      `</table><div class="hint" style="margin:4px 0 0">Wheels touch the tape at about 15 cm from the lane centre.</div></div>`;
  }
  if (data.faults) {
    const n = data.faults.filter((f) => f.pass).length;
    html += `<div class="chart-card"><h4>Fault injection <em>${n}/${data.faults.length} fail safe</em></h4><table class="res">` +
      data.faults.map((f) => `<tr><td class="${f.pass ? 'ok' : 'bad'}">${f.pass ? 'PASS' : 'FAIL'}</td><td>${f.name}<br><span style="color:#8a9bc2;font-size:10.5px">${f.detail}</span></td></tr>`).join('') +
      `</table></div>`;
  }
  if (data.stopping) {
    html += `<div class="chart-card"><h4>Emergency braking <em>final gap to the wall</em></h4>
      ${lineSvg([{ color: '#2dd4bf', pts: data.stopping.map((q) => [q.speed, q.gap_cm]) }], 330, 120, 'speed when the wall appeared (m/s)')}
      <div class="hint" style="margin:4px 0 0">Stops short of the wall (cm) at every tested speed, no crashes.</div></div>`;
  }
  if (data.parking) {
    const p = data.parking;
    html += `<div class="chart-card"><h4>Auto-parking <em>${p.runs} random starts</em></h4>
      <table class="res"><tr><td>Parked correctly</td><td class="ok">${p.success}/${p.runs}</td></tr>
      <tr><td>Crashes</td><td class="${p.crashes ? 'bad' : 'ok'}">${p.crashes}</td></tr>
      <tr><td>Lateral error (median / 95th)</td><td>${p.median_lateral_cm?.toFixed(1)} / ${p.p95_lateral_cm?.toFixed(1)} cm</td></tr>
      <tr><td>Yaw error (median / 95th)</td><td>${p.median_yaw_deg?.toFixed(1)} / ${p.p95_yaw_deg?.toFixed(1)}&deg;</td></tr>
      <tr><td>Time (median)</td><td>${p.median_time_s?.toFixed(0)} s</td></tr></table></div>`;
  }
  if (data.bypass) {
    const row = (name, s) => `<tr><td>${name}</td><td class="${s.success === s.runs ? 'ok' : 'bad'}">${s.success}/${s.runs}</td>
      <td class="${s.crashes ? 'bad' : 'ok'}">${s.crashes}</td><td>${s.median_lateral_cm?.toFixed(1)} / ${s.p95_lateral_cm?.toFixed(1)} cm</td>
      <td>${s.median_clearance_cm?.toFixed(1)} cm (min ${s.min_clearance_cm?.toFixed(1)})</td></tr>`;
    html += `<div class="chart-card"><h4>Obstacle bypass <em>random obstacles, walls and gaps</em></h4>
      <table class="res"><tr><th></th><th>ok</th><th>crash</th><th>final lateral (median / 95th)</th><th>clearance</th></tr>
      ${row('default car', data.bypass.sim)}${row('real car (measured)', data.bypass.real)}</table>
      <div class="hint" style="margin:4px 0 0">ok = rejoined the line within 6 cm and 4&deg;, or correctly refused a gap that is too narrow; ${data.bypass.real.refused} of ${data.bypass.real.runs} runs were refusals.</div></div>`;
  }
  if (data.sweeps) {
    html += `<div class="chart-card"><h4>Robustness <em>crash rate %, ADAS on vs off</em></h4>` +
      Object.values(data.sweeps).map((d) => `<div class="hint" style="margin:8px 0 0">${d.label}</div>` +
        lineSvg([{ color: '#ef4444', pts: d.rows.map((r) => [r.value, r.off]) }, { color: '#2dd4bf', pts: d.rows.map((r) => [r.value, r.on]) }], 330, 90, '', 100)).join('') +
      `<div class="hint" style="margin:6px 0 0">Flat teal line = the parameter barely matters. A rising one is worth calibrating carefully.</div></div>`;
  }
  box.innerHTML = html || '<p class="hint">Results file is empty.</p>';
}
