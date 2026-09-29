"""Simulator lab tabs for the dashboard (gui/dashboard.py): they run jobs on the digital twin as child processes and
plot their live logs, so nothing here needs the car or the relay.

  MonteCarloTab  (TODO M2) the Monte Carlo of the car's own decision code (sim/relay_mc.py --live): pick the systems
                 (driver only / brake only / ADAS / ADAS + intent), driver styles, runs and intent model; live summary
                 table, crash and needless-intervention bars, paired Wilcoxon tests, and a replay of any scenario with
                 every system's path drawn over the room.
  TrainingTab    (TODO M3) digital-twin ML training of the driver-intent model: the twin's identified parameters and
                 the domain-randomisation ranges, data generation (sim/twin_intent_data.py), GPU training
                 (sim/train_intent_torch.py) with live loss / average-precision curves per model, then the per-situation
                 comparison with the old model, a reliability diagram and risk-over-time on held-out drives.
"""
import json
import os
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
MC_DIR = os.path.join(ROOT, "models", "mc_live")
TRAIN_DIR = os.path.join(ROOT, "models", "intent_training")
from gui import theme
COLS = [theme.C[k] for k in ("s1", "s2", "s3", "s5", "s4", "s6")] + ["#B48EAD", "#88C0D0"]
VARIANT_LABEL = {"off": "driver only", "brake-only": "brake only", "adas": "ADAS", "adas+intent": "ADAS + intent"}
VARIANT_COL = {"off": theme.C["bad"], "brake-only": theme.C["warn"], "adas": theme.C["s1"], "adas+intent": theme.C["accent"]}


def _refresh_palette():
    COLS[:] = [theme.C[k] for k in ("s1", "s2", "s3", "s5", "s4", "s6")] + ["#B48EAD", "#88C0D0"]
    VARIANT_COL.update({"off": theme.C["bad"], "brake-only": theme.C["warn"], "adas": theme.C["s1"], "adas+intent": theme.C["accent"]})


theme.on_change(_refresh_palette)


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


class JsonlTail:
    """Reads the lines appended to a JSON-lines file since the last call."""

    def __init__(self, path):
        self.path, self.pos = path, 0

    def reset(self):
        self.pos = 0

    def new(self):
        out = []
        try:
            with open(self.path, "rb") as f:
                if os.fstat(f.fileno()).st_size < self.pos:
                    self.pos = 0                   # the file was restarted
                f.seek(self.pos)
                data = f.read()
        except OSError:
            return out
        *complete, partial = data.split(b"\n")
        self.pos += len(data) - len(partial)       # an unfinished last line is read next time
        for line in complete:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
        return out


class Job(QtCore.QObject):
    """A child process (python -m ...) whose output goes to a text box."""

    finished = QtCore.Signal()

    def __init__(self, console):
        super().__init__()
        self.console = console
        self.proc = None

    def running(self):
        return self.proc is not None and self.proc.state() != QtCore.QProcess.NotRunning

    def start(self, args, label):
        if self.running():
            return False
        self.proc = QtCore.QProcess()
        self.proc.setWorkingDirectory(ROOT)
        self.proc.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._out)
        self.proc.finished.connect(lambda *_: (self.console.appendPlainText(f"--- {label} finished"),
                                               self.finished.emit()))
        self.console.appendPlainText(f"--- {label}: python {' '.join(args)}")
        self.proc.start(sys.executable, ["-u"] + args)
        return True

    def stop(self):
        if self.running():
            self.proc.kill()

    def _out(self):
        text = bytes(self.proc.readAllStandardOutput()).decode(errors="replace")
        for line in text.replace("\r", "\n").splitlines():
            if line.strip():
                self.console.appendPlainText(line)


def small_label(text, dim=True):
    lab = QtWidgets.QLabel(text)
    lab.setWordWrap(True)
    if dim:
        lab.setStyleSheet(f"color: {theme.C['dim']}; font-size: 11px;")
    return lab


def plot(title, left=None, bottom=None):
    p = pg.PlotWidget(title=title)
    theme.style_plot(p, title)
    p.addLegend(offset=(5, 5), labelTextSize="8pt")
    for ax in ("left", "bottom"):
        p.getAxis(ax).enableAutoSIPrefix(False)
    if left:
        p.setLabel("left", left)
    if bottom:
        p.setLabel("bottom", bottom)
    return p


# ====================================================================== Monte Carlo
class MonteCarloTab(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.runs, self.tail = [], JsonlTail(os.path.join(MC_DIR, "runs.jsonl"))
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        pages = QtWidgets.QTabWidget()
        outer.addWidget(pages)
        run_page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(run_page)
        lay.addWidget(small_label(
            "The Monte Carlo drives the car's OWN decision code (relay assists, path gate, evasive steer, intent model) "
            "on the digital twin: the same room, driver and attention lapses are driven by every system, and a "
            "needless intervention is one where the same driver, left alone, would not have come within 2 cm of "
            "anything in the next 2 s (counterfactual).", dim=True))
        ctl = QtWidgets.QHBoxLayout()
        self.var_boxes = {}
        for v in ("off", "brake-only", "adas", "adas+intent"):
            b = QtWidgets.QCheckBox(VARIANT_LABEL[v])
            b.setChecked(v != "brake-only")
            b.setStyleSheet(f"color: {VARIANT_COL[v]};")
            self.var_boxes[v] = b
            ctl.addWidget(b)
        ctl.addSpacing(16)
        self.style_boxes = {}
        for s in ("lapsing", "late", "good", "distracted", "aggressive"):
            b = QtWidgets.QCheckBox(s)
            b.setChecked(s in ("lapsing", "late"))
            self.style_boxes[s] = b
            ctl.addWidget(b)
        ctl.addSpacing(16)
        ctl.addWidget(QtWidgets.QLabel("rooms"))
        self.n_runs = QtWidgets.QSpinBox()
        self.n_runs.setRange(1, 200)
        self.n_runs.setValue(12)
        ctl.addWidget(self.n_runs)
        self.model = QtWidgets.QComboBox()
        self.model.addItem("intent v2 (car until now)", os.path.join(ROOT, "models", "intent_net.json"))
        if os.path.exists(os.path.join(ROOT, "models", "intent_v3.json")):
            self.model.addItem("intent v3 (twin-trained)", os.path.join(ROOT, "models", "intent_v3.json"))
        ctl.addWidget(self.model)
        self.run_btn = QtWidgets.QPushButton("Run Monte Carlo")
        self.run_btn.clicked.connect(self.start)
        stop = QtWidgets.QPushButton("Stop")
        ctl.addWidget(self.run_btn)
        ctl.addWidget(stop)
        ctl.addStretch(1)
        lay.addLayout(ctl)
        self.progress = QtWidgets.QProgressBar()
        lay.addWidget(self.progress)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        lay.addWidget(self.console, 1)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left)
        self.table = QtWidgets.QTableWidget()
        self.table.setMinimumHeight(170)
        ll.addWidget(self.table)
        from gui.controls import Card
        tcard = Card(padding=12)
        self.tests = small_label("", dim=False)
        self.tests.setStyleSheet(f"color: {theme.C['text2']}; font-size: 12px; font-family: 'Consolas', monospace;")
        tcard.body.addWidget(self.tests)
        ll.addWidget(tcard)
        self.bars = plot("crashes (red) and needless interventions (orange) per system")
        ll.addWidget(self.bars, 1)
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("replay"))
        self.scen = QtWidgets.QComboBox()
        self.scen.currentIndexChanged.connect(self.draw_scenario)
        row.addWidget(self.scen, 1)
        rl.addLayout(row)
        self.map = plot("every system in the same room: x crash, star goal, dots needed / needless")
        self.map.setAspectLocked(True)
        rl.addWidget(self.map, 1)
        pages.addTab(left, "RESULTS")
        pages.addTab(right, "REPLAY MAP")
        pages.addTab(run_page, "RUN")
        self.pages = pages
        self.job = Job(self.console)
        stop.clicked.connect(self.job.stop)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1000)
        self.load_existing()

    def start(self):
        variants = [v for v, b in self.var_boxes.items() if b.isChecked()]
        styles = [s for s, b in self.style_boxes.items() if b.isChecked()]
        if not variants or not styles:
            return
        self.runs = []
        self.scen.clear()
        self.tail.reset()
        self.job.start(["-m", "sim.relay_mc", "--live", str(self.n_runs.value()), ",".join(variants), ",".join(styles),
                        self.model.currentData()], "Monte Carlo")

    def load_existing(self):
        self.poll()

    def poll(self):
        new = self.tail.new()
        if new:
            self.runs += new
            keys = sorted({(r["seed"], r["style"]) for r in self.runs})
            have = {self.scen.itemData(i) for i in range(self.scen.count())}
            for k in keys:
                if k not in have:
                    self.scen.addItem(f"room {k[0]} - {k[1]} driver", k)
        st = read_json(os.path.join(MC_DIR, "status.json"), {})
        if st:
            self.progress.setRange(0, max(1, st.get("total", 1)))
            self.progress.setValue(st.get("done", 0))
            self.progress.setFormat(f"{st.get('state', '')}  %v / %m runs")
            self.show_summary(st.get("summary") or {}, st.get("tests") or {})
        if new and self.scen.currentIndex() >= 0:
            self.draw_scenario()

    def show_summary(self, summ, tests):
        cols = [("runs", "runs"), ("crashes", "crashes"), ("reached_goal", "reached goal"),
                ("interventions", "interventions"), ("needless_takeovers", "needless takeovers"),
                ("needless_brakes", "needless brakes"), ("needless_limits", "needless speed limits"),
                ("overridden_needlessly_s", "overridden needlessly (s)"), ("median_time_s", "median time (s)"),
                ("min_clearance_cm", "min clearance (cm)")]
        vs = list(summ)
        self.table.setRowCount(len(vs))
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels([c[1] for c in cols])
        self.table.setVerticalHeaderLabels([VARIANT_LABEL.get(v, v) for v in vs])
        for i, v in enumerate(vs):
            for j, (k, _) in enumerate(cols):
                val = summ[v].get(k)
                txt = "-" if val is None or (isinstance(val, float) and np.isnan(val)) else \
                    (f"{val:.1f}" if isinstance(val, float) else str(val))
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        self.bars.clear()
        x = np.arange(len(vs))
        if vs:
            self.bars.addItem(pg.BarGraphItem(x=x - 0.2, height=[summ[v]["crashes"] for v in vs], width=0.38,
                                              brush=theme.C["bad"]))
            self.bars.addItem(pg.BarGraphItem(x=x + 0.2, height=[summ[v]["needless"] for v in vs], width=0.38,
                                              brush=theme.C["warn"]))
            self.bars.getAxis("bottom").setTicks([[(i, VARIANT_LABEL.get(v, v)) for i, v in enumerate(vs)]])
        if tests:
            lines = [f"Paired one-sided Wilcoxon, ADAS + intent vs ADAS ({tests.get('pairs')} paired drives):"]
            for k, t in tests.items():
                if isinstance(t, dict):
                    lines.append(f"  {k}: {t['adas']:.0f} -> {t['adas+intent']:.0f}  ({t['better']} better, "
                                 f"{t['worse']} worse, p = {t['p']:.4f})")
            self.tests.setText("\n".join(lines))
        else:
            self.tests.setText("")

    def draw_scenario(self):
        k = self.scen.currentData()
        if k is None:
            return
        from sim.relay_mc import scenario
        self.map.clear()
        world, goal, _ = scenario(int(k[0]))
        for x1, y1, x2, y2 in world.segments():
            self.map.plot([x1, x2], [y1, y2], pen=pg.mkPen(theme.C["hair2"], width=2))
        self.map.plot([goal[0]], [goal[1]], pen=None, symbol="star", symbolSize=18, symbolBrush=theme.C["accent"])
        for r in self.runs:
            if (r["seed"], r["style"]) != tuple(k):
                continue
            tr = np.array(r["trace"]) if r["trace"] else np.zeros((1, 2))
            col = VARIANT_COL.get(r["variant"], theme.C["text"])
            self.map.plot(tr[:, 0], tr[:, 1], pen=pg.mkPen(col, width=2.5), name=VARIANT_LABEL.get(r["variant"]))
            if r["crashed"]:
                self.map.plot([tr[-1, 0]], [tr[-1, 1]], pen=None, symbol="x", symbolSize=16, symbolBrush=col)
            for e in r["events"]:
                self.map.plot([e[0]], [e[1]], pen=None, symbol="o", symbolSize=8,
                              symbolBrush=theme.C["warn"] if e[2] == "fp" else theme.C["ok"])


# ====================================================================== ML training
DR_LABELS = {"v_max_x": ("top speed", "x fitted"), "deadband": ("throttle dead-band", "PWM"),
             "tau_motor": ("motor lag", "s"), "coast_decel": ("coast deceleration", "m/s2"),
             "brake_decel": ("brake deceleration", "m/s2"), "delay_s": ("command delay", "s"),
             "k_curv_x": ("steering gain", "x fitted"), "servo_offset": ("servo centre error", "deg"),
             "noise": ("LiDAR range noise", "m"), "dropout": ("LiDAR dropout", "fraction"),
             "yaw_err_deg": ("LiDAR yaw-offset error", "deg"), "jitter_deg": ("vibration jitter", "deg"),
             "v_lag_s": ("speed-estimate lag", "s"), "v_noise": ("speed-estimate noise", "m/s"),
             "v_scale": ("speed-estimate scale", "x")}


class TrainingTab(QtWidgets.QWidget):
    device_found = QtCore.Signal(str)

    def __init__(self):
        super().__init__()
        self.tail = JsonlTail(os.path.join(TRAIN_DIR, "log.jsonl"))
        self.curves, self.hist = {}, {}
        self.report_mtime = None
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        pages = QtWidgets.QTabWidget()
        outer.addWidget(pages)
        run_page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(run_page)
        lay.addWidget(small_label(
            "The twin is the car model identified from real drive logs; every simulated drive draws a different car "
            "and sensor around it (domain randomisation), so the real car - vibration and all - is just one more draw. "
            "Labels come from the twin itself: did this driver, left alone, touch something within 2 s?", dim=True))
        ctl = QtWidgets.QHBoxLayout()
        ctl.addWidget(QtWidgets.QLabel("training drives"))
        self.n_train = QtWidgets.QSpinBox()
        self.n_train.setRange(50, 20000)
        self.n_train.setValue(3000)
        ctl.addWidget(self.n_train)
        ctl.addWidget(QtWidgets.QLabel("test drives"))
        self.n_test = QtWidgets.QSpinBox()
        self.n_test.setRange(20, 5000)
        self.n_test.setValue(600)
        ctl.addWidget(self.n_test)
        ctl.addWidget(QtWidgets.QLabel("randomisation"))
        self.level = QtWidgets.QDoubleSpinBox()
        self.level.setRange(0.0, 2.0)
        self.level.setSingleStep(0.2)
        self.level.setValue(1.0)
        self.level.setToolTip("0 = the nominal twin only; 1 = the standard ranges; >1 = wider than standard")
        self.level.valueChanged.connect(self.show_dr)
        ctl.addWidget(self.level)
        gen = QtWidgets.QPushButton("1. Generate twin data")
        gen.clicked.connect(self.generate)
        train = QtWidgets.QPushButton("2. Train on the GPU")
        train.clicked.connect(lambda: self.train(False))
        quick = QtWidgets.QPushButton("Quick train (check)")
        quick.clicked.connect(lambda: self.train(True))
        stop = QtWidgets.QPushButton("Stop")
        for b in (gen, train, quick, stop):
            ctl.addWidget(b)
        ctl.addStretch(1)
        self.device = QtWidgets.QLabel("")
        self.device_found.connect(self.device.setText)
        ctl.addWidget(self.device)
        lay.addLayout(ctl)
        self.progress = QtWidgets.QProgressBar()
        lay.addWidget(self.progress)
        self.console = QtWidgets.QPlainTextEdit()
        self.console.setReadOnly(True)
        lay.addWidget(self.console, 1)
        twin_page = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(twin_page)
        ll.addWidget(QtWidgets.QLabel("IDENTIFIED TWIN (fitted to real drive logs)"))
        self.twin = small_label("", dim=False)
        ll.addWidget(self.twin)
        ll.addWidget(QtWidgets.QLabel("DOMAIN RANDOMISATION PER DRIVE"))
        self.dr_table = QtWidgets.QTableWidget()
        ll.addWidget(self.dr_table, 1)
        self.data_info = small_label("")
        ll.addWidget(self.data_info)
        # right: live curves and results
        self.p_loss = plot("validation loss per model (lower = better)", "BCE", "epoch")
        self.p_ap = plot("validation average precision per model (higher = better)", "AP", "epoch")
        self.p_rel = plot("calibration: predicted vs observed (diagonal = perfect)", "observed rate", "predicted P")
        self.p_ex = plot("risk over time on a held-out drive (red band: contact within 2 s)", "", "s")
        self.p_slices = plot("average precision per situation: grey = v2 (car until now), blue = v3 twin-trained")
        grid = QtWidgets.QTabWidget()                 # three pages: there is no room for five plots at once
        page = QtWidgets.QWidget()
        pl = QtWidgets.QHBoxLayout(page)
        pl.addWidget(self.p_loss)
        pl.addWidget(self.p_ap)
        grid.addTab(page, "training curves")
        page = QtWidgets.QWidget()
        pl = QtWidgets.QHBoxLayout(page)
        pl.addWidget(self.p_rel, 1)
        exw = QtWidgets.QWidget()
        el = QtWidgets.QVBoxLayout(exw)
        el.setContentsMargins(0, 0, 0, 0)
        self.ex_sel = QtWidgets.QComboBox()
        self.ex_sel.currentIndexChanged.connect(self.draw_example)
        el.addWidget(self.ex_sel)
        el.addWidget(self.p_ex, 1)
        pl.addWidget(exw, 1)
        grid.addTab(page, "calibration + example drives")
        grid.addTab(self.p_slices, "per situation (v2 vs v3)")
        self.summary = small_label("", dim=False)
        self.summary.hide()                            # the KPI tiles above the tabs replace this text
        pages.addTab(grid, "RESULTS")
        pages.addTab(twin_page, "TWIN && RANDOMISATION")
        pages.addTab(run_page, "RUN")
        self.pages = pages
        self.job = Job(self.console)
        stop.clicked.connect(self.job.stop)
        self.examples = []
        self.show_twin()
        self.show_dr()
        QtCore.QTimer.singleShot(200, self.check_device)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(1000)
        self.poll()

    # --- left column
    def show_twin(self):
        try:
            from sim.hw_sim import load_car_model
            m = load_car_model()
            self.twin.setText(f"top speed {m['v_max']:.3f} m/s, dead-band {m['deadband']:.1f} PWM, motor lag "
                              f"{m['tau_motor'] * 1000:.0f} ms, command delay {m['delay_s'] * 1000:.0f} ms,\nsteering "
                              f"{m['k_curv_per_deg']:.4f} 1/m per servo deg, coast {m['coast_decel']:.1f} m/s2, brake "
                              f"{m['brake_decel']:.1f} m/s2 (measured 28 Sep)")
        except Exception as e:
            self.twin.setText(f"(car model unavailable: {e})")

    def show_dr(self):
        from sim.twin_intent_data import randomise
        lo, hi = {}, {}
        rng = np.random.default_rng(0)
        for _ in range(400):                       # the ranges at this level, by sampling
            d = randomise(rng, self.level.value())
            for k, v in d.items():
                lo[k], hi[k] = min(lo.get(k, v), v), max(hi.get(k, v), v)
        keys = list(DR_LABELS)
        self.dr_table.setRowCount(len(keys))
        self.dr_table.setColumnCount(3)
        self.dr_table.setHorizontalHeaderLabels(["parameter", "range", "unit"])
        for i, k in enumerate(keys):
            name, unit = DR_LABELS[k]
            fmt = "{:.3f}" if abs(hi[k]) < 1 else "{:.2f}"
            for j, txt in enumerate((name, f"{fmt.format(lo[k])} - {fmt.format(hi[k])}", unit)):
                self.dr_table.setItem(i, j, QtWidgets.QTableWidgetItem(txt))
        self.dr_table.resizeColumnsToContents()

    def check_device(self):
        def work():
            try:
                import torch
                txt = f"GPU: {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU only (no CUDA)"
            except Exception:
                txt = "PyTorch not installed"
            self.device_found.emit(txt)                # queued to the GUI thread
        import threading
        threading.Thread(target=work, daemon=True).start()

    # --- jobs
    def generate(self):
        self.job.start(["-m", "sim.twin_intent_data", str(self.n_train.value()), str(self.n_test.value()),
                        str(self.level.value())], "twin data generation")

    def train(self, quick):
        for c in self.curves.values():
            c.clear()
        self.curves, self.hist = {}, {}
        self.p_loss.clear()
        self.p_ap.clear()
        self.tail.reset()
        self.job.start(["-m", "sim.train_intent_torch"] + (["--quick"] if quick else []), "training")

    # --- live
    def poll(self):
        for row in self.tail.new():
            name = row.get("model")
            if row.get("val_loss") is None:
                continue
            h = self.hist.setdefault(name, {"epoch": [], "val_loss": [], "val_ap": []})
            for k in h:
                h[k].append(row[k])
            if name not in self.curves:
                col = COLS[len(self.curves) % len(COLS)]
                self.curves[name] = (self.p_loss.plot(pen=pg.mkPen(col, width=2), name=name),
                                     self.p_ap.plot(pen=pg.mkPen(col, width=2), name=name))
            cl, ca = self.curves[name]
            cl.setData(h["epoch"], h["val_loss"])
            ca.setData(h["epoch"], h["val_ap"])
        ds = read_json(os.path.join(TRAIN_DIR, "data_status.json"), {})
        st = read_json(os.path.join(TRAIN_DIR, "status.json"), {})
        if self.job.running() and ds and ds.get("done", 0) < ds.get("total", 1):
            self.progress.setRange(0, ds["total"])
            self.progress.setValue(ds["done"])
            self.progress.setFormat(f"generating {ds['set']}: %v / %m drives (randomisation {ds['level']:.1f})")
        elif st:
            if st.get("state") == "training":
                self.progress.setRange(0, st.get("epochs", 1))
                self.progress.setValue(st.get("epoch", 0))
                self.progress.setFormat(f"training {st.get('model')}: epoch %v / %m, val AP {st.get('val_ap', 0):.3f}")
            elif st.get("state") == "done":
                self.progress.setRange(0, 1)
                self.progress.setValue(1)
                self.progress.setFormat(f"done - chosen {st.get('chosen')}")
            d = st.get("data")
            if d:
                self.data_info.setText(f"data: {d.get('train_runs')} training drives ({d.get('train_ticks')} ticks), "
                                       f"{d.get('val_runs')} validation, {d.get('test_runs')} held-out test drives; "
                                       f"positive (contact within 2 s) {100 * d.get('positive_rate_train', 0):.1f} % "
                                       f"of ticks. Real car: sim-to-real check waits for labelled real drives (B15).")
        rp = os.path.join(ROOT, "models", "intent_v3_report.json")
        if os.path.exists(rp) and os.path.getmtime(rp) != self.report_mtime:
            self.report_mtime = os.path.getmtime(rp)
            self.show_report(read_json(rp, {}))

    def show_report(self, rep):
        if not rep:
            return
        # reliability
        self.p_rel.clear()
        self.p_rel.plot([0, 1], [0, 1], pen=pg.mkPen(theme.C["hair2"], width=1, style=QtCore.Qt.DashLine))
        for who, col, lab in (("old_v2", theme.C["dim"], "v2"), ("new", theme.C["accent"], "v3 + physics floor")):
            r = np.array(rep.get("reliability", {}).get(who) or [[0, 0, 0]])
            self.p_rel.plot(r[:, 0], r[:, 1], pen=pg.mkPen(col, width=2), symbol="o", symbolSize=6,
                            symbolBrush=col, name=lab)
        # slices
        sl = rep.get("slices", {})
        names = ["all"] + [k for k in sl if k != "all" and " / " not in k and (sl[k]["new"].get("positives") or 0) >= 20]
        self.p_slices.clear()
        y = np.arange(len(names))
        old = [sl[n]["old_v2"].get("ap") or 0 for n in names]
        new = [sl[n]["new_floor"].get("ap") or 0 for n in names]
        self.p_slices.addItem(pg.BarGraphItem(x0=0, y=y - 0.2, height=0.38, width=old, brush=theme.C["dim"]))
        self.p_slices.addItem(pg.BarGraphItem(x0=0, y=y + 0.2, height=0.38, width=new, brush=theme.C["accent"]))
        self.p_slices.getAxis("left").setTicks([[(i, n) for i, n in enumerate(names)]])
        self.p_slices.invertY(True)
        a = sl.get("all", {})
        rl = rep.get("run_level", {})
        cands = rep.get("candidates", {})
        txt = [f"Chosen model: {rep.get('chosen')}  |  held-out test, all situations: AP {fmt(a.get('old_v2', {}).get('ap'))} "
               f"(v2) -> {fmt(a.get('new_floor', {}).get('ap'))} (v3), AUC {fmt(a.get('old_v2', {}).get('auc'))} -> "
               f"{fmt(a.get('new_floor', {}).get('auc'))}, recall at 0.5 {fmt(a.get('old_v2', {}).get('recall'))} -> "
               f"{fmt(a.get('new_floor', {}).get('recall'))}, false alarms {fmt(a.get('old_v2', {}).get('false_alarm'))}"
               f" -> {fmt(a.get('new_floor', {}).get('false_alarm'))}, calibration error "
               f"{fmt(a.get('old_v2', {}).get('ece'))} -> {fmt(a.get('new_floor', {}).get('ece'))}"]
        if rl:
            o, n = rl.get("old_v2", {}), rl.get("new_floor", {})
            txt.append(f"Drives that end in contact: warned >= 1 s before {fmt(o.get('warned_1s_before'))} -> "
                       f"{fmt(n.get('warned_1s_before'))}, missed {fmt(o.get('missed'))} -> {fmt(n.get('missed'))}; "
                       f"safe drives with a false alarm {fmt(o.get('safe_runs_with_false_alarm'))} -> "
                       f"{fmt(n.get('safe_runs_with_false_alarm'))}")
        txt.append("Candidates (validation AP, parameters): " + ", ".join(
            f"{k} {fmt(c.get('val_ap'))} ({c.get('params')})" for k, c in cands.items()) +
            f"  |  numpy on the car: {rep.get('numpy_ms_per_tick_laptop', 0):.2f} ms per tick on the laptop "
            f"(~{3.5 * rep.get('numpy_ms_per_tick_laptop', 0):.1f} ms on the Pi)")
        self.summary.setText("\n".join(txt))
        self.examples = read_json(os.path.join(TRAIN_DIR, "examples.json"), []) or []
        self.ex_sel.blockSignals(True)
        self.ex_sel.clear()
        for e in self.examples:
            self.ex_sel.addItem(f"{e['family']} / {e['style']} - {'ends in contact' if e['crashed'] else 'no contact'}")
        self.ex_sel.blockSignals(False)
        self.draw_example()

    def draw_example(self):
        i = self.ex_sel.currentIndex()
        if not (0 <= i < len(self.examples)):
            return
        e = self.examples[i]
        self.p_ex.clear()
        t = np.array(e["t"])
        y = np.array(e["y"], float)
        self.p_ex.addItem(pg.FillBetweenItem(pg.PlotDataItem(t, y), pg.PlotDataItem(t, 0 * y),
                                             brush=pg.mkBrush(248, 113, 113, 50)))
        self.p_ex.plot(t, np.minimum(np.array(e["free"]), 2.5) / 2.5, pen=pg.mkPen(theme.C["accent"], width=1),
                       name="free way (/2.5 m)")
        self.p_ex.plot(t, np.abs(np.array(e["v"])), pen=pg.mkPen(theme.C["s3"], width=1), name="speed (m/s)")
        self.p_ex.plot(t, e["p_old"], pen=pg.mkPen(theme.C["dim"], width=2), name="P v2")
        self.p_ex.plot(t, e["p_new"], pen=pg.mkPen(theme.C["accent"], width=3), name="P v3")
        self.p_ex.setYRange(0, 1.05)


def fmt(v):
    return "-" if v is None else f"{v:.3f}"
