"""The two lab windows: a 3D showcase on top, the process and the numbers underneath (user, 28-29 Sep: "showcase the
usage and how it was trained - not just graphs, at least 2-3 3D cars moving", professional EV look).

  TrainingWindow    a pipeline stepper (identify twin -> randomise -> generate -> train -> validate -> deploy check), three
                    randomised digital-twin cars generating labelled data side by side (car colour = the trained model's
                    live crash risk, red band on the floor = ground truth), an overlay card per car with what was
                    randomised, KPI tiles (v2 -> v3) and the charts / controls
  MonteCarloWindow  one room driven at the same time by every system - driver only, brake only, ADAS, ADAS + intent -
                    replayed with trails, KPI tiles per system, the runs table and statistics underneath
"""
import math
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from gui import theme
from gui.ev import CarModel, Glass, Scene, ribbon, walls_mesh
from gui.lab_tabs import (MC_DIR, TRAIN_DIR, VARIANT_COL, VARIANT_LABEL, MonteCarloTab, TrainingTab, read_json)
from gui.theme import C, rgb
from gui.widgets import Kpi, RiskBar, Stepper, risk_rgb

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAR_SHOW_SCALE = 3.0                     # the twin cars are drawn larger than life so they read on screen


def _headings(xy):
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return np.zeros(len(xy))
    d = np.diff(xy, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    moving = np.hypot(d[:, 0], d[:, 1]) > 1e-4
    for i in range(1, len(th)):                     # hold the heading while standing (atan2 of zero is noise)
        if not moving[i]:
            th[i] = th[i - 1]
    return np.append(th, th[-1])


def _hex_rgb(h, a=1.0):
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)) + (a,)


class Stage(QtWidgets.QWidget):
    """3D stage + a header with play controls; overlay cards are children of the scene."""

    def __init__(self, title, subtitle):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        head = QtWidgets.QFrame()
        head.setObjectName("top")
        hl = QtWidgets.QHBoxLayout(head)
        hl.setContentsMargins(20, 10, 20, 10)
        col = QtWidgets.QVBoxLayout()
        col.setSpacing(2)
        t = QtWidgets.QLabel(title.upper())
        t.setFont(theme.semibold(15, spacing=3.0))
        s = QtWidgets.QLabel(subtitle)
        s.setStyleSheet(f"color: {C['dim']};")
        col.addWidget(t)
        col.addWidget(s)
        hl.addLayout(col, 1)
        self.play = QtWidgets.QPushButton("PAUSE")
        self.play.setObjectName("app")
        self.play.clicked.connect(self._toggle)
        hl.addWidget(self.play)
        self.rate = QtWidgets.QComboBox()
        self.rate.addItems(["1x", "2x", "4x"])
        hl.addWidget(self.rate)
        self.restart = QtWidgets.QPushButton("RESTART")
        self.restart.setObjectName("app")
        hl.addWidget(self.restart)
        lay.addWidget(head)
        self.scene = Scene(grid=60)
        lay.addWidget(self.scene, 1)
        self.running, self.t = True, 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.start(33)

    def _toggle(self):
        self.running = not self.running
        self.play.setText("PAUSE" if self.running else "PLAY")

    def step_dt(self):
        return 0.033 * {"1x": 1, "2x": 2, "4x": 4}[self.rate.currentText()] if self.running else 0.0


class TwinCard(Glass):
    """Overlay card for one showcase car."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setMinimumSize(300, 124)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(16, 12, 16, 12)
        lay.setSpacing(6)
        row = QtWidgets.QHBoxLayout()
        self.title = QtWidgets.QLabel("")
        self.title.setFont(theme.semibold(13, spacing=0.6))
        self.chip = QtWidgets.QLabel("")
        row.addWidget(self.title, 1)
        row.addWidget(self.chip)
        lay.addLayout(row)
        rrow = QtWidgets.QHBoxLayout()
        self.risk_lbl = QtWidgets.QLabel("RISK 0 %")
        self.risk_lbl.setFont(theme.font(11, spacing=1.0))
        self.risk_lbl.setStyleSheet(f"color: {C['dim']};")
        self.bar = RiskBar()
        rrow.addWidget(self.risk_lbl)
        rrow.addWidget(self.bar, 1)
        lay.addLayout(rrow)
        self.params = QtWidgets.QLabel("")
        self.params.setWordWrap(True)
        self.params.setFont(theme.font(11))
        self.params.setStyleSheet(f"color: {C['text2']};")
        lay.addWidget(self.params)

    def set(self, name, contact, risk, params):
        self.title.setText(name)
        colour = "bad" if contact else "ok"
        self.chip.setText("CONTACT \u2264 2 S" if contact else "SAFE")
        self.chip.setStyleSheet(theme.chip_css(colour))
        self.risk_lbl.setText(f"RISK {100 * risk:.0f} %")
        self.bar.set(risk)
        self.params.setText(params)


# ====================================================================== ML training
class TrainingWindow(QtWidgets.QMainWindow):
    """Randomised digital-twin cars generating labelled training data (models/intent_training/showcase.json, made by
    `python -m sim.twin_intent_data showcase`), three at a time, coloured by the v3 model's live crash risk."""

    GAP = 8.0
    STEPS = ["Identify twin", "Randomise", "Generate data", "Train on GPU", "Validate", "Deploy check"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ML training lab - driver-intent model trained on randomised digital twins")
        self.setStyleSheet(theme.STYLE)
        self.resize(1560, 1000)
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QtWidgets.QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.stepper = Stepper(self.STEPS)
        sf = QtWidgets.QFrame()
        sf.setObjectName("bar")
        sl = QtWidgets.QVBoxLayout(sf)
        sl.setContentsMargins(20, 8, 20, 4)
        sl.addWidget(self.stepper)
        lay.addWidget(sf)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.stage = Stage("Digital-twin training", "Three randomised twin cars generating labelled data. Car colour = "
                           "the trained model's live crash risk; red band on the floor = ground truth (contact within 2 s).")
        top = QtWidgets.QWidget()
        tl = QtWidgets.QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        tl.addWidget(self.stage, 1)
        cardrow = QtWidgets.QFrame()
        cardrow.setObjectName("bar")
        cr = QtWidgets.QHBoxLayout(cardrow)
        cr.setContentsMargins(14, 10, 14, 10)
        cr.setSpacing(12)
        self.cards = [TwinCard(None) for _ in range(3)]
        for c in self.cards:
            cr.addWidget(c, 1)
        tl.addWidget(cardrow)
        split.addWidget(top)
        low = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(low)
        ll.setContentsMargins(14, 10, 14, 6)
        krow = QtWidgets.QHBoxLayout()
        self.kpis = {k: Kpi(t, u, d) for k, (t, u, d) in {
            "ap": ("Average precision", "", 2), "auc": ("AUC", "", 2), "recall": ("Recall at 0.5", "%", 0),
            "fa": ("False alarms", "%", 1), "ece": ("Calibration error", "", 3), "lead": ("Warned ≥ 1 s early", "%", 0)}.items()}
        for k in self.kpis.values():
            krow.addWidget(k)
        ll.addLayout(krow)
        self.tab = TrainingTab()
        ll.addWidget(self.tab, 1)
        split.addWidget(low)
        split.setSizes([640, 360])
        lay.addWidget(split, 1)
        self.drives = read_json(os.path.join(TRAIN_DIR, "showcase.json"), []) or []
        self.batch, self.slots = 0, []
        self.stage.restart.clicked.connect(lambda: self.load(0))
        self.stage.timer.timeout.connect(self.tick)
        self.report_mtime = None
        if not self.drives:
            self.statusBar().showMessage("no showcase yet - run: python -m sim.twin_intent_data showcase")
        self.load(0)
        self._kpis()
        QtCore.QTimer(self, interval=1000, timeout=self._poll).start()

    def _poll(self):
        st = read_json(os.path.join(TRAIN_DIR, "status.json"), {}) or {}
        ds = read_json(os.path.join(TRAIN_DIR, "data_status.json"), {}) or {}
        detail = {0: "identified", 1: "15 parameters"}
        active = 6
        if self.tab.job.running():
            if ds and ds.get("done", 0) < ds.get("total", 1) and "train" not in self.tab.job.proc.arguments()[-1]:
                active = 2
                detail[2] = f"{ds['done']}/{ds['total']} drives"
            else:
                active = 3
                detail[3] = f"epoch {st.get('epoch', 0)} · {st.get('model', '')}"
        elif st.get("state") == "done":
            active = 6
            detail[3] = st.get("chosen", "")
            detail[4] = "held-out test"
            detail[5] = "numpy = torch"
        self.stepper.set(active, detail)
        rp = os.path.join(ROOT, "models", "intent_v3_report.json")
        if os.path.exists(rp) and os.path.getmtime(rp) != self.report_mtime:
            self._kpis()

    def _kpis(self):
        rp = os.path.join(ROOT, "models", "intent_v3_report.json")
        rep = read_json(rp, {}) or {}
        if not rep:
            return
        self.report_mtime = os.path.getmtime(rp)
        a = rep["slices"]["all"]
        old, new = a["old_v2"], a["new_floor"]
        rl = rep.get("run_level", {})

        def d(o, n, pct=False, better_high=True, dec=2):
            f = 100 if pct else 1
            delta = (n - o) * f
            good = (delta > 0) == better_high
            unit = " %" if pct else ""
            sign = "+" if delta > 0 else "−"
            return f"was {o * f:.{dec}f}{unit}  ·  {sign}{abs(delta):.{dec}f}{' pts' if pct else ''}", good
        self.kpis["ap"].set(new["ap"], *d(old["ap"], new["ap"]))
        self.kpis["auc"].set(new["auc"], *d(old["auc"], new["auc"]))
        self.kpis["recall"].set(100 * new["recall"], *d(old["recall"], new["recall"], True, True, 0))
        self.kpis["fa"].set(100 * new["false_alarm"], *d(old["false_alarm"], new["false_alarm"], True, False, 1))
        self.kpis["ece"].set(new["ece"], *d(old["ece"], new["ece"], False, False, 3))
        if rl:
            o, n = rl["old_v2"]["warned_1s_before"], rl["new_floor"]["warned_1s_before"]
            self.kpis["lead"].set(100 * n, *d(o, n, True, True, 0))

    def load(self, batch):
        sc = self.stage.scene
        if not self.drives:
            return
        self.batch = batch
        picks = [self.drives[(3 * batch + k) % len(self.drives)] for k in range(3)]
        if not self.slots:
            for _ in range(3):
                self.slots.append({"car": CarModel(sc, scale=CAR_SHOW_SCALE)})
        for k, (slot, d) in enumerate(zip(self.slots, picks)):
            off = np.array([0.0, (k - 1) * self.GAP])
            seg = np.asarray(d["segments"], float).reshape(-1, 4) + np.tile(off, 2)
            sc.mesh(f"walls{k}", walls_mesh(seg, 0.30), C["wall"], opts="opaque")
            pose = np.asarray(d["pose"], float)
            pose[:, :2] += off
            band = pose[np.asarray(d["y"]) == 1, :2]
            sc.mesh(f"label{k}", ribbon(band, 0.36, 0.003) if len(band) >= 2 else None, rgb("bad", 0.40))
            slot.update(d=d, pose=pose)
        # frame all three twin rooms: the camera looks at the middle of everything the twins will do
        allp = np.vstack([np.vstack([np.asarray(d["pose"], float)[:, :2] + np.array([0.0, (k - 1) * self.GAP])
                                     for k, d in enumerate(picks)])])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        ctr, span = (lo + hi) / 2, float(max(hi - lo)) + 4.0
        sc.setCameraPosition(pos=QtGui.QVector3D(float(ctr[0]), float(ctr[1]), 0), distance=span * 1.05, elevation=64,
                             azimuth=180)
        self.stage.t = 0.0

    def tick(self):
        if not self.slots or not self.drives:
            return
        self.stage.t += self.stage.step_dt()
        i_now = int(self.stage.t / 0.05)
        done = True
        for k, slot in enumerate(self.slots):
            d, pose = slot["d"], slot["pose"]
            i = min(i_now, len(pose) - 1)
            done &= i_now >= len(pose) + 25
            risk = d["risk"][i] if d.get("risk") else 0.0
            slot["car"].items[0].setColor(risk_rgb(risk))
            slot["car"].place(pose[i, 0], pose[i, 1], pose[i, 2])
            slot["car"].pulse(self.stage.t, "bad" if d["y"][i] else "accent")
            dr = d["dr"]
            self.cards[k].set(f"TWIN {k + 1} · {d['family']} / {d['style']}".upper(), bool(d["y"][i]), risk,
                              f"randomised: top speed ×{dr['v_max_x']:.2f} · brake {dr['brake_decel']:.1f} m/s² · "
                              f"LiDAR noise {1000 * dr['noise']:.0f} mm · jitter {dr['jitter_deg']:.1f}° · "
                              f"delay {1000 * dr['delay_s']:.0f} ms")
        for kp in self.kpis.values():
            kp.tick(0.033)
        if done:
            self.load(self.batch + 1)


# ====================================================================== Monte Carlo
class SystemCard(QtWidgets.QFrame):
    """Live status of one system (driver only / brake only / ADAS / ADAS + intent) in the replayed room: what the car is
    doing now, how far from the goal, and the run's outcome so far - a research-console card, not a legend entry."""

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.setObjectName("syscard")
        col = VARIANT_COL[variant]
        self.setStyleSheet(f"QFrame#syscard {{ background: {C['bg1']}; border: 1px solid {C['hair']}; border-radius: 12px; }} "
                           f"QLabel {{ background: transparent; }}")
        self.setMinimumHeight(146)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        dot = QtWidgets.QLabel("●")
        dot.setStyleSheet(f"color: {col}; font-size: 16px;")
        head.addWidget(dot)
        name = QtWidgets.QLabel(VARIANT_LABEL[variant].upper())
        name.setFont(theme.semibold(12, spacing=1.4))
        head.addWidget(name, 1)
        self.chip = QtWidgets.QLabel("")
        head.addWidget(self.chip)
        lay.addLayout(head)
        row = QtWidgets.QHBoxLayout()
        self.stats = {}
        for key, cap in (("v", "KM/H"), ("t", "TIME S"), ("d", "TO GOAL M")):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(0)
            val = QtWidgets.QLabel("-")
            val.setFont(theme.light(22))
            val.setMinimumHeight(30)
            c = QtWidgets.QLabel(cap)
            c.setFont(theme.font(10, spacing=1.2))
            c.setStyleSheet(f"color: {C['dim']};")
            box.addWidget(val)
            box.addWidget(c)
            row.addLayout(box, 1)
            self.stats[key] = val
        lay.addLayout(row)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(6)
        self.bar.setStyleSheet(f"QProgressBar {{ background: {C['hair']}; border: none; border-radius: 3px; }} "
                               f"QProgressBar::chunk {{ background: {col}; border-radius: 3px; }}")
        lay.addWidget(self.bar)
        self.foot = QtWidgets.QLabel("")
        self.foot.setFont(theme.font(11))
        self.foot.setStyleSheet(f"color: {C['dim']};")
        self.foot.setWordWrap(True)
        self.foot.setMinimumHeight(30)
        lay.addWidget(self.foot)

    def show_run(self, run, tr, i, dt=0.05):
        """run: the run record; tr: its (x, y) trace; i: replay index."""
        n = len(tr)
        i = min(i, n - 1)
        j = max(0, i - 2)
        v = float(np.hypot(*(tr[i] - tr[j]))) / max(dt * (i - j), 1e-6) if i > j else 0.0
        goal = np.asarray(run["goal"], float)
        d = float(np.hypot(*(tr[i] - goal)))
        finished = i >= n - 1
        if run["crashed"] and finished:
            chip, colr = "CRASHED", "bad"
        elif finished and run["reached"] is not None:
            chip, colr = "GOAL REACHED", "ok"
        elif finished:
            chip, colr = "STOPPED SHORT", "warn"
        elif v < 0.03:
            chip, colr = "STANDING", "info"
        else:
            chip, colr = "DRIVING", "accent"
        self.chip.setText(chip)
        self.chip.setStyleSheet(theme.chip_css(colr))
        self.stats["v"].setText(f"{v * 14 * 3.6:.0f}")
        self.stats["t"].setText(f"{i * dt:.1f}")
        self.stats["d"].setText(f"{d:.2f}")
        d0 = float(np.hypot(*(tr[0] - goal))) or 1.0
        self.bar.setValue(int(1000 * max(0.0, min(1.0, 1 - d / d0))))
        self.foot.setText(f"closest {100 * run['min_clear']:.0f} cm · {run['interventions']} interventions "
                          f"({run['false_positives']} needless) · {run['lapses']} lapses")


class MonteCarloWindow(QtWidgets.QMainWindow):
    """Research console: the 3D room with every system driving the same scenario at once (left), a live status card per
    system and the replay timeline (right), results and controls underneath."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Monte Carlo lab - the car's own code on the digital twin")
        self.setStyleSheet(theme.STYLE)
        self.resize(1640, 1000)
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        lay = QtWidgets.QVBoxLayout(root)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        upper = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.stage = Stage("Monte Carlo", "One room, one driver, the same attention lapses - every system drives it at the "
                           "same time on the digital twin. Trails are the paths; the counterfactual decides which "
                           "interventions were needed.")
        upper.addWidget(self.stage)
        side = QtWidgets.QWidget()
        side.setMinimumWidth(360)
        side.setMaximumWidth(460)
        sl = QtWidgets.QVBoxLayout(side)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(10)
        cap = QtWidgets.QLabel("SYSTEMS UNDER TEST")
        cap.setFont(theme.semibold(11, spacing=1.6))
        cap.setStyleSheet(f"color: {C['dim']};")
        sl.addWidget(cap)
        self.cards = {}
        for v in ("off", "brake-only", "adas", "adas+intent"):
            c = SystemCard(v)
            c.hide()
            self.cards[v] = c
            sl.addWidget(c)
        sl.addStretch(1)
        self.scrub = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scrub.setRange(0, 1000)
        self.scrub_lbl = QtWidgets.QLabel("REPLAY TIMELINE")
        self.scrub_lbl.setFont(theme.semibold(11, spacing=1.6))
        self.scrub_lbl.setStyleSheet(f"color: {C['dim']};")
        sl.addWidget(self.scrub_lbl)
        sl.addWidget(self.scrub)
        upper.addWidget(side)
        upper.setSizes([1200, 420])
        split.addWidget(upper)
        low = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(low)
        ll.setContentsMargins(14, 10, 14, 6)
        self.krow = QtWidgets.QHBoxLayout()
        self.kpis = {}
        for v in ("off", "brake-only", "adas", "adas+intent"):
            k = Kpi(VARIANT_LABEL[v] + " · crashes", "", 0)
            k.hide()
            self.kpis[v] = k
            self.krow.addWidget(k)
        ll.addLayout(self.krow)
        self.tab = MonteCarloTab()
        ll.addWidget(self.tab, 1)
        split.addWidget(low)
        split.setSizes([720, 330])
        lay.addWidget(split, 1)
        self.legend = Glass(self.stage.scene)
        lg = QtWidgets.QHBoxLayout(self.legend)
        lg.setContentsMargins(16, 8, 16, 8)
        lg.setSpacing(18)
        for v in ("off", "brake-only", "adas", "adas+intent"):
            lab = QtWidgets.QLabel(f"<span style='color:{VARIANT_COL[v]}'>●</span>  {VARIANT_LABEL[v].upper()}")
            lab.setFont(theme.font(11, spacing=1.0))
            lg.addWidget(lab)
        self.legend.move(20, 16)
        self.cars, self.trails, self.runs, self.key = {}, {}, [], None
        self._n_runs = 0
        self.scrubbing = False
        self.scrub.sliderPressed.connect(lambda: setattr(self, "scrubbing", True))
        self.scrub.sliderReleased.connect(lambda: setattr(self, "scrubbing", False))
        self.scrub.sliderMoved.connect(self._seek)
        self.tab.scen.currentIndexChanged.connect(self.load)
        self.stage.restart.clicked.connect(lambda: setattr(self.stage, "t", 0.0))
        self.stage.timer.timeout.connect(self.tick)
        self.load()

    def _seek(self, val):
        n = max((len(tr) for _, _, tr, _ in self.runs), default=1)
        self.stage.t = val / 1000.0 * n * 0.05

    def load(self, *_):
        from sim.relay_mc import scenario
        k = self.tab.scen.currentData()
        if k is None:
            return
        self.key = tuple(k)
        sc = self.stage.scene
        world, goal, _ = scenario(int(k[0]))
        segs = np.asarray(world.segments(), float).reshape(-1, 4)
        sc.mesh("walls", walls_mesh(segs, 0.30), C["wall"], opts="opaque")
        a = np.linspace(0, 2 * math.pi, 40)
        V = np.vstack([[goal[0], goal[1], 0.004],
                       np.column_stack([goal[0] + 0.18 * np.cos(a), goal[1] + 0.18 * np.sin(a), np.full(40, 0.004)])])
        sc.mesh("goal", (V, np.array([[0, 1 + i, 1 + (i + 1) % 40] for i in range(40)])), rgb("accent", 0.55))
        self.runs = []
        for r in self.tab.runs:
            if (r["seed"], r["style"]) != self.key or not r["trace"]:
                continue
            tr = np.array(r["trace"], float)
            self.runs.append((r["variant"], r, tr, _headings(tr)))
            if r["variant"] not in self.cars:
                col = _hex_rgb(VARIANT_COL[r["variant"]])
                self.cars[r["variant"]] = CarModel(sc, colour=col, ring=(*col[:3], 0.8), scale=CAR_SHOW_SCALE * 0.5)
                self.trails[r["variant"]] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(*col[:3], 0.95), width=3,
                                                              antialias=True)
                sc.addItem(self.trails[r["variant"]])
        present = {x[0] for x in self.runs}
        for v, car in self.cars.items():
            car.set_visible(v in present)
            self.trails[v].setVisible(v in present)
        for v, c in self.cards.items():
            c.setVisible(v in present)
        # frame the whole room so every car is on screen (top-down, distance from the room's extent)
        pts = np.vstack([segs[:, :2], segs[:, 2:], np.atleast_2d(goal)])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        ctr, span = (lo + hi) / 2, float(max(hi - lo)) or 6.0
        sc.setCameraPosition(pos=QtGui.QVector3D(float(ctr[0]), float(ctr[1]), 0), distance=span * 1.25, elevation=68,
                             azimuth=180)
        self.stage.t = 0.0

    def _kpis(self):
        st = read_json(os.path.join(MC_DIR, "status.json"), {}) or {}
        summ = st.get("summary") or {}
        for v, k in self.kpis.items():
            if v in summ:
                s = summ[v]
                k.show()
                k.set(s["crashes"], f"{s['needless']} needless interventions · {s['runs']} runs",
                      good=None if v == "off" else s["crashes"] == 0)

    def tick(self):
        self._kpis()
        for k in self.kpis.values():
            k.tick(0.033)
        if len(self.tab.runs) != self._n_runs:               # a Monte Carlo is running: pick up systems as they finish
            self._n_runs = len(self.tab.runs)
            have = {x[0] for x in self.runs}
            now = {r["variant"] for r in self.tab.runs if (r["seed"], r["style"]) == self.key and r["trace"]}
            if now != have:
                t_keep = self.stage.t
                self.load()
                self.stage.t = t_keep
        if not self.runs:
            return
        n_max = max(len(tr) for _, _, tr, _ in self.runs)
        if not self.scrubbing:
            self.stage.t += self.stage.step_dt()
        i_now = int(self.stage.t / 0.05)
        if i_now > n_max + 40:                                       # loop the replay
            self.stage.t, i_now = 0.0, 0
        if not self.scrubbing:
            self.scrub.setValue(int(1000 * min(i_now, n_max) / n_max))
        for v, run, tr, th in self.runs:
            i = min(i_now, len(tr) - 1)
            self.cars[v].place(tr[i, 0], tr[i, 1], th[i])
            seg = tr[:i + 1]
            self.trails[v].setData(pos=np.column_stack([seg, np.full(len(seg), 0.01)]) if len(seg) >= 2
                                   else np.zeros((2, 3)))
            self.cards[v].show_run(run, tr, i)
