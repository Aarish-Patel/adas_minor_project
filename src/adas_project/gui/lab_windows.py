"""The two lab windows with a 3D showcase on top (user, 28 Sep: "showcase the usage and how it was trained - not just
graphs, at least 2-3 3D cars moving"), and the lab controls / charts (gui/lab_tabs.py) underneath.

  MonteCarloWindow  one room of the Monte Carlo, driven at the same time by every system - driver only (red), brake
                    only (amber), ADAS (blue), ADAS + intent (teal) - replayed from the runs, with trails
  TrainingWindow    three randomised digital-twin cars generating labelled training data side by side, coloured by
                    the trained model's live crash risk, labelled with what was randomised and the ground truth
"""
import math
import os

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from gui.lab_tabs import TRAIN_DIR, VARIANT_COL, VARIANT_LABEL, MonteCarloTab, TrainingTab, read_json


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


def _rgb(hexcol):
    return tuple(int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5))


class Showcase(QtWidgets.QWidget):
    """3D stage + play controls shared by both labs."""

    def __init__(self, title):
        super().__init__()
        from gui.ev import Scene
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 0)
        row = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel(title)
        t.setStyleSheet("font-size: 15px; font-weight: 600;")
        t.setWordWrap(True)
        row.addWidget(t, 1)
        self.play = QtWidgets.QPushButton("Pause")
        self.play.clicked.connect(self._toggle)
        row.addWidget(self.play)
        row.addWidget(QtWidgets.QLabel("speed"))
        self.rate = QtWidgets.QComboBox()
        self.rate.addItems(["x1", "x2", "x4"])
        row.addWidget(self.rate)
        self.restart = QtWidgets.QPushButton("Restart")
        row.addWidget(self.restart)
        lay.addLayout(row)
        self.scene = Scene(grid=40)
        lay.addWidget(self.scene, 1)
        self.running = True
        self.t = 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.start(40)

    def _toggle(self):
        self.running = not self.running
        self.play.setText("Pause" if self.running else "Play")

    def step_dt(self):
        return 0.04 * {"x1": 1, "x2": 2, "x4": 4}[self.rate.currentText()] if self.running else 0.0


class MonteCarloWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        from gui.ev import EV_STYLE
        self.setWindowTitle("Monte Carlo lab - the car's own code on the digital twin")
        self.setStyleSheet(EV_STYLE)
        self.resize(1500, 950)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.show3d = Showcase("Same room, same driver, same attention lapses - driven by each system at once")
        split.addWidget(self.show3d)
        self.tab = MonteCarloTab()
        split.addWidget(self.tab)
        split.setSizes([560, 390])
        self.setCentralWidget(split)
        self.cars, self.trails, self.runs, self.key = {}, {}, [], None
        self.tab.scen.currentIndexChanged.connect(self.load)
        self.show3d.restart.clicked.connect(lambda: setattr(self.show3d, "t", 0.0))
        self.show3d.timer.timeout.connect(self.tick)
        legend = QtWidgets.QLabel("   ".join(f"<span style='color:{VARIANT_COL[v]}'>&#9632; {VARIANT_LABEL[v]}</span>"
                                             for v in ("off", "brake-only", "adas", "adas+intent")) +
                                  "   - yellow ring: goal")
        self.statusBar().addWidget(legend)
        self.load()

    def load(self, *_):
        from gui.ev import CarModel, walls_mesh
        from sim.relay_mc import scenario
        k = self.tab.scen.currentData()
        if k is None:
            return
        self.key = tuple(k)
        sc = self.show3d.scene
        world, goal, _ = scenario(int(k[0]))
        sc.mesh("walls", walls_mesh(world.segments(), 0.25), (0.55, 0.6, 0.72, 0.9), opts="opaque")
        a = np.linspace(0, 2 * math.pi, 40)
        V = np.vstack([[goal[0], goal[1], 0.004],
                       np.column_stack([goal[0] + 0.15 * np.cos(a), goal[1] + 0.15 * np.sin(a), np.full(40, 0.004)])])
        sc.mesh("goal", (V, np.array([[0, 1 + i, 1 + (i + 1) % 40] for i in range(40)])), (0.98, 0.8, 0.08, 0.7))
        self.runs = []
        for r in self.tab.runs:
            if (r["seed"], r["style"]) != self.key or not r["trace"]:
                continue
            tr = np.array(r["trace"], float)
            self.runs.append((r["variant"], tr, _headings(tr)))
            if r["variant"] not in self.cars:
                rgb = _rgb(VARIANT_COL[r["variant"]])
                self.cars[r["variant"]] = CarModel(sc, colour=(*rgb, 1.0), ring=(*rgb, 0.8))
                self.trails[r["variant"]] = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(*rgb, 0.9), width=3)
                sc.addItem(self.trails[r["variant"]])
        for v, car in self.cars.items():
            on = any(x[0] == v for x in self.runs)
            car.set_visible(on)
            self.trails[v].setVisible(on)
        sc.setCameraPosition(pos=QtGui.QVector3D(2.5, 0, 0), distance=8.5, elevation=48, azimuth=200)
        self.show3d.t = 0.0

    def tick(self):
        if not self.runs:
            return
        self.show3d.t += self.show3d.step_dt()
        i_now = int(self.show3d.t / 0.05)
        if i_now > max(len(tr) for _, tr, _ in self.runs) + 40:      # loop the replay
            self.show3d.t, i_now = 0.0, 0
        for v, tr, th in self.runs:
            i = min(i_now, len(tr) - 1)
            self.cars[v].place(tr[i, 0], tr[i, 1], th[i])
            seg = tr[:i + 1]
            self.trails[v].setData(pos=np.column_stack([seg, np.full(len(seg), 0.01)]) if len(seg) >= 2
                                   else np.zeros((2, 3)))


class TrainingWindow(QtWidgets.QMainWindow):
    """Randomised digital-twin cars generating labelled training data (models/intent_training/showcase.json, made by
    `python -m sim.twin_intent_data showcase`), three at a time, coloured by the v3 model's live crash risk."""

    GAP = 7.0

    def __init__(self):
        super().__init__()
        from gui.ev import EV_STYLE
        self.setWindowTitle("ML training lab - driver-intent model trained on randomised digital twins")
        self.setStyleSheet(EV_STYLE)
        self.resize(1500, 950)
        split = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.show3d = Showcase("Three randomised twin cars generating labelled data - car colour = the trained model's "
                               "crash risk (teal low, red high); red band on the floor = ground truth 'contact within 2 s'")
        split.addWidget(self.show3d)
        self.tab = TrainingTab()
        split.addWidget(self.tab)
        split.setSizes([560, 390])
        self.setCentralWidget(split)
        self.drives = read_json(os.path.join(TRAIN_DIR, "showcase.json"), []) or []
        self.batch = 0
        self.slots = []
        self.show3d.restart.clicked.connect(lambda: self.load(0))
        self.show3d.timer.timeout.connect(self.tick)
        if not self.drives:
            self.statusBar().showMessage("no showcase yet - run: python -m sim.twin_intent_data showcase")
        self.load(0)

    def load(self, batch):
        from gui.ev import CarModel, ribbon, walls_mesh
        sc = self.show3d.scene
        if not self.drives:
            return
        self.batch = batch
        picks = [self.drives[(3 * batch + k) % len(self.drives)] for k in range(3)]
        if not self.slots:
            for _ in range(3):
                car = CarModel(sc)
                label = gl.GLTextItem(pos=(0, 0, 0), text="", color=(230, 238, 252, 255))
                sc.addItem(label)
                self.slots.append({"car": car, "label": label})
        for k, (slot, d) in enumerate(zip(self.slots, picks)):
            off = np.array([0.0, (k - 1) * self.GAP])
            seg = np.asarray(d["segments"], float).reshape(-1, 4) + np.tile(off, 2)
            sc.mesh(f"walls{k}", walls_mesh(seg, 0.25), (0.55, 0.6, 0.72, 0.9), opts="opaque")
            pose = np.asarray(d["pose"], float)
            pose[:, :2] += off
            band = pose[np.asarray(d["y"]) == 1, :2]
            sc.mesh(f"label{k}", ribbon(band, 0.26, 0.003) if len(band) >= 2 else None, (0.97, 0.3, 0.3, 0.35))
            slot.update(d=d, pose=pose)
        sc.setCameraPosition(pos=QtGui.QVector3D(1.5, 0, 0), distance=17.0, elevation=52, azimuth=205)
        self.show3d.t = 0.0

    def tick(self):
        if not self.slots or not self.drives:
            return
        self.show3d.t += self.show3d.step_dt()
        i_now = int(self.show3d.t / 0.05)
        done = True
        for slot in self.slots:
            d, pose = slot["d"], slot["pose"]
            i = min(i_now, len(pose) - 1)
            done &= i_now >= len(pose) + 25
            risk = d["risk"][i] if d.get("risk") else 0.0
            slot["car"].items[0].setColor((0.13 + 0.84 * risk, 0.83 - 0.5 * risk, 0.93 - 0.6 * risk, 1.0))
            slot["car"].place(pose[i, 0], pose[i, 1], pose[i, 2])
            dr = d["dr"]
            lab = "contact within 2 s" if d["y"][i] else "safe"
            slot["label"].setData(pos=(pose[i, 0], pose[i, 1], 0.45),
                                  text=f"{d['family']} / {d['style']}   risk {100 * risk:.0f} %   label: {lab}   |   "
                                       f"top speed x{dr['v_max_x']:.2f}, brake {dr['brake_decel']:.1f} m/s2, LiDAR "
                                       f"noise {1000 * dr['noise']:.0f} mm, jitter {dr['jitter_deg']:.1f} deg")
        if done:
            self.load(self.batch + 1)
