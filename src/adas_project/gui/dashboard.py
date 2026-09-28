"""RC-ADAS dashboard - a native Qt application (PySide6 + pyqtgraph/OpenGL), the way car HMIs and robot tools (RViz)
are built, instead of the browser pages: the relay pushes a live frame 20 times a second over UDP and the window
redraws at ~30 Hz locally, with no HTTP polling.

    python gui/dashboard.py                    the laptop simulator (tools/sim_car.py) on this machine
    python gui/dashboard.py --host 192.168.1.6 the car

Modes (tabs): Drive 3D - Map & click-to-go - Assists - Diagnostics - Reports, plus an event log. The instrument
cluster on the left is always visible: speed, gear, throttle (driver vs sent), steering, time to contact, "what might
happen", the learned driver-intent model and the speed estimate. Driving itself stays on rc_controller.py.
"""
import argparse
import collections
import json
import math
import os
import socket
import struct
import sys
import threading
import time
import urllib.request

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
CTRL_PORT, HTTP_PORT = 4210, 8090

COL = {"bg": "#0b1020", "panel": "#121a2e", "edge": "#243150", "text": "#e5ecff", "dim": "#8b9ac0",
       "ok": "#34d399", "warn": "#facc15", "bad": "#f87171", "cyan": "#22d3ee", "violet": "#a78bfa", "blue": "#60a5fa"}
STYLE = f"""
QWidget {{ background: {COL['bg']}; color: {COL['text']}; font-family: 'Segoe UI', Arial; font-size: 13px; }}
QLabel {{ background: transparent; }}
QFrame#card {{ background: {COL['panel']}; border: 1px solid {COL['edge']}; border-radius: 10px; }}
QLabel#title {{ color: {COL['dim']}; font-size: 11px; letter-spacing: 1px; }}
QLabel#big {{ font-size: 54px; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid {COL['edge']}; border-radius: 8px; }}
QTabBar::tab {{ background: {COL['panel']}; padding: 8px 18px; margin-right: 2px; border-top-left-radius: 6px;
               border-top-right-radius: 6px; color: {COL['dim']}; }}
QTabBar::tab:selected {{ background: {COL['edge']}; color: {COL['text']}; }}
QPushButton {{ background: {COL['panel']}; border: 1px solid {COL['edge']}; border-radius: 7px; padding: 7px 12px; }}
QPushButton:checked {{ background: #065f46; border-color: {COL['ok']}; }}
QPushButton:hover {{ border-color: {COL['blue']}; }}
QProgressBar {{ background: {COL['panel']}; border: 1px solid {COL['edge']}; border-radius: 5px; height: 12px;
               text-align: center; color: {COL['text']}; }}
QListWidget {{ background: {COL['panel']}; border: 1px solid {COL['edge']}; border-radius: 8px; }}
"""


# ------------------------------------------------------------------ car geometry (the measured body)
def load_geometry():
    try:
        from adas.config import load_tuning
        from pi.relay_assists import car_params
        p = car_params(load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json")).mount)
        return {"front": p.front_x, "rear": p.rear_x, "width": p.width, "lidar_x": p.lidar_x, "wheelbase": p.wheelbase}
    except Exception:
        return {"front": 0.28, "rear": -0.05, "width": 0.20, "lidar_x": 0.12, "wheelbase": 0.20}


GEO = load_geometry()


def polar_xy(polar, lidar_x=GEO["lidar_x"]):
    """The relay's polar form (angle deg clockwise from ahead, distance from the LiDAR) -> vehicle frame (x ahead of
    the rear axle, y left)."""
    if polar is None or len(polar) == 0:
        return np.empty((0, 2))
    arr = np.asarray(polar, float).reshape(-1, 2)
    a = np.radians(arr[:, 0])
    return np.column_stack([lidar_x + arr[:, 1] * np.cos(a), -arr[:, 1] * np.sin(a)])


# ------------------------------------------------------------------ the link to the relay
class Link:
    """Subscribes to the relay's live stream (re-subscribing every 2 s) and sends commands to its control port."""

    def __init__(self, host):
        self.host = host
        self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.rx.bind(("0.0.0.0", 0))
        self.rx.settimeout(0.5)
        self.port = self.rx.getsockname()[1]
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tx.settimeout(0.2)
        self.latest, self.latest_t = None, 0.0
        self.arrivals = collections.deque(maxlen=60)
        self.lock = threading.Lock()
        threading.Thread(target=self._subscribe_loop, daemon=True).start()
        threading.Thread(target=self._receive_loop, daemon=True).start()

    def send(self, cmd):
        try:
            self.tx.sendto(cmd.encode(), (self.host, CTRL_PORT))
            self.tx.recvfrom(64)                       # the relay answers "OK" (ignore if it does not)
        except OSError:
            pass

    def http_post(self, path):
        try:
            urllib.request.urlopen(urllib.request.Request(f"http://{self.host}:{HTTP_PORT}{path}", method="POST"),
                                   timeout=2)
        except Exception:
            pass

    def _subscribe_loop(self):
        while True:
            try:
                self.tx.sendto(f"GUI_SUBSCRIBE {self.port}".encode(), (self.host, CTRL_PORT))
            except OSError:
                pass
            time.sleep(2.0)

    def _receive_loop(self):
        while True:
            try:
                data, _ = self.rx.recvfrom(1 << 17)
            except (socket.timeout, OSError):
                continue
            if data[:3] != b"RC1" or len(data) < 7:
                continue
            n = struct.unpack("<I", data[3:7])[0]
            try:
                state = json.loads(data[7:7 + n])
            except ValueError:
                continue
            raw = np.frombuffer(data[7 + n:], dtype="<i2").reshape(-1, 2)
            state["_pts"] = np.column_stack([raw[:, 0] / 10.0, raw[:, 1] / 1000.0]) if len(raw) else np.empty((0, 2))
            now = time.time()
            with self.lock:
                self.latest, self.latest_t = state, now
                self.arrivals.append(now)

    def snapshot(self):
        with self.lock:
            return self.latest, self.latest_t, list(self.arrivals)


# ------------------------------------------------------------------ small widgets
def card(title):
    f = QtWidgets.QFrame()
    f.setObjectName("card")
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(12, 10, 12, 10)
    t = QtWidgets.QLabel(title.upper())
    t.setObjectName("title")
    lay.addWidget(t)
    return f, lay


class Ring(QtWidgets.QWidget):
    """Time-to-contact ring: full and green when nothing is ahead, shrinking and turning red as contact nears."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(150, 150)
        self.ttc, self.text = None, "--"

    def set(self, ttc):
        self.ttc = ttc
        self.text = "clear" if ttc is None else f"{ttc:.1f} s"
        self.update()

    def paintEvent(self, _):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        s = min(self.width(), self.height()) - 16
        r = QtCore.QRectF((self.width() - s) / 2, (self.height() - s) / 2, s, s)
        p.setPen(QtGui.QPen(QtGui.QColor(COL["edge"]), 12))
        p.drawArc(r, 0, 360 * 16)
        frac = 1.0 if self.ttc is None else max(0.0, min(1.0, self.ttc / 3.0))
        color = COL["ok"] if frac > 0.6 else COL["warn"] if frac > 0.3 else COL["bad"]
        p.setPen(QtGui.QPen(QtGui.QColor(color), 12, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawArc(r, 90 * 16, int(-360 * 16 * frac))
        p.setPen(QtGui.QColor(COL["text"]))
        f = p.font()
        f.setPointSize(16)
        f.setBold(True)
        p.setFont(f)
        p.drawText(r, QtCore.Qt.AlignCenter, f"{self.text}\n")
        f.setPointSize(9)
        f.setBold(False)
        p.setFont(f)
        p.setPen(QtGui.QColor(COL["dim"]))
        p.drawText(r.adjusted(0, 34, 0, 0), QtCore.Qt.AlignCenter, "time to contact")


# ------------------------------------------------------------------ the 3D view (vehicle frame, car at the origin)
def box_mesh(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return gl.MeshData(vertexes=v, faces=f)


class View3D(gl.GLViewWidget):
    def __init__(self):
        super().__init__()
        self.setBackgroundColor(COL["bg"])
        grid = gl.GLGridItem()
        grid.setSize(12, 12)
        grid.setSpacing(0.5, 0.5)
        grid.setColor((40, 55, 90, 255))
        self.addItem(grid)
        body = gl.GLMeshItem(meshdata=box_mesh(GEO["rear"], GEO["front"], -GEO["width"] / 2, GEO["width"] / 2,
                                               0.02, 0.10), color=(0.85, 0.9, 1.0, 1.0), smooth=False,
                            drawEdges=True, edgeColor=(0.13, 0.83, 0.93, 1.0))
        self.addItem(body)
        lidar = gl.GLMeshItem(meshdata=box_mesh(GEO["lidar_x"] - 0.04, GEO["lidar_x"] + 0.04, -0.04, 0.04, 0.10, 0.15),
                              color=(0.2, 0.25, 0.35, 1.0), shader="shaded")
        self.addItem(lidar)
        self.points = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=3, color=(0.38, 0.65, 0.98, 1.0), pxMode=True)
        self.addItem(self.points)
        self.walls = gl.GLLinePlotItem(pos=np.zeros((2, 3)), mode="lines", color=(0.45, 0.5, 0.6, 1.0), width=2)
        self.addItem(self.walls)
        self.pred = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.2, 0.83, 0.6, 1), width=5, antialias=True)
        self.left = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.2, 0.83, 0.6, 0.6), width=2)
        self.right = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.2, 0.83, 0.6, 0.6), width=2)
        self.man = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.13, 0.83, 0.93, 1), width=4)
        self.line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.6, 0.64, 0.72, 0.8), width=1)
        self.hit = gl.GLLinePlotItem(pos=np.zeros((2, 3)), mode="lines", color=(0.94, 0.27, 0.27, 1), width=5)
        self.goal = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=18, color=(0.65, 0.55, 0.98, 0.0))
        for it in (self.pred, self.left, self.right, self.man, self.line, self.hit, self.goal):
            self.addItem(it)
        self.camera("chase")

    def camera(self, which):
        if which == "chase":
            self.setCameraPosition(pos=QtGui.QVector3D(0.6, 0, 0), distance=2.6, elevation=24, azimuth=180)
        elif which == "top":
            self.setCameraPosition(pos=QtGui.QVector3D(0.8, 0, 0), distance=6.0, elevation=89, azimuth=180)
        else:
            self.setCameraPosition(pos=QtGui.QVector3D(0.4, 0, 0), distance=3.5, elevation=35, azimuth=135)

    @staticmethod
    def _xyz(xy, z=0.03):
        if len(xy) < 2:
            return np.zeros((2, 3))
        return np.column_stack([xy, np.full(len(xy), z)])

    def show(self, st):
        pts = polar_xy(st.get("_pts"))
        self.points.setData(pos=self._xyz(pts, 0.08) if len(pts) else np.zeros((1, 3)))
        segs = []
        for poly in ((st.get("sim") or {}).get("walls") or []):
            xy = polar_xy(poly)
            for i in range(len(xy) - 1):
                segs += [xy[i], xy[i + 1]]
        self.walls.setData(pos=self._xyz(np.array(segs), 0.02) if segs else np.zeros((2, 3)))
        plan = st.get("plan") or {}
        state = plan.get("state", "clear")
        rgb = {"collision": (0.97, 0.44, 0.44), "limited": (0.98, 0.8, 0.08)}.get(state, (0.2, 0.83, 0.6))
        for item, key, alpha in ((self.pred, "pred", 1.0), (self.left, "left", 0.55), (self.right, "right", 0.55)):
            item.setData(pos=self._xyz(polar_xy(plan.get(key))), color=(*rgb, alpha if plan.get(key) else 0.0))
        self.man.setData(pos=self._xyz(polar_xy(plan.get("maneuver")), 0.05))
        self.line.setData(pos=self._xyz(polar_xy(plan.get("line")), 0.01))
        if plan.get("hit"):
            hx, hy = polar_xy([plan["hit"]])[0]
            k = 0.07
            self.hit.setData(pos=np.array([[hx - k, hy - k, 0.05], [hx + k, hy + k, 0.05],
                                           [hx - k, hy + k, 0.05], [hx + k, hy - k, 0.05]]))
        else:
            self.hit.setData(pos=np.zeros((2, 3)))
        goal = (st.get("nav") or {}).get("goal")
        if goal:
            g = polar_xy([goal])[0]
            self.goal.setData(pos=np.array([[g[0], g[1], 0.1]]), color=(0.65, 0.55, 0.98, 1.0))
        else:
            self.goal.setData(color=(0.65, 0.55, 0.98, 0.0))


# ------------------------------------------------------------------ the 2D map with click-to-go
class MapView(pg.PlotWidget):
    """Top view in the vehicle frame (the car points up). Click to send a click-to-go goal."""

    def __init__(self, link, on_goal):
        super().__init__(background=COL["bg"])
        self.link, self.on_goal = link, on_goal
        self.setAspectLocked(True)
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setXRange(-2.5, 2.5)
        self.setYRange(-1.5, 3.5)
        self.getPlotItem().hideButtons()
        self.scatter = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(COL["blue"]), pen=None)
        self.walls = pg.PlotCurveItem(pen=pg.mkPen("#475569", width=2), connect="pairs")
        self.pred = pg.PlotCurveItem(pen=pg.mkPen(COL["ok"], width=3))
        self.man = pg.PlotCurveItem(pen=pg.mkPen(COL["cyan"], width=3))
        self.line = pg.PlotCurveItem(pen=pg.mkPen(COL["dim"], width=1, style=QtCore.Qt.DashLine))
        self.goal = pg.ScatterPlotItem(size=20, symbol="star", brush=pg.mkBrush(COL["violet"]), pen=None)
        hw = GEO["width"] / 2
        body = np.array([[-hw, GEO["rear"]], [hw, GEO["rear"]], [hw, GEO["front"]], [-hw, GEO["front"]],
                         [-hw, GEO["rear"]]])
        self.body = pg.PlotCurveItem(body[:, 0], body[:, 1], pen=pg.mkPen(COL["text"], width=2),
                                     fillLevel=None, brush=pg.mkBrush(229, 236, 255, 90))
        for it in (self.walls, self.scatter, self.line, self.pred, self.man, self.goal, self.body):
            self.addItem(it)
        self.scene().sigMouseClicked.connect(self._click)

    @staticmethod
    def _sc(xy):
        return (-xy[:, 1], xy[:, 0]) if len(xy) else ([], [])

    def _click(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            return
        p = self.getPlotItem().vb.mapSceneToView(ev.scenePos())
        x, y = float(p.y()), float(-p.x())                  # screen up = ahead, screen right = the car's right
        self.link.send(f"GOTO {x:.3f} {y:.3f}")
        self.on_goal(x, y)

    def show(self, st):
        pts = polar_xy(st.get("_pts"))
        self.scatter.setData(*self._sc(pts))
        segs = []
        for poly in ((st.get("sim") or {}).get("walls") or []):
            xy = polar_xy(poly)
            for i in range(len(xy) - 1):
                segs += [xy[i], xy[i + 1]]
        self.walls.setData(*self._sc(np.array(segs)) if segs else ([], []))
        plan = st.get("plan") or {}
        color = {"collision": COL["bad"], "limited": COL["warn"]}.get(plan.get("state"), COL["ok"])
        self.pred.setPen(pg.mkPen(color, width=3))
        self.pred.setData(*self._sc(polar_xy(plan.get("pred"))))
        self.man.setData(*self._sc(polar_xy(plan.get("maneuver"))))
        self.line.setData(*self._sc(polar_xy(plan.get("line"))))
        goal = (st.get("nav") or {}).get("goal")
        self.goal.setData(*self._sc(polar_xy([goal])) if goal else ([], []))


# ------------------------------------------------------------------ the main window
ASSISTS = [("evasive", "Evasive steer", "Hybrid A* round an obstacle and back to your line; MPPI fallback"),
           ("nudge", "Steering correction", "a small steering nudge instead of braking when that makes your path safe"),
           ("centring", "Corridor centring", "keeps the car centred between walls; yields to a deliberate turn"),
           ("limiter", "Speed / steer limiter", "caps lateral acceleration in turns"),
           ("narrow", "Narrow gap", "slows for tight gaps, warns when the car will not fit"),
           ("proximity", "Side / rear alerts", "warns about things beside or behind the car")]


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, link):
        super().__init__()
        self.link = link
        self.setWindowTitle(f"RC-ADAS dashboard - {link.host}")
        self.resize(1500, 900)
        self.hist = collections.deque(maxlen=900)          # ~30 s of (t, values) for the diagnostics plots
        self.last_msgs = {}
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        outer = QtWidgets.QVBoxLayout(root)
        outer.addLayout(self._top_bar())
        body = QtWidgets.QHBoxLayout()
        outer.addLayout(body, 1)
        body.addWidget(self._cluster(), 0)
        self.tabs = QtWidgets.QTabWidget()
        body.addWidget(self.tabs, 1)
        self.view3d = View3D()
        self.tabs.addTab(self._drive_tab(), "Drive 3D")
        self.tabs.addTab(self._map_tab(), "Map && click-to-go")
        self.tabs.addTab(self._assist_tab(), "Assists")
        self.tabs.addTab(self._diag_tab(), "Diagnostics")
        self.tabs.addTab(self._report_tab(), "Reports")
        self.tabs.addTab(self._events_tab(), "Events")
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)

    # --- layout pieces
    def _top_bar(self):
        bar = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel("RC-ADAS")
        t.setStyleSheet("font-size: 20px; font-weight: 700; letter-spacing: 2px;")
        bar.addWidget(t)
        self.mode = QtWidgets.QLabel("CONNECTING")
        self.mode.setStyleSheet(self._chip(COL["dim"]))
        bar.addWidget(self.mode)
        self.alert = QtWidgets.QLabel("")
        self.alert.setStyleSheet(f"color: {COL['warn']}; font-weight: 600; padding-left: 12px;")
        bar.addWidget(self.alert, 1)
        self.sim = QtWidgets.QLabel("")
        self.sim.setStyleSheet(f"color: {COL['dim']};")
        bar.addWidget(self.sim)
        self.reset_btn = QtWidgets.QPushButton("Reset simulated car")
        self.reset_btn.clicked.connect(lambda: self.link.http_post("/api/sim/reset"))
        self.reset_btn.hide()
        bar.addWidget(self.reset_btn)
        self.conn = QtWidgets.QLabel("")
        self.conn.setStyleSheet(f"color: {COL['dim']}; padding-left: 12px;")
        bar.addWidget(self.conn)
        return bar

    @staticmethod
    def _chip(color):
        return (f"background: {color}33; border: 1px solid {color}; color: {color}; border-radius: 12px; "
                f"padding: 4px 14px; font-weight: 700; margin-left: 14px;")

    def _cluster(self):
        w = QtWidgets.QWidget()
        w.setFixedWidth(340)
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 8, 0)
        c, l = card("speed")
        row = QtWidgets.QHBoxLayout()
        self.speed = QtWidgets.QLabel("0.00")
        self.speed.setObjectName("big")
        row.addWidget(self.speed)
        right = QtWidgets.QVBoxLayout()
        self.unit = QtWidgets.QLabel("m/s")
        self.unit.setStyleSheet(f"color: {COL['dim']};")
        self.gear = QtWidgets.QLabel("P")
        self.gear.setStyleSheet("font-size: 30px; font-weight: 700;")
        right.addWidget(self.unit)
        right.addWidget(self.gear)
        row.addLayout(right)
        l.addLayout(row)
        self.kmh = QtWidgets.QLabel("")
        self.kmh.setStyleSheet(f"color: {COL['dim']};")
        l.addWidget(self.kmh)
        for name, attr in (("throttle - driver", "thr_in"), ("throttle - sent to the motor", "thr_out")):
            lab = QtWidgets.QLabel(name)
            lab.setStyleSheet(f"color: {COL['dim']}; font-size: 11px;")
            bar = QtWidgets.QProgressBar()
            bar.setRange(-255, 255)
            bar.setFormat("%v")
            setattr(self, attr, bar)
            l.addWidget(lab)
            l.addWidget(bar)
        lab = QtWidgets.QLabel("steering (servo degrees from straight)")
        lab.setStyleSheet(f"color: {COL['dim']}; font-size: 11px;")
        self.steer = QtWidgets.QProgressBar()
        self.steer.setRange(-55, 55)
        self.steer.setFormat("%v")
        l.addWidget(lab)
        l.addWidget(self.steer)
        lay.addWidget(c)
        c, l = card("what might happen")
        self.ring = Ring()
        l.addWidget(self.ring, 0, QtCore.Qt.AlignHCenter)
        self.might = QtWidgets.QLabel("")
        self.might.setWordWrap(True)
        l.addWidget(self.might)
        lay.addWidget(c)
        c, l = card("driver intent (learned model)")
        self.risk = QtWidgets.QProgressBar()
        self.risk.setRange(0, 100)
        self.risk.setFormat("P(crash) %p%")
        l.addWidget(self.risk)
        self.intent = QtWidgets.QLabel("")
        self.intent.setWordWrap(True)
        l.addWidget(self.intent)
        lay.addWidget(c)
        c, l = card("speed estimate")
        self.speed_est = QtWidgets.QLabel("")
        self.speed_est.setWordWrap(True)
        l.addWidget(self.speed_est)
        lay.addWidget(c)
        lay.addStretch(1)
        return w

    def _drive_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        for name in ("chase", "top", "orbit"):
            b = QtWidgets.QPushButton(f"{name.title()} camera")
            b.clicked.connect(lambda _=False, n=name: self.view3d.camera(n))
            row.addWidget(b)
        row.addStretch(1)
        legend = QtWidgets.QLabel("green/yellow/red: where the car goes at your stick and throttle  |  X: first "
                                  "contact  |  cyan: planned manoeuvre  |  grey dashes: your line  |  star: goal")
        legend.setStyleSheet(f"color: {COL['dim']}; font-size: 11px;")
        row.addWidget(legend)
        lay.addLayout(row)
        lay.addWidget(self.view3d, 1)
        return w

    def _map_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        row = QtWidgets.QHBoxLayout()
        self.nav_status = QtWidgets.QLabel("Click on the map to drive there autonomously - hold the throttle on the "
                                           "controller as the dead-man switch; steer or brake to take over.")
        self.nav_status.setWordWrap(True)
        row.addWidget(self.nav_status, 1)
        cancel = QtWidgets.QPushButton("Cancel autonomy")
        cancel.clicked.connect(lambda: self.link.send("GOTO CANCEL"))
        row.addWidget(cancel)
        lay.addLayout(row)
        self.map = MapView(self.link, lambda x, y: self._event(f"click-to-go goal sent: {x:.2f} m ahead, "
                                                               f"{y:+.2f} m left"))
        lay.addWidget(self.map, 1)
        return w

    def _assist_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        info = QtWidgets.QLabel("The path-predicted brake is always on. Each assist below can be switched on or off; "
                                "the brake still has the last word on the throttle.")
        info.setWordWrap(True)
        lay.addWidget(info)
        grid = QtWidgets.QGridLayout()
        self.assist_btns = {}
        for i, (name, label, desc) in enumerate(ASSISTS):
            b = QtWidgets.QPushButton(label)
            b.setCheckable(True)
            b.setMinimumWidth(200)
            b.clicked.connect(lambda checked, n=name: self.link.send(f"ASSIST {n} {'ON' if checked else 'OFF'}"))
            self.assist_btns[name] = b
            d = QtWidgets.QLabel(desc)
            d.setStyleSheet(f"color: {COL['dim']};")
            grid.addWidget(b, i, 0)
            grid.addWidget(d, i, 1)
        lay.addLayout(grid)
        row = QtWidgets.QHBoxLayout()
        on = QtWidgets.QPushButton("All assists on")
        on.clicked.connect(lambda: self.link.send("ASSIST all ON"))
        off = QtWidgets.QPushButton("All assists off")
        off.clicked.connect(lambda: self.link.send("ASSIST all OFF"))
        self.follow_btn = QtWidgets.QPushButton("Follow the leader")
        self.follow_btn.setCheckable(True)
        self.follow_btn.clicked.connect(lambda c: self.link.send("FOLLOW_ON" if c else "FOLLOW_OFF"))
        self.override_btn = QtWidgets.QPushButton("ADAS override (no safety - careful)")
        self.override_btn.setCheckable(True)
        self.override_btn.clicked.connect(lambda c: self.link.send("ADAS_OVERRIDE_ON" if c else "ADAS_OVERRIDE_OFF"))
        for b in (on, off, self.follow_btn, self.override_btn):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)
        lay.addStretch(1)
        return w

    def _diag_tab(self):
        w = pg.GraphicsLayoutWidget()
        w.setBackground(COL["bg"])
        self.p_speed = w.addPlot(title="speed (m/s): EKF with LiDAR range flow vs throttle model")
        self.c_v = self.p_speed.plot(pen=pg.mkPen(COL["cyan"], width=2), name="EKF")
        self.c_vm = self.p_speed.plot(pen=pg.mkPen(COL["dim"], width=1), name="model")
        w.nextRow()
        self.p_thr = w.addPlot(title="throttle PWM: driver (grey) vs sent to the motor (green)")
        self.c_in = self.p_thr.plot(pen=pg.mkPen(COL["dim"], width=1))
        self.c_out = self.p_thr.plot(pen=pg.mkPen(COL["ok"], width=2))
        w.nextRow()
        self.p_free = w.addPlot(title="free distance on the path (m, brake gate) and P(crash) from the intent model")
        self.c_free = self.p_free.plot(pen=pg.mkPen(COL["warn"], width=2))
        self.c_risk = self.p_free.plot(pen=pg.mkPen(COL["bad"], width=1))
        w.nextRow()
        self.p_rate = w.addPlot(title="stream: frames per second")
        self.c_rate = self.p_rate.plot(pen=pg.mkPen(COL["blue"], width=1))
        for p in (self.p_speed, self.p_thr, self.p_free, self.p_rate):
            p.showGrid(x=True, y=True, alpha=0.15)
        return w

    def _report_tab(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        self.report_sel = QtWidgets.QComboBox()
        self.reports = [(f, os.path.join(ROOT, "reports", f)) for f in
                        ("monte_carlo_relay.png", "twin_lidar.png", "twin_path.png", "odometry.png", "autonav.png")
                        if os.path.exists(os.path.join(ROOT, "reports", f))]
        self.report_sel.addItems([f for f, _ in self.reports] or ["(no reports yet - run the evaluations)"])
        lay.addWidget(self.report_sel)
        self.report_img = QtWidgets.QLabel()
        self.report_img.setAlignment(QtCore.Qt.AlignCenter)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidget(self.report_img)
        scroll.setWidgetResizable(True)
        lay.addWidget(scroll, 1)
        self.report_sel.currentIndexChanged.connect(self._show_report)
        self._show_report(0)
        return w

    def _show_report(self, i):
        if 0 <= i < len(self.reports):
            pm = QtGui.QPixmap(self.reports[i][1])
            self.report_img.setPixmap(pm.scaledToWidth(1100, QtCore.Qt.SmoothTransformation))

    def _events_tab(self):
        self.events = QtWidgets.QListWidget()
        return self.events

    def _event(self, text):
        self.events.insertItem(0, f"{time.strftime('%H:%M:%S')}  {text}")
        if self.events.count() > 500:
            self.events.takeItem(500)

    # --- live update
    def refresh(self):
        st, t_rx, arrivals = self.link.snapshot()
        now = time.time()
        if st is None or now - t_rx > 1.5:
            self.mode.setText("NO DATA - is the relay / simulator running?")
            self.mode.setStyleSheet(self._chip(COL["bad"]))
            self.conn.setText(f"waiting for {self.link.host}:{CTRL_PORT}")
            return
        fps = (len(arrivals) - 1) / max(arrivals[-1] - arrivals[0], 1e-3) if len(arrivals) > 1 else 0.0
        self.conn.setText(f"{self.link.host}  {fps:4.1f} frames/s")
        drive = st.get("drive") or {}
        assist = st.get("assist") or {}
        gate = st.get("gate") or {}
        plan = st.get("plan") or {}
        nav = st.get("nav") or {}
        intent = st.get("intent") or {}
        sim = st.get("sim")
        # mode chip
        info = assist.get("info") or {}
        act = str(gate.get("action") or "")
        if st.get("mode") == "override":
            mode, color = "OVERRIDE - NO ADAS", COL["bad"]
        elif nav.get("state") not in (None, "idle"):
            mode, color = "AUTONOMY: CLICK-TO-GO", COL["violet"]
        elif assist.get("evading") or assist.get("phase") in ("EXECUTE", "WAIT"):
            mode, color = "AUTONOMOUS MANOEUVRE", COL["cyan"]
        elif "brak" in act or act.startswith("hold") or act == "stopped":
            mode, color = "EMERGENCY BRAKE", COL["bad"]
        elif "nudge" in info:
            mode, color = "STEERING CORRECTION", COL["cyan"]
        elif act == "limited":
            mode, color = "SPEED LIMITED", COL["warn"]
        else:
            mode, color = "GUARDIAN", COL["ok"]
        self.mode.setText(mode)
        self.mode.setStyleSheet(self._chip(color))
        msgs = [str(v) for v in info.values()] + ([str(gate["action"])] if gate.get("action") else [])
        self.alert.setText("  |  ".join(msgs))
        for k, v in list(info.items()) + [("gate", gate.get("action"))]:
            if v and self.last_msgs.get(k) != v:
                self._event(f"{k}: {v}")
            self.last_msgs[k] = v
        # simulator
        if sim:
            self.sim.setText(f"SIMULATOR  {sim.get('world', '')}   crashes {sim.get('crashes', 0)}")
            self.reset_btn.show()
        else:
            self.sim.setText("CAR")
            self.reset_btn.hide()
        # cluster
        v = float(drive.get("v", 0.0))
        self.speed.setText(f"{abs(v):.2f}")
        self.kmh.setText(f"{abs(v) * 3.6:.1f} km/h   yaw {math.degrees(float(drive.get('w', 0.0))):+.0f} deg/s")
        pin, pout = float(drive.get("pwm_in", 0) or 0), float(drive.get("pwm_out", 0) or 0)
        self.gear.setText("D" if pin > 5 else "R" if pin < -5 else ("N" if abs(v) > 0.03 else "P"))
        self.thr_in.setValue(int(pin))
        self.thr_out.setValue(int(pout))
        self.steer.setValue(int(round(float(drive.get("servo", 87)) - float(drive.get("centre", 87)))))
        self.ring.set(plan.get("ttc") if plan.get("hit") else None)
        lines = [f"path: {plan.get('state', 'clear')}"]
        if plan.get("hit_m") is not None:
            lines.append(f"first contact in {plan['hit_m']:.2f} m ({plan.get('ttc', 0):.1f} s)")
        if gate.get("free_m") is not None:
            lines.append(f"free way {gate['free_m']:.2f} m, allowed {gate.get('v_allowed', 0):.2f} m/s")
        for k in ("evasive", "nudge", "autonomy", "centring", "narrow", "limiter", "proximity"):
            if info.get(k):
                lines.append(f"{k}: {info[k]}")
        if gate.get("intent"):
            lines.append(str(gate["intent"]))
        self.might.setText("\n".join(lines))
        p = intent.get("p_crash")
        self.risk.setValue(int(round(100 * p)) if p is not None else 0)
        self.intent.setText(("trusted - the driver is handling it" if intent.get("trusted") else "not trusted") +
                            (", stick active" if intent.get("attentive") else ", stick idle") +
                            f"\nusual reaction distance {intent.get('reaction_m', 0):.2f} m")
        self.speed_est.setText(f"EKF (car model + LiDAR range flow) {v:+.2f} m/s\nthrottle model "
                               f"{float(drive.get('v_model', 0.0)):+.2f} m/s")
        # assists
        enabled = assist.get("enabled") or {}
        for name, b in self.assist_btns.items():
            b.blockSignals(True)
            b.setChecked(bool(enabled.get(name)))
            b.blockSignals(False)
        for b, on in ((self.follow_btn, st.get("follow_enabled")), (self.override_btn, st.get("mode") == "override")):
            b.blockSignals(True)
            b.setChecked(bool(on))
            b.blockSignals(False)
        state = nav.get("state", "idle")
        if state != "idle" or nav.get("msg"):
            self.nav_status.setText(f"autonomy: {state} - {nav.get('msg', '')}"
                                    + (f" (plan {nav['plan_ms']:.0f} ms)" if nav.get("plan_ms") else ""))
        # views (only the visible one is redrawn)
        tab = self.tabs.currentIndex()
        if tab == 0:
            self.view3d.show(st)
        elif tab == 1:
            self.map.show(st)
        # history + diagnostics
        self.hist.append((now, v, float(drive.get("v_model", 0.0)), pin, pout,
                          gate.get("free_m") if gate.get("free_m") is not None else np.nan,
                          p if p is not None else np.nan, fps))
        if tab == 3 and len(self.hist) > 2:
            h = np.array(self.hist, float)
            t = h[:, 0] - h[-1, 0]
            self.c_v.setData(t, h[:, 1])
            self.c_vm.setData(t, h[:, 2])
            self.c_in.setData(t, h[:, 3])
            self.c_out.setData(t, h[:, 4])
            self.c_free.setData(t, np.nan_to_num(h[:, 5], nan=0.0))
            self.c_risk.setData(t, np.nan_to_num(h[:, 6], nan=0.0))
            self.c_rate.setData(t, h[:, 7])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="the relay: 127.0.0.1 for the laptop simulator, "
                                                         "192.168.1.6 for the car")
    ap.add_argument("--tab", type=int, default=0, help="the mode to open: 0 Drive 3D, 1 Map, 2 Assists, "
                                                        "3 Diagnostics, 4 Reports, 5 Events")
    ap.add_argument("--snapshot", help="save the window as this PNG after --after seconds, then quit (tests, slides)")
    ap.add_argument("--after", type=float, default=6.0)
    a = ap.parse_args()
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = Dashboard(Link(a.host))
    win.tabs.setCurrentIndex(a.tab)
    win.show()
    if a.snapshot:
        def snap():
            win.grab().save(a.snapshot)
            app.quit()
        QtCore.QTimer.singleShot(int(a.after * 1000), snap)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
