"""EV-style main window for the RC-ADAS dashboard (user, 28 Sep: "make it feel like an actual EV GUI", essentials on
the main screen, everything else in pop-up / expandable panels; reference: an automated-valet-parking screen).

    python gui/dashboard.py [--host 192.168.1.6]          (this window; --classic for the old tabbed one)

Main screen: a 3D chase view of the car (detailed model, blue ring), the path it will take as a ribbon on the floor
(green / amber / red by what the brake predicts), the planned manoeuvre / autonomy path in blue, obstacles as low walls
and speed-limit zones as tinted floor areas. Overlays: speed (full-size equivalent km/h at 1:14, and m/s), gear, a
speed-limit sign, steering (with the speed-dependent limit), time to contact, the autonomy banner, alerts and a
mini-map. The bottom bar opens drawers: Map & Zones (world map, zone editor, click-to-go with a heading), Assists,
Car setup (the Pi's control panel: drive, calibrations, thresholds), Diagnostics and Events; the two labs open in
their own windows.
"""
import collections
import json
import math
import os
import threading
import time
import urllib.request

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from gui.controls import FeatureTile, Segmented, section_label
from gui.dashboard import ASSISTS, CTRL_PORT, GEO, polar_xy

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAR_SCALE = 14.0
PANEL_PORT = 8080
ZONES_FILE = os.path.join(ROOT, "gui", "zones.json")

from gui import theme
from gui.theme import C, rgb

# legacy names used through this file, mapped onto the design system (graphite + brass, not navy + cyan)
EV = {"bg": C["bg"], "bg2": C["bg1"], "edge": C["hair"], "text": C["text"], "dim": C["dim"], "blue": C["accent"],
      "cyan": C["accent"], "ok": C["ok"], "warn": C["warn"], "bad": C["bad"], "violet": C["info"]}
EV_STYLE = theme.STYLE


def kmh(v):
    """Full-size equivalent speed (km/h) of the car's m/s at 1:CAR_SCALE."""
    return abs(v) * CAR_SCALE * 3.6


def world_to_vehicle(xy, pose):
    px, py, pth = pose
    c, s = math.cos(pth), math.sin(pth)
    dx, dy = np.asarray(xy, float)[:, 0] - px, np.asarray(xy, float)[:, 1] - py
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy])


def vehicle_to_world(xy, pose):
    px, py, pth = pose
    c, s = math.cos(pth), math.sin(pth)
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.column_stack([px + c * xy[:, 0] - s * xy[:, 1], py + s * xy[:, 0] + c * xy[:, 1]])


def zone_outline(z, n=40):
    k = z.get("kind")
    if k == "rect":
        x0, x1, y0, y1 = z["x0"], z["x1"], z["y0"], z["y1"]
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    if k == "circle":
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return np.column_stack([z["x"] + z["r"] * np.cos(a), z["y"] + z["r"] * np.sin(a)])
    return np.array(z.get("pts", []), float).reshape(-1, 2)


def zone_colour(kph):
    """Low limits red, high green (RGB 0-1)."""
    f = max(0.0, min(1.0, (kph - 5.0) / 45.0))
    return (0.97 - 0.6 * f, 0.35 + 0.5 * f, 0.35 + 0.1 * f)


# ====================================================================== 3D building blocks
def box(x0, x1, y0, y1, z0, z1):
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                  [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return v, f


def merge(parts):
    V, F, n = [], [], 0
    for v, f in parts:
        V.append(v)
        F.append(f + n)
        n += len(v)
    return np.vstack(V), np.vstack(F)


def ribbon(xy, width, z=0.004):
    """A flat strip of the given width along a polyline (for paths on the floor)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return None
    d = np.gradient(xy, axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    L, R = xy + n * width / 2, xy - n * width / 2
    V = np.vstack([np.column_stack([L, np.full(len(L), z)]), np.column_stack([R, np.full(len(R), z)])])
    m = len(xy)
    F = np.array([[i, i + 1, m + i] for i in range(m - 1)] + [[i + 1, m + i + 1, m + i] for i in range(m - 1)])
    return V, F


def walls_mesh(segments, h):
    """Vertical quads over floor segments ((N, 4): x1, y1, x2, y2)."""
    seg = np.asarray(segments, float).reshape(-1, 4)
    if not len(seg):
        return None
    V, F = [], []
    for i, (x1, y1, x2, y2) in enumerate(seg):
        V += [[x1, y1, 0], [x2, y2, 0], [x2, y2, h], [x1, y1, h]]
        b = 4 * i
        F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    return np.array(V, float), np.array(F)


def point_walls(xy, h=0.10, half=0.02, join=0.12):
    """Obstacles as surfaces: consecutive LiDAR points closer than `join` become one wall strip (a continuous
    face, like a mapped surface), isolated points a small tile facing the car."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return None
    seg = []
    d = np.hypot(*(xy[1:] - xy[:-1]).T)
    for i in range(len(xy) - 1):
        if d[i] < join:
            seg.append([*xy[i], *xy[i + 1]])
    lone = np.ones(len(xy), bool)
    for i in range(len(xy) - 1):
        if d[i] < join:
            lone[i] = lone[i + 1] = False
    for x, y in xy[lone]:
        a = math.atan2(y, x) + math.pi / 2
        seg.append([x - math.cos(a) * half, y - math.sin(a) * half, x + math.cos(a) * half, y + math.sin(a) * half])
    return walls_mesh(np.array(seg), h) if seg else None


class CarModel:
    """A small EV body: chassis, cabin, glass, wheels, lights, and the ground ring (brass, pulsing).
    All parts hang off one parent, so the whole car moves with one transform (labs animate several)."""

    def __init__(self, view, colour=(0.90, 0.91, 0.92, 1.0), ring=None, scale=1.0):
        ring = ring or rgb('accent', 0.85)
        self.scale = scale
        r, f, w = GEO["rear"] - 0.03, GEO["front"], GEO["width"] / 2
        wb = GEO["wheelbase"]
        a = np.linspace(0, 2 * math.pi, 60)
        self.parent = gl.GLLinePlotItem(pos=np.column_stack([0.1 + 0.30 * np.cos(a), 0.30 * np.sin(a),
                                                             np.full(60, 0.003)]), color=ring, width=3,
                                        antialias=True)
        view.addItem(self.parent)
        L = f - r
        parts = [(merge([box(r, f, -w, w, 0.035, 0.085)]), colour),
                 (merge([box(r + 0.28 * L, r + 0.78 * L, -w * 0.82, w * 0.82, 0.085, 0.13)]), (0.07, 0.08, 0.10, 1)),
                 (merge([box(-0.03, 0.03, s * w - 0.02, s * w + 0.02, 0.0, 0.06) for s in (-1, 1)] +
                        [box(wb - 0.03, wb + 0.03, s * w - 0.02, s * w + 0.02, 0.0, 0.06) for s in (-1, 1)]),
                  (0.04, 0.045, 0.05, 1)),
                 (merge([box(f - 0.006, f + 0.004, s * w * 0.55 - 0.02, s * w * 0.55 + 0.02, 0.06, 0.075)
                         for s in (-1, 1)]), (1.0, 0.97, 0.88, 1)),
                 (merge([box(r - 0.004, r + 0.006, -w * 0.8, w * 0.8, 0.065, 0.075)]), (0.86, 0.16, 0.18, 1))]
        self.items = []
        for k, ((V, F), col) in enumerate(parts):
            edge = tuple(0.55 * c for c in col[:3]) + (1.0,)
            it = gl.GLMeshItem(vertexes=V, faces=F, color=col, smooth=False, drawEdges=k < 2, edgeColor=edge,
                               glOptions="opaque")
            it.setParentItem(self.parent)
            self.items.append(it)

    def pulse(self, t, colour_name="accent", speed=1.6):
        a = 0.55 + 0.35 * math.sin(t * speed)
        self.parent.setData(color=rgb(colour_name, a))

    def place(self, x, y, th_rad):
        m = QtGui.QMatrix4x4()
        m.translate(x, y, 0)
        m.rotate(math.degrees(th_rad), 0, 0, 1)
        m.scale(self.scale, self.scale, self.scale)
        self.parent.setTransform(m)

    def set_visible(self, on):
        self.parent.setVisible(on)


class Scene(gl.GLViewWidget):
    """Shared 3D stage: floor, camera presets, meshes that can be replaced every frame."""

    def __init__(self, grid=16):
        super().__init__()
        self.setBackgroundColor(C["bg"])
        g = gl.GLGridItem()
        g.setSize(grid, grid)
        g.setSpacing(0.5, 0.5)
        g.setColor((70, 74, 82, 70))
        self.addItem(g)
        self._meshes = {}

    def mesh(self, key, data, colour, opts="translucent"):
        it = self._meshes.get(key)
        if data is None:
            if it is not None:
                it.setVisible(False)
            return
        V, F = data
        if it is None:
            it = gl.GLMeshItem(vertexes=V, faces=F, color=colour, smooth=False, glOptions=opts)
            self.addItem(it)
            self._meshes[key] = it
        else:
            it.setMeshData(vertexes=V, faces=F, color=colour)
            it.setColor(colour)
            it.setVisible(True)

    def chase(self, target=(0.6, 0.0), azimuth=180, distance=2.4, elevation=22):
        self.setCameraPosition(pos=QtGui.QVector3D(target[0], target[1], 0), distance=distance,
                               elevation=elevation, azimuth=azimuth)


# ====================================================================== the drive scene
class DriveScene(Scene):
    """The car at the origin of its own frame (it points along +x), the world moves round it."""

    def __init__(self):
        super().__init__()
        self.car = CarModel(self)
        self.car.place(0, 0, 0)
        self.goal = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(0.85, 0.7, 0.35, 0.0), width=4, antialias=True)
        self.addItem(self.goal)
        self.hit = gl.GLLinePlotItem(pos=np.zeros((2, 3)), mode="lines", color=(0.94, 0.32, 0.31, 0.0), width=5)
        self.addItem(self.hit)
        self.labels = []
        self.cam = "chase"
        self.camera("chase")

    def camera(self, which):
        self.cam = which
        if which == "chase":
            self.chase((0.7, 0.0), 180, 2.2, 20)
        elif which == "top":
            self.setCameraPosition(pos=QtGui.QVector3D(0.8, 0, 0), distance=6.0, elevation=89, azimuth=180)
        else:
            self.setCameraPosition(pos=QtGui.QVector3D(0.4, 0, 0), distance=3.2, elevation=35, azimuth=135)

    def _labels(self, items):
        """items: [(x, y, text, rgb)] - reuses GLTextItems."""
        while len(self.labels) < len(items):
            t = gl.GLTextItem(pos=(0, 0, 0), text="", color=(255, 255, 255, 230))
            self.addItem(t)
            self.labels.append(t)
        for i, t in enumerate(self.labels):
            if i < len(items):
                x, y, txt, rgb = items[i]
                t.setData(pos=(x, y, 0.02), text=txt, color=tuple(int(255 * c) for c in rgb) + (255,))
                t.setVisible(True)
            else:
                t.setVisible(False)

    ARC_SECTORS = [(-150, -110), (-105, -75), (-70, -30), (-25, 25), (30, 70), (75, 105), (110, 150), (155, 205)]

    def _proximity_arcs(self, pts):
        """Parking-sensor style arcs round the car: hidden when clear, grey -> amber -> red as an obstacle gets closer
        (ultrasonic arcs of production HMIs, drawn from the LiDAR)."""
        if len(pts):
            ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0]))
            dist = np.hypot(pts[:, 0] - 0.10, pts[:, 1])
        for i, (a0, a1) in enumerate(self.ARC_SECTORS):
            near = None
            if len(pts):
                a = np.where(ang < 0, ang + 360, ang) if a0 > 150 else ang
                m = (a >= a0) & (a <= a1)
                near = dist[m].min() if m.any() else None
            key = f"arc{i}"
            if near is None or near > 0.9:
                self.mesh(key, None, (0, 0, 0, 0))
                continue
            col = rgb('bad') if near < 0.25 else rgb('warn') if near < 0.5 else (0.62, 0.65, 0.70, 0.8)
            r0 = 0.32
            t = np.radians(np.linspace(a0, a1, 14))
            xy = np.column_stack([0.10 + r0 * np.cos(t), r0 * np.sin(t)])
            self.mesh(key, ribbon(xy, 0.025 + 0.03 * (1 - min(near, 0.9) / 0.9), 0.012), col)

    def show(self, st):
        pts = polar_xy(st.get("_pts"))
        plan = st.get("plan") or {}
        # objects coloured by relevance, as production displays do: grey = ignored, blue = on the planned path,
        # red = the point the car will hit (Tesla FSD visualisation reference, RESEARCH.md section 8)
        pred_xy = polar_xy(plan.get("pred"))
        on_path = np.zeros(len(pts), bool)
        if len(pts) and len(pred_xy) >= 2:
            d = np.hypot(pts[:, None, 0] - pred_xy[None, :, 0], pts[:, None, 1] - pred_xy[None, :, 1]).min(axis=1)
            on_path = d < GEO["width"] / 2 + 0.05
        hit = polar_xy([plan["hit"]])[0] if plan.get("hit") else None
        red = np.zeros(len(pts), bool)
        if hit is not None and len(pts):
            red = np.hypot(pts[:, 0] - hit[0], pts[:, 1] - hit[1]) < 0.15
        idle = ~(on_path | red)
        self.mesh("obstacles", point_walls(pts[idle]), (0.72, 0.75, 0.80, 0.80))
        self.mesh("obst_path", point_walls(pts[on_path & ~red]), (*rgb('accent')[:3], 0.95))
        self.mesh("obst_hit", point_walls(pts[red]), (*rgb('bad')[:3], 0.95))
        self._proximity_arcs(pts)
        segs = []
        for poly in ((st.get("sim") or {}).get("walls") or []):
            xy = polar_xy(poly)
            segs += [[*xy[i], *xy[i + 1]] for i in range(len(xy) - 1)]
        self.mesh("walls", walls_mesh(segs, 0.05), (0.5, 0.52, 0.56, 0.10))     # simulator ground truth, faint
        state = plan.get("state", "clear")
        col = {"collision": rgb('bad')[:3], "limited": rgb('warn')[:3]}.get(state, (0.93, 0.94, 0.95))
        self.mesh("pred", ribbon(polar_xy(plan.get("pred")), GEO["width"] * 0.9, 0.004), (*col, 0.30))
        man = polar_xy(plan.get("maneuver"))
        self.mesh("plan", ribbon(man, 0.12, 0.006) if len(man) >= 2 else None, rgb('accent', 0.9))
        self.mesh("line", ribbon(polar_xy(plan.get("line")), 0.02, 0.002), (0.85, 0.87, 0.9, 0.35))
        if plan.get("hit"):
            hx, hy = polar_xy([plan["hit"]])[0]
            k = 0.07
            self.hit.setData(pos=np.array([[hx - k, hy - k, 0.03], [hx + k, hy + k, 0.03],
                                           [hx - k, hy + k, 0.03], [hx + k, hy - k, 0.03]]), color=rgb('bad'))
        else:
            self.hit.setData(color=(0.94, 0.32, 0.31, 0.0))
        goal = (st.get("nav") or {}).get("goal")
        if goal:
            g = polar_xy([goal])[0]
            a = np.linspace(0, 2 * math.pi, 40)
            ring = np.column_stack([g[0] + 0.12 * np.cos(a), g[1] + 0.12 * np.sin(a), np.full(40, 0.01)])
            self.goal.setData(pos=np.vstack([ring, [[g[0], g[1], 0.01], [g[0], g[1], 0.35]]]),
                              color=rgb('accent'))
        else:
            self.goal.setData(color=(0.85, 0.7, 0.35, 0.0))
        # speed-limit zones, from the world frame into the car's
        world = st.get("world") or {}
        pose = world.get("pose") or [0, 0, 0]
        labels = []
        Vs, Fs, n = [], [], 0
        for z in world.get("zones") or []:
            out = zone_outline(z)
            if len(out) < 3:
                continue
            v = world_to_vehicle(out, pose)
            c = v.mean(axis=0)
            V = np.vstack([[c[0], c[1], 0.002], np.column_stack([v, np.full(len(v), 0.002)])])
            F = np.array([[0, 1 + i, 1 + (i + 1) % len(v)] for i in range(len(v))])
            self.mesh(f"zone{n}", (V, F), (*zone_colour(z["kph"]), 0.22))
            labels.append((c[0], c[1], f"{z['kph']:.0f} km/h", zone_colour(z["kph"])))
            n += 1
        k = n
        while f"zone{k}" in self._meshes:
            self._meshes[f"zone{k}"].setVisible(False)
            k += 1
        self._labels(labels)


# ====================================================================== overlays
class Glass(QtWidgets.QFrame):
    """A frosted panel that fades in and out (opacity effect, 180 ms) instead of popping."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glass")
        self._fx = QtWidgets.QGraphicsOpacityEffect(self)
        self._fx.setOpacity(1.0)
        self.setGraphicsEffect(self._fx)
        self._anim = QtCore.QPropertyAnimation(self._fx, b"opacity", self)
        self._anim.setDuration(180)
        self._anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        self._anim.finished.connect(self._hide_if_off)
        self._shown = True

    def fade(self, on):
        if on == self._shown:
            return
        self._shown = on
        self._anim.stop()
        self._anim.setStartValue(self._fx.opacity())
        self._anim.setEndValue(1.0 if on else 0.0)
        if on:
            self.show()
        self._anim.start()

    def _hide_if_off(self):
        if not self._shown:
            self.hide()


class SpeedCluster(Glass):
    """The instrument cluster: a large DIN-style speed readout that tweens, a thin arc gauge, the gear strip
    (P R N D), the speed-limit sign and the steering-range bar. One custom-painted widget."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setFixedSize(360, 250)
        self.spd = theme.Tween(0.0, 10.0)
        self.steer = theme.Tween(0.0, 14.0)
        self.steer_max = 38.0
        self.gear, self.limit, self.ms = "P", None, 0.0

    def set(self, v, gear, limit, steer_deg, steer_max):
        self.spd.set(kmh(v))
        self.ms = abs(v)
        self.gear, self.limit = gear, limit
        self.steer.set(steer_deg)
        self.steer_max = steer_max or 55.0

    def tick(self, dt):
        moving = self.spd.step(dt) | self.steer.step(dt)
        if moving:
            self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.TextAntialiasing)
        w, h = self.width(), self.height()
        r = QtCore.QRectF(22, 18, 190, 190)                      # arc gauge 0..60 km/h
        p.setPen(QtGui.QPen(theme.qcolor("hair"), 5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
        p.drawArc(r, 225 * 16, -270 * 16)
        frac = max(0.0, min(1.0, self.spd.value / 60.0))
        if frac > 0.004:
            p.setPen(QtGui.QPen(theme.qcolor("accent"), 5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawArc(r, 225 * 16, int(-270 * 16 * frac))
        if self.limit:                                            # the zone limit as a red tick on the arc
            a = math.radians(225 - 270 * min(1.0, self.limit / 60.0))
            c = r.center()
            p.setPen(QtGui.QPen(theme.qcolor("bad"), 3, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
            p.drawLine(QtCore.QPointF(c.x() + 89 * math.cos(a), c.y() - 89 * math.sin(a)),
                       QtCore.QPointF(c.x() + 101 * math.cos(a), c.y() - 101 * math.sin(a)))
        p.setPen(theme.qcolor("text"))
        p.setFont(theme.light(78))
        p.drawText(QtCore.QRectF(22, 50, 190, 100), QtCore.Qt.AlignCenter, f"{self.spd.value:.0f}")
        p.setPen(theme.qcolor("dim"))
        p.setFont(theme.font(13, spacing=2.0))
        p.drawText(QtCore.QRectF(22, 138, 190, 22), QtCore.Qt.AlignCenter, "KM/H")
        p.setFont(theme.font(11))
        p.drawText(QtCore.QRectF(22, 162, 190, 18), QtCore.Qt.AlignCenter,
                   f"{self.ms:.2f} m/s  \u00b7  1:{CAR_SCALE:.0f} scale")
        p.setFont(theme.semibold(17))                             # gear strip
        x0 = 236
        for i, gname in enumerate("PRND"):
            on = gname == self.gear
            p.setPen(theme.qcolor("accent") if on else theme.qcolor("faint"))
            p.drawText(QtCore.QRectF(x0, 20 + i * 27, 30, 26), QtCore.Qt.AlignCenter, gname)
            if on:
                p.setPen(QtGui.QPen(theme.qcolor("accent"), 2))
                p.drawLine(x0 + 36, 24 + i * 27, x0 + 36, 40 + i * 27)
        if self.limit:                                            # the speed-limit sign
            c = QtCore.QPointF(w - 50, 150)
            p.setBrush(QtGui.QColor("#F4F5F6"))
            p.setPen(QtGui.QPen(QtGui.QColor("#D93A3F"), 6))
            p.drawEllipse(c, 27, 27)
            p.setPen(QtGui.QColor("#15171A"))
            p.setFont(theme.semibold(19))
            p.drawText(QtCore.QRectF(c.x() - 27, c.y() - 27, 54, 54), QtCore.Qt.AlignCenter, f"{self.limit:.0f}")
        y = h - 26                                                # steering range bar
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(theme.qcolor("hair"))
        p.drawRoundedRect(QtCore.QRectF(24, y - 2, w - 48, 4), 2, 2)
        cx, half = w / 2, (w - 48) / 2
        band = min(1.0, self.steer_max / 55.0) * half
        p.setBrush(theme.qcolor("accent", 90))
        p.drawRoundedRect(QtCore.QRectF(cx - band, y - 2, 2 * band, 4), 2, 2)
        x = cx + max(-1.0, min(1.0, self.steer.value / 55.0)) * half
        p.setBrush(theme.qcolor("text"))
        p.drawEllipse(QtCore.QPointF(x, y), 5.5, 5.5)
        p.setPen(theme.qcolor("dim"))
        p.setFont(theme.font(10, spacing=1.2))
        p.drawText(QtCore.QRectF(24, y - 24, w - 48, 14), QtCore.Qt.AlignLeft, "STEERING")
        p.drawText(QtCore.QRectF(24, y - 24, w - 48, 14), QtCore.Qt.AlignRight,
                   f"MAX {self.steer_max:.0f}\u00b0 AT THIS SPEED")


class Banner(Glass):
    """Top-centre: what the car is doing (autonomy, manoeuvre) - fades away when there is nothing to say."""

    def __init__(self, parent):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(18, 9, 22, 9)
        lay.setSpacing(12)
        self.dot = QtWidgets.QLabel("")
        self.dot.setFixedSize(8, 8)
        self.text = QtWidgets.QLabel("")
        self.text.setFont(theme.font(15))
        lay.addWidget(self.dot)
        lay.addWidget(self.text)
        self._shown = True
        self.fade(False)
        self.hide()

    def set(self, icon, text, colour=None):
        c = colour or C["accent"]
        if text:
            self.dot.setStyleSheet(f"background: {c}; border-radius: 4px;")
            if text != self.text.text():
                self.text.setText(text)
                self.adjustSize()
        self.fade(bool(text))


class SafetyPill(Glass):
    """Bottom-centre: the brake's view of the path - clear / time to contact."""

    def __init__(self, parent):
        super().__init__(parent)
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(16, 7, 20, 7)
        lay.setSpacing(10)
        self.dot = QtWidgets.QLabel("")
        self.dot.setFixedSize(8, 8)
        self.text = QtWidgets.QLabel("path clear")
        self.text.setFont(theme.font(13, spacing=0.4))
        lay.addWidget(self.dot)
        lay.addWidget(self.text)

    def set(self, text, colour):
        self.dot.setStyleSheet(f"background: {colour}; border-radius: 4px;")
        if text != self.text.text():
            self.text.setText(text)
            self.adjustSize()


class MiniMap(Glass):
    """Bottom-right: the world round the car (heading up), zones, planned path; click to open Map & Zones."""

    clicked = QtCore.Signal()

    def __init__(self, parent, store):
        super().__init__(parent)
        self.setFixedSize(230, 170)
        self.store = store
        self.st = None
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self, _):
        self.clicked.emit()

    def set(self, st):
        self.st = st
        self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        st = self.st
        if not st:
            return
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        scale = 38.0                                   # px per metre
        cx, cy = w / 2, h * 0.62
        world = st.get("world") or {}
        pose = world.get("pose") or [0, 0, 0]

        def to_px(v):                                  # vehicle frame -> screen (ahead = up)
            return cx - v[:, 1] * scale, cy - v[:, 0] * scale
        for z in world.get("zones") or []:
            out = zone_outline(z)
            if len(out) >= 3:
                v = world_to_vehicle(out, pose)
                xs, ys = to_px(v)
                col = QtGui.QColor.fromRgbF(*zone_colour(z["kph"]), 0.35)
                p.setBrush(col)
                p.setPen(QtCore.Qt.NoPen)
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(a, b) for a, b in zip(xs, ys)]))
        wp = self.store.points()
        if len(wp):
            v = world_to_vehicle(wp, pose)
            xs, ys = to_px(v)
            p.setPen(QtGui.QPen(theme.qcolor("text2", 190), 2))
            for a, b in zip(xs, ys):
                if 0 <= a <= w and 0 <= b <= h:
                    p.drawPoint(QtCore.QPointF(a, b))
        man = polar_xy((st.get("plan") or {}).get("maneuver"))
        if len(man) >= 2:
            xs, ys = to_px(man)
            p.setPen(QtGui.QPen(theme.qcolor("accent"), 3))
            p.drawPolyline(QtGui.QPolygonF([QtCore.QPointF(a, b) for a, b in zip(xs, ys)]))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(C["text"]))
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx, cy - 10), QtCore.QPointF(cx - 6, cy + 6),
                                       QtCore.QPointF(cx + 6, cy + 6)]))
        p.setPen(QtGui.QColor(EV["dim"]))
        p.setFont(theme.font(10, spacing=1.2))
        p.drawText(QtCore.QRectF(12, 8, w - 24, 14), QtCore.Qt.AlignLeft, "MAP")


class WorldStore:
    """The room as the car has seen it: LiDAR points in the world frame on a 5 cm grid (a simple map that builds
    up while driving, from the relay's scan-matched pose). Shared by the mini-map and the Map & Zones drawer."""

    def __init__(self, cell=0.05, max_cells=40000):
        self.cell, self.max = cell, max_cells
        self.cells = collections.OrderedDict()
        self.n = 0

    def add(self, st):
        self.n += 1
        if self.n % 3:
            return
        world = st.get("world")
        if not world or "pose" not in world:
            return
        v = polar_xy(st.get("_pts"))
        if not len(v):
            return
        w = vehicle_to_world(v, world["pose"])
        for key in map(tuple, np.round(w / self.cell).astype(int)):
            self.cells[key] = self.cells.get(key, 0) + 1
            self.cells.move_to_end(key)
        while len(self.cells) > self.max:
            self.cells.popitem(last=False)

    def points(self, min_hits=2):
        pts = [k for k, c in self.cells.items() if c >= min_hits]
        return np.array(pts, float) * self.cell if pts else np.empty((0, 2))

    def clear(self):
        self.cells.clear()


# ====================================================================== drawers
class MapZonesPanel(QtWidgets.QWidget):
    """World map (builds up while driving), click-to-go with a heading, and the speed-limit zone editor."""

    def __init__(self, link, store, on_event):
        super().__init__()
        self.link, self.store, self.on_event = link, store, on_event
        self.zones = self._load()
        self.pose = (0.0, 0.0, 0.0)
        self._press, self._poly = None, []
        self._sent_n = None
        lay = QtWidgets.QVBoxLayout(self)
        tools = QtWidgets.QHBoxLayout()
        self.mode = {}
        grp = QtWidgets.QButtonGroup(self)
        for key, text, tip in (("goto", "Go to", "click = drive there; press-drag-release = arrive facing that way"),
                               ("rect", "▭ Zone", "drag a rectangle"), ("circle", "◯ Zone", "drag from the centre"),
                               ("poly", "⬠ Zone", "click the corners, right-click to finish"),
                               ("erase", "Erase", "click a zone to delete it")):
            b = QtWidgets.QPushButton(text)
            b.setCheckable(True)
            b.setToolTip(tip)
            grp.addButton(b)
            self.mode[key] = b
            tools.addWidget(b)
        self.mode["goto"].setChecked(True)
        tools.addWidget(QtWidgets.QLabel("limit"))
        self.kph = QtWidgets.QSpinBox()
        self.kph.setRange(5, 120)
        self.kph.setValue(20)
        self.kph.setSuffix(" km/h")
        tools.addWidget(self.kph)
        lay.addLayout(tools)
        self.hint = QtWidgets.QLabel("")
        self.hint.setStyleSheet(f"color: {EV['dim']}; font-size: 11px;")
        self.hint.setWordWrap(True)
        lay.addWidget(self.hint)
        self.plot = theme.style_plot(pg.PlotWidget())
        self.plot.setAspectLocked(True)
        self.plot.showGrid(x=True, y=True, alpha=0.12)
        self.plot.getPlotItem().hideButtons()
        self.plot.getPlotItem().vb.setMouseEnabled(x=False, y=False)
        self.plot.setXRange(-3, 3)
        self.plot.setYRange(-2, 4)
        self.map_pts = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(180, 186, 195, 150), pen=None)
        self.live = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(C["accent"]), pen=None)
        self.path = pg.PlotCurveItem(pen=pg.mkPen(C["accent"], width=4))
        self.car = pg.PlotCurveItem(pen=pg.mkPen(C["text"], width=2))
        self.draft = pg.PlotCurveItem(pen=pg.mkPen(C["warn"], width=2, style=QtCore.Qt.DashLine))
        for it in (self.map_pts, self.live, self.path, self.car, self.draft):
            self.plot.addItem(it)
        self.zone_items = []
        lay.addWidget(self.plot, 1)
        row = QtWidgets.QHBoxLayout()
        for text, fn in (("Return to start", lambda: (self.link.send("HOME"), self.on_event("return to start"))),
                         ("Send zones to the car", self.send_zones), ("Delete all zones", self.clear_zones),
                         ("Reset origin here", self.reset_origin), ("Clear map", self.store.clear),
                         ("Centre on car", self.centre), ("Cancel autonomy", lambda: self.link.send("GOTO CANCEL"))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        lay.addLayout(row)
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(90)
        lay.addWidget(self.list)
        for key, b in self.mode.items():
            b.toggled.connect(self._update_hint)
        self._update_hint()
        self.plot.scene().installEventFilter(self)
        self._draw_zones()

    # --- zones on disk (so they survive restarts of the GUI and are re-sent to a restarted relay)
    def _load(self):
        try:
            return json.load(open(ZONES_FILE))
        except Exception:
            return []

    def _save(self):
        try:
            json.dump(self.zones, open(ZONES_FILE, "w"))
        except OSError:
            pass

    def current_mode(self):
        return next(k for k, b in self.mode.items() if b.isChecked())

    def _update_hint(self, *_):
        self.hint.setText({"goto": "Click on the map to drive there (hold the throttle as the dead-man switch). Press, "
                                   "drag and release to choose the direction the car arrives in.",
                           "rect": "Drag a rectangle; it gets the limit shown. Then 'Send zones to the car'.",
                           "circle": "Press at the centre and drag out the radius.",
                           "poly": "Click the corners; right-click to close the shape.",
                           "erase": "Click inside a zone to delete it."}[self.current_mode()])

    # --- coordinates: screen up = world x, screen right = world -y (as the car's own view)
    def _world(self, scene_pos):
        p = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        return float(p.y()), float(-p.x())

    @staticmethod
    def _sc(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return -xy[:, 1], xy[:, 0]

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QtCore.QEvent.GraphicsSceneMousePress:
            w = self._world(ev.scenePos())
            mode = self.current_mode()
            if mode == "poly":
                if ev.button() == QtCore.Qt.RightButton:
                    if len(self._poly) >= 3:
                        self._add({"kind": "poly", "pts": [list(p) for p in self._poly]})
                    self._poly = []
                    self.draft.setData([], [])
                else:
                    self._poly.append(w)
                    self.draft.setData(*self._sc(np.array(self._poly + [self._poly[0]])))
                return True
            if mode == "erase":
                from pi.zones import contains
                keep = [z for z in self.zones if not contains(z, *w)]
                if len(keep) != len(self.zones):
                    self.zones = keep
                    self._changed()
                return True
            if ev.button() == QtCore.Qt.LeftButton:
                self._press = w
                return True
        elif t == QtCore.QEvent.GraphicsSceneMouseMove and self._press is not None:
            w = self._world(ev.scenePos())
            x0, y0 = self._press
            mode = self.current_mode()
            if mode == "rect":
                self.draft.setData(*self._sc(np.array([[x0, y0], [w[0], y0], [w[0], w[1]], [x0, w[1]], [x0, y0]])))
            elif mode == "circle":
                r = math.hypot(w[0] - x0, w[1] - y0)
                a = np.linspace(0, 2 * math.pi, 50)
                self.draft.setData(*self._sc(np.column_stack([x0 + r * np.cos(a), y0 + r * np.sin(a)])))
            else:
                self.draft.setData(*self._sc(np.array([[x0, y0], w])))
            return True
        elif t == QtCore.QEvent.GraphicsSceneMouseRelease and self._press is not None:
            w = self._world(ev.scenePos())
            (x0, y0), mode = self._press, self.current_mode()
            self._press = None
            self.draft.setData([], [])
            if mode == "rect" and abs(w[0] - x0) > 0.1 and abs(w[1] - y0) > 0.1:
                self._add({"kind": "rect", "x0": x0, "y0": y0, "x1": w[0], "y1": w[1]})
            elif mode == "circle" and math.hypot(w[0] - x0, w[1] - y0) > 0.1:
                self._add({"kind": "circle", "x": x0, "y": y0, "r": math.hypot(w[0] - x0, w[1] - y0)})
            elif mode == "goto":
                self._goto((x0, y0), None if math.hypot(w[0] - x0, w[1] - y0) < 0.08 else
                           math.atan2(w[1] - y0, w[0] - x0))
            return True
        return False

    def _goto(self, gw, heading_world):
        """A goal in the world frame -> the relay's GOTO in the car's frame now."""
        v = world_to_vehicle(np.array([gw]), self.pose)[0]
        cmd = f"GOTO {v[0]:.3f} {v[1]:.3f}"
        txt = f"go to {v[0]:.2f} m ahead, {v[1]:+.2f} m left"
        if heading_world is not None:
            hd = math.degrees(math.remainder(heading_world - self.pose[2], 2 * math.pi))
            cmd += f" {hd:.1f}"
            txt += f", arriving facing {hd:+.0f} deg"
        self.link.send(cmd)
        self.on_event(txt)

    def _add(self, z):
        z["kph"] = float(self.kph.value())
        z = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in z.items()}
        self.zones.append(z)
        self._changed()

    def _changed(self):
        self._save()
        self._draw_zones()
        self.send_zones()

    def send_zones(self):
        self.link.send("ZONES " + json.dumps(self.zones, separators=(",", ":")))
        self.on_event(f"{len(self.zones)} speed-limit zone(s) sent to the car")

    def clear_zones(self):
        self.zones = []
        self._changed()

    def reset_origin(self):
        self.link.send("ORIGIN")
        self.store.clear()
        self.on_event("world origin reset at the car's position - zones are relative to this")

    def centre(self):
        x, y, _ = self.pose
        self.plot.setXRange(-y - 3, -y + 3)
        self.plot.setYRange(x - 2, x + 4)

    def _draw_zones(self):
        for it in self.zone_items:
            self.plot.removeItem(it)
        self.zone_items = []
        self.list.clear()
        for i, z in enumerate(self.zones):
            out = zone_outline(z)
            if len(out) < 3:
                continue
            xs, ys = self._sc(np.vstack([out, out[:1]]))
            r, g, b = (int(255 * c) for c in zone_colour(z["kph"]))
            curve = pg.PlotCurveItem(xs, ys, pen=pg.mkPen((r, g, b), width=2), fillLevel=None,
                                     brush=pg.mkBrush(r, g, b, 60))
            fill = pg.FillBetweenItem(curve, pg.PlotCurveItem(xs, ys), brush=pg.mkBrush(r, g, b, 50))
            text = pg.TextItem(f"{z['kph']:.0f}", color=(r, g, b), anchor=(0.5, 0.5))
            c = out.mean(axis=0)
            text.setPos(-c[1], c[0])
            for it in (curve, fill, text):
                self.plot.addItem(it)
                self.zone_items.append(it)
            self.list.addItem(f"{i + 1}. {z['kind']} - {z['kph']:.0f} km/h (car {z['kph'] / 3.6 / CAR_SCALE:.2f} m/s)")

    def show_state(self, st):
        world = st.get("world") or {}
        self.pose = tuple(world.get("pose") or (0.0, 0.0, 0.0))
        if world and len(world.get("zones") or []) != len(self.zones) and self._sent_n != len(self.zones):
            self._sent_n = len(self.zones)             # the relay restarted without our zones: send them again
            self.send_zones()
        if not self.isVisible():
            return
        wp = self.store.points()
        self.map_pts.setData(*self._sc(wp)) if len(wp) else self.map_pts.setData([], [])
        v = polar_xy(st.get("_pts"))
        self.live.setData(*self._sc(vehicle_to_world(v, self.pose))) if len(v) else self.live.setData([], [])
        man = polar_xy((st.get("plan") or {}).get("maneuver"))
        self.path.setData(*self._sc(vehicle_to_world(man, self.pose))) if len(man) >= 2 else self.path.setData([], [])
        hw = GEO["width"] / 2
        body = np.array([[GEO["rear"], -hw], [GEO["front"], -hw], [GEO["front"] + 0.05, 0], [GEO["front"], hw],
                         [GEO["rear"], hw], [GEO["rear"], -hw]])
        self.car.setData(*self._sc(vehicle_to_world(body, self.pose)))


class RearCameraPanel(QtWidgets.QWidget):
    """Reverse-camera view: the rear webcam (pi/rear_camera.py, MJPEG on :8091) with the steering-dependent reverse
    guidelines drawn on the floor, objects the vision pipeline found (with looming time to contact) and its status."""

    frame_ready = QtCore.Signal(object)
    state_ready = QtCore.Signal(object)

    def __init__(self, host, port=8091):
        super().__init__()
        self.url, self.state_url = f"http://{host}:{port}/stream", f"http://{host}:{port}/state"
        self.kappa, self.running, self.have = 0.0, False, False
        self.hazards, self.speed = None, 0.0
        lay = QtWidgets.QVBoxLayout(self)
        self.view = QtWidgets.QLabel()
        self.view.setMinimumSize(320, 240)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet(f"background: {C['bg1']}; border: 1px solid {C['hair']}; border-radius: 12px;")
        self.placeholder()
        lay.addWidget(self.view, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {C['text2']};")
        lay.addWidget(self.status)
        row = QtWidgets.QHBoxLayout()
        self.guides = QtWidgets.QPushButton("REVERSE GUIDELINES")
        self.guides.setCheckable(True)
        self.guides.setChecked(True)
        row.addWidget(self.guides)
        self.ghost = QtWidgets.QPushButton("GHOST CAR && ASSIST")
        self.ghost.setCheckable(True)
        self.ghost.setChecked(True)
        row.addWidget(self.ghost)
        row.addStretch(1)
        lay.addLayout(row)
        self.frame_ready.connect(self._on_frame)
        self.state_ready.connect(self._on_state)

    def placeholder(self):
        self.view.setText("NO REAR CAMERA\n\nConnect the webcam to the Pi and start\npi/rear_camera.py  (or  --sim  to try it)")
        self.view.setFont(theme.font(13, spacing=1.0))

    def set_steering(self, kappa):
        self.kappa = kappa

    def set_hazards(self, xy, speed):
        """LiDAR points behind the car (vehicle frame) and the signed speed, for the ghost car and the reverse assists."""
        self.hazards, self.speed = xy, speed

    def showEvent(self, ev):
        super().showEvent(ev)
        if not self.running:
            self.running = True
            threading.Thread(target=self._reader, daemon=True).start()
            threading.Thread(target=self._poller, daemon=True).start()

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self.running = False

    def _reader(self):
        import cv2
        while self.running:
            cap = cv2.VideoCapture(self.url)
            got = False
            while self.running and cap.isOpened():
                ok, f = cap.read()
                if not ok:
                    break
                got = True
                self.frame_ready.emit(f)
            cap.release()
            if not got:
                self.frame_ready.emit(None)
            time.sleep(1.5)

    def _poller(self):
        while self.running:
            try:
                with urllib.request.urlopen(self.state_url, timeout=1.5) as r:
                    self.state_ready.emit(json.loads(r.read()))
            except Exception:
                self.state_ready.emit(None)
            time.sleep(0.5)

    def _on_frame(self, f):
        if f is None:
            self.have = False
            self.placeholder()
            return
        self.have = True
        if self.guides.isChecked() or self.ghost.isChecked():
            try:
                from adas.vision import guidelines as gl
                from tools.camera_calibrate import load_camera
                cam = load_camera()
                cam = type(cam)(**{**cam.__dict__, "width": f.shape[1], "height": f.shape[0]})
                if self.ghost.isChecked():
                    haz = self.hazards
                    if haz is not None and len(haz):
                        haz = haz[haz[:, 0] < GEO["rear"] + 0.02]              # only what is behind the bumper
                    import cv2
                    patches = gl.floor_patches(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cam, self.kappa, GEO["width"], GEO["rear"])
                    f, info = gl.draw_reverse_assist(f, cam, self.kappa, GEO["width"], GEO["rear"], GEO["front"], haz, patches,
                                                     speed=max(0.0, -self.speed))
                    self.assist_info = info
                else:
                    f = gl.draw_guidelines(f, cam, self.kappa, GEO["width"], GEO["rear"])
            except Exception:
                pass
        h, w = f.shape[:2]
        img = QtGui.QImage(f.data, w, h, 3 * w, QtGui.QImage.Format_BGR888).copy()
        self.view.setPixmap(QtGui.QPixmap.fromImage(img).scaled(self.view.size(), QtCore.Qt.KeepAspectRatio,
                                                                QtCore.Qt.SmoothTransformation))

    def _on_state(self, st):
        if not st or not st.get("ok"):
            self.status.setText("camera service not reachable")
            return
        o, q = st.get("odometry"), st.get("quality") or {}
        parts = [f"{st.get('fps', 0):.0f} fps, {st.get('pipeline_ms', 0):.0f} ms per frame"]
        if o:
            parts.append(f"visual odometry {o['v']:+.2f} m/s, yaw {o['w']:+.2f} rad/s (quality {o['quality']:.2f})")
        ttcs = [x["ttc"] for x in st.get("objects", []) if x.get("ttc")]
        parts.append(f"{len(st.get('objects', []))} object(s)" + (f", nearest contact in {min(ttcs):.1f} s" if ttcs else ""))
        parts.append(f"image: blur {q.get('blur', 0):.0f}, shake {q.get('jitter_deg', 0):.2f} deg" +
                     (f"  -  DEGRADED: {', '.join(q.get('why', []))}" if q.get("degraded") else ""))
        self.status.setText("\n".join(parts))


class AssistsPanel(QtWidgets.QWidget):
    GLYPH = {"evasive": "\u2934", "nudge": "\u21c6", "centring": "\u2b1a", "limiter": "\u25d4", "narrow": "\u27f7",
             "proximity": "\u25c9", "moving": "\u27a4"}

    def __init__(self, link):
        super().__init__()
        self.link = link
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 14)
        lay.setSpacing(8)
        lay.addWidget(section_label("Driver assists"))
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        self.btns = {}
        for i, (name, label, desc) in enumerate(ASSISTS):
            t = FeatureTile(self.GLYPH.get(name, "\u25cf"), label, desc)
            t.toggled.connect(lambda on, n=name: self.link.send(f"ASSIST {n} {'ON' if on else 'OFF'}"))
            t.hovered.connect(self._detail)
            grid.addWidget(t, i // 2, i % 2)
            self.btns[name] = t
        lay.addLayout(grid)
        row = QtWidgets.QHBoxLayout()
        for text, cmd in (("ALL ON", "ASSIST all ON"), ("ALL OFF", "ASSIST all OFF")):
            bt = QtWidgets.QPushButton(text)
            bt.setFont(theme.semibold(11, spacing=1.4))
            bt.clicked.connect(lambda _=False, c=cmd: self.link.send(c))
            row.addWidget(bt)
        lay.addLayout(row)
        self.detail = QtWidgets.QLabel("Hover a tile to see what it does. The path-predicted brake, the speed-dependent "
                                       "steering limit and the speed-limit zones are always on.")
        self.detail.setWordWrap(True)
        self.detail.setFont(theme.font(12))
        self.detail.setMinimumHeight(48)
        self.detail.setStyleSheet(f"color: {C['dim']}; padding: 4px 2px;")
        lay.addWidget(self.detail)
        lay.addWidget(section_label("Drive mode"))
        self.modes = Segmented((("eco", "ECO"), ("normal", "NORMAL"), ("sport", "SPORT")))
        self.modes.select("normal")
        self.modes.changed.connect(lambda k: self.link.send(f"MODE {k}"))
        lay.addWidget(self.modes)
        md = QtWidgets.QLabel("Eco limits the throttle to 55 % and eases it in; Sport responds faster. The safety margins "
                              "are the same in every mode.")
        md.setWordWrap(True)
        md.setFont(theme.font(12))
        md.setStyleSheet(f"color: {C['dim']};")
        lay.addWidget(md)
        lay.addWidget(section_label("Autonomy and safety"))
        g2 = QtWidgets.QGridLayout()
        g2.setSpacing(8)
        self.follow = FeatureTile("\u2b95", "Follow the leader", "keeps a set distance behind whatever is in front")
        self.follow.toggled.connect(lambda c: self.link.send("FOLLOW_ON" if c else "FOLLOW_OFF"))
        self.follow.hovered.connect(self._detail)
        self.override = FeatureTile("\u26a0", "ADAS override", "switches EVERY safety function off - the car will hit things",
                                    danger=True)
        self.override.toggled.connect(lambda c: self.link.send("ADAS_OVERRIDE_ON" if c else "ADAS_OVERRIDE_OFF"))
        self.override.hovered.connect(self._detail)
        g2.addWidget(self.follow, 0, 0)
        g2.addWidget(self.override, 0, 1)
        lay.addLayout(g2)
        lay.addStretch(1)

    def _detail(self, text):
        if text:
            self.detail.setText(text)

    def show_state(self, st):
        enabled = (st.get("assist") or {}).get("enabled") or {}
        for name, c in self.btns.items():
            c.setChecked(bool(enabled.get(name)))
        m = (st.get("world") or {}).get("mode")
        if m:
            self.modes.select(m)
        self.follow.setChecked(bool(st.get("follow_enabled")))
        self.override.setChecked(st.get("mode") == "override")


class CarSetupPanel(QtWidgets.QWidget):
    """The Pi's control panel (pi/control_panel.py, port 8080) inside the GUI: drive mode, calibrations with their
    set-up instructions, results and 'apply', and the safety thresholds."""

    MODES = (("drive", "Drive with safety stops"), ("lidar", "Calibrate LiDAR front"),
             ("center", "Calibrate steering centre"), ("speed", "Calibrate speed + stopping"),
             ("turn", "Calibrate turning"), ("oa", "Obstacle-avoidance demo"), ("logdrive", "Logging drive"))
    state_ready = QtCore.Signal(object)

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.snap = None
        lay = QtWidgets.QVBoxLayout(self)
        self.status = QtWidgets.QLabel("connecting to the car's control panel ...")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("font-size: 14px;")
        lay.addWidget(self.status)
        grid = QtWidgets.QGridLayout()
        self.mode_btns = {}
        for i, (key, text) in enumerate(self.MODES):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, k=key: self.start(k))
            grid.addWidget(b, i // 2, i % 2)
            self.mode_btns[key] = b
        stop = QtWidgets.QPushButton("STOP (motors zeroed)")
        stop.setStyleSheet(f"border-color: {EV['bad']}; color: {EV['bad']}; font-weight: 700;")
        stop.clicked.connect(lambda: self.post("/api/stop", {}))
        grid.addWidget(stop, (len(self.MODES) + 1) // 2, 0, 1, 2)
        lay.addLayout(grid)
        lay.addWidget(QtWidgets.QLabel("Calibration results (applied only when you press Apply):"))
        self.results = QtWidgets.QLabel("")
        self.results.setWordWrap(True)
        self.results.setStyleSheet(f"color: {EV['dim']}; font-size: 12px;")
        lay.addWidget(self.results)
        row = QtWidgets.QHBoxLayout()
        for kind, text in (("lidar", "Apply LiDAR"), ("servo_center", "Apply centre"), ("speed", "Apply speed"),
                           ("turn", "Apply turning")):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _=False, k=kind: self.apply(k))
            row.addWidget(b)
        lay.addLayout(row)
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        lay.addWidget(self.log)
        lay.addStretch(1)
        self.state_ready.connect(self._show)
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(2000)
        self.poll()

    def _url(self, path):
        return f"http://{self.host}:{PANEL_PORT}{path}"

    def poll(self):
        if not self.isVisible() and self.snap is not None:
            return

        def work():
            try:
                with urllib.request.urlopen(self._url("/api/state"), timeout=2) as r:
                    self.state_ready.emit(json.loads(r.read()))
            except Exception as e:
                self.state_ready.emit({"error": str(e)})
        threading.Thread(target=work, daemon=True).start()

    def post(self, path, body):
        def work():
            try:
                req = urllib.request.Request(self._url(path), data=json.dumps(body).encode(), method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    res = json.loads(r.read()).get("result")
            except Exception as e:
                res = f"failed: {e}"
            self.state_ready.emit({"message_only": res})
        threading.Thread(target=work, daemon=True).start()

    def start(self, mode):
        prep = ((self.snap or {}).get("modes") or {}).get(mode, {}).get("prep", "")
        if mode != "drive":
            ok = QtWidgets.QMessageBox.question(self, "Start " + mode, (prep or "Start this job?") +
                                                "\n\nThe car may move. Start now?")
            if ok != QtWidgets.QMessageBox.Yes:
                return
        self.post("/api/start", {"mode": mode})

    def apply(self, kind):
        ok = QtWidgets.QMessageBox.question(self, "Apply result", f"Write the {kind} calibration result into the car's "
                                            "tuning file (a backup is kept)?")
        if ok == QtWidgets.QMessageBox.Yes:
            self.post("/api/apply", {"kind": kind})

    def _show(self, s):
        if "message_only" in s:
            self.log.appendPlainText(f"{time.strftime('%H:%M:%S')}  {s['message_only']}")
            self.poll()
            return
        if "error" in s:
            self.status.setText("Control panel not reachable (the Pi's rc-panel service, port 8080) - in the laptop "
                                "simulator there is none.")
            return
        self.snap = s
        run = "running" if s.get("running") else "idle"
        self.status.setText(f"<b>{s.get('mode') or 'no job'}</b> - {run}<br>{s.get('message', '')}<br>"
                            f"relay (drive mode): {'on' if s.get('relay') else 'off'}")
        res = s.get("results") or {}
        lines = []
        if "lidar" in res:
            lines.append(f"LiDAR front: offset {res['lidar'].get('yaw_offset_deg', 0):.1f} deg "
                         f"(was {res['lidar'].get('was', 0):.1f}) - {res['lidar'].get('time', '')}")
        for k in ("servo_center", "speed", "turn"):
            if k in res:
                lines.append(f"{k}: " + ", ".join(f"{a}={b:.3g}" if isinstance(b, float) else f"{a}={b}"
                                                  for a, b in res[k].items()))
        self.results.setText("\n".join(lines) or "no results yet")
        tail = s.get("log") or []
        if tail:
            self.log.setPlainText("\n".join(tail))


class DiagnosticsPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        self.health = QtWidgets.QLabel("")
        self.health.setWordWrap(True)
        lay.addWidget(self.health)
        w = pg.GraphicsLayoutWidget()
        w.setBackground(C["bg1"])
        self.p1 = w.addPlot(title="speed m/s: estimate (cyan) vs throttle model (grey)")
        self.c_v = self.p1.plot(pen=pg.mkPen(EV["cyan"], width=2))
        self.c_vm = self.p1.plot(pen=pg.mkPen(EV["dim"], width=1))
        w.nextRow()
        self.p2 = w.addPlot(title="throttle PWM: driver (grey) vs sent (green)")
        self.c_in = self.p2.plot(pen=pg.mkPen(EV["dim"], width=1))
        self.c_out = self.p2.plot(pen=pg.mkPen(EV["ok"], width=2))
        w.nextRow()
        self.p3 = w.addPlot(title="free way on the path (m, amber) and crash risk (red)")
        self.c_free = self.p3.plot(pen=pg.mkPen(EV["warn"], width=2))
        self.c_risk = self.p3.plot(pen=pg.mkPen(EV["bad"], width=1))
        lay.addWidget(w, 1)
        self.hist = collections.deque(maxlen=900)

    def add(self, now, st):
        d = st.get("drive") or {}
        g = st.get("gate") or {}
        p = (st.get("intent") or {}).get("p_crash")
        self.hist.append((now, float(d.get("v", 0) or 0), float(d.get("v_model", 0) or 0), float(d.get("pwm_in", 0) or 0),
                          float(d.get("pwm_out", 0) or 0), g.get("free_m") if g.get("free_m") is not None else np.nan,
                          p if p is not None else np.nan))
        if not self.isVisible() or len(self.hist) < 3:
            return
        h = np.array(self.hist, float)
        t = h[:, 0] - h[-1, 0]
        for c, i in ((self.c_v, 1), (self.c_vm, 2), (self.c_in, 3), (self.c_out, 4), (self.c_free, 5), (self.c_risk, 6)):
            c.setData(t, np.nan_to_num(h[:, i], nan=0.0))
        hl, esp = st.get("health") or {}, st.get("esp32") or {}
        self.health.setText(f"health: <b>{hl.get('state', '?')}</b> {'; '.join(hl.get('causes') or [])} - LiDAR "
                            f"{hl.get('lidar_hz')} Hz, link {hl.get('link_hz')} packets/s, Pi {hl.get('temp_c')} C<br>"
                            f"ESP32: <b>{esp.get('state', '?')}</b> {esp.get('detail', '')}, reboots {esp.get('reboots', 0)}")


# ====================================================================== the main window
class EVWindow(QtWidgets.QMainWindow):
    def __init__(self, link):
        super().__init__()
        self.link = link
        self.setWindowTitle(f"RC-ADAS - {link.host}")
        scr = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1600, int(scr.width() * 0.94)), min(950, int(scr.height() * 0.92)))
        self.setMinimumSize(900, 560)
        self.store = WorldStore()
        self.last_msgs = {}
        self.labs = {}
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        v.addWidget(self._top())
        self.scene = DriveScene()
        v.addWidget(self.scene, 1)
        v.addWidget(self._appbar())
        # overlays on the 3D scene
        self.cluster = SpeedCluster(self.scene)
        self.banner = Banner(self.scene)
        self.alert = Banner(self.scene)
        self.pill = SafetyPill(self.scene)
        self.mini = MiniMap(self.scene, self.store)
        self.mini.clicked.connect(lambda: self.toggle("map", True))
        self.intent = Glass(self.scene)
        il = QtWidgets.QVBoxLayout(self.intent)
        il.setContentsMargins(14, 8, 14, 8)
        self.intent_txt = QtWidgets.QLabel("DRIVER RISK")
        self.intent_txt.setFont(theme.font(11, spacing=1.2))
        self.intent_bar = QtWidgets.QProgressBar()
        self.intent_bar.setRange(0, 100)
        self.intent_bar.setTextVisible(False)
        il.addWidget(self.intent_txt)
        il.addWidget(self.intent_bar)
        # drawers
        self.events = QtWidgets.QListWidget()
        self.panels = {"map": ("Map and zones", MapZonesPanel(link, self.store, self.add_event)),
                       "assists": ("Assists", AssistsPanel(link)), "setup": ("Car setup", CarSetupPanel(link.host)),
                       "rear": ("Rear camera", RearCameraPanel(link.host)),
                       "diag": ("Diagnostics", DiagnosticsPanel()), "events": ("Events", self.events)}
        self.docks = {}
        for key, (title, w) in self.panels.items():
            d = QtWidgets.QDockWidget(title, self)
            # a drawer must never be wider than the window: long texts wrap, and what still does not fit scrolls
            for lb in w.findChildren(QtWidgets.QLabel):
                if lb.pixmap() is None and not lb.wordWrap() and lb.minimumWidth() == 0 and lb.minimumHeight() == 0:
                    lb.setWordWrap(True)
            if isinstance(w, QtWidgets.QListWidget):
                d.setWidget(w)
            else:
                sa = QtWidgets.QScrollArea()
                sa.setWidgetResizable(True)
                sa.setFrameShape(QtWidgets.QFrame.NoFrame)
                sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
                sa.setWidget(w)
                d.setWidget(sa)
            d.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable | QtWidgets.QDockWidget.DockWidgetFloatable)
            d.setMinimumWidth(360)
            self.addDockWidget(QtCore.Qt.RightDockWidgetArea, d)
            d.hide()
            d.visibilityChanged.connect(lambda vis, k=key: self.app_btns[k].setChecked(vis))
            self.docks[key] = d
        self.tabifyDockWidget(self.docks["assists"], self.docks["setup"])
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(33)

    # --- chrome
    def _chip(self, colour):
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        return (f"background: rgba({r},{g},{b},26); border: 1px solid rgba({r},{g},{b},140); color: {colour}; "
                f"border-radius: 11px; padding: 3px 12px; font-family: '{theme.DISPLAY}'; font-size: 11px; "
                f"letter-spacing: 1.2px; font-weight: 600;")

    def _top(self):
        f = QtWidgets.QFrame()
        f.setObjectName("top")
        lay = QtWidgets.QHBoxLayout(f)
        lay.setContentsMargins(18, 8, 18, 8)
        brand = QtWidgets.QLabel("RC‑ADAS")
        brand.setFont(theme.semibold(17, spacing=4.0))
        lay.addWidget(brand)
        self.clock = QtWidgets.QLabel("")
        self.clock.setFont(theme.font(13))
        self.clock.setStyleSheet(f"color: {EV['dim']}; padding-left: 16px;")
        lay.addWidget(self.clock)
        lay.addStretch(1)
        self.mode = QtWidgets.QLabel("CONNECTING")
        self.mode.setStyleSheet(self._chip(EV["dim"]))
        lay.addWidget(self.mode)
        lay.addStretch(1)
        self.chips = {}
        for k in ("src", "health", "esp", "link"):
            c = QtWidgets.QLabel("")
            lay.addWidget(c)
            self.chips[k] = c
        f.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        return f

    def _appbar(self):
        f = QtWidgets.QFrame()
        f.setObjectName("bar")
        lay = QtWidgets.QHBoxLayout(f)
        lay.setContentsMargins(14, 6, 14, 6)
        self.app_btns = {}
        for key, text in (("map", "⌖  Map && zones"), ("assists", "◎  Assists"), ("setup", "⚙  Car setup"),
                          ("rear", "◉  Rear camera"),
                          ("diag", "∿  Diagnostics"), ("events", "☰  Events")):
            b = QtWidgets.QPushButton(text)
            b.setObjectName("app")
            b.setCheckable(True)
            b.clicked.connect(lambda checked, k=key: self.toggle(k, checked))
            b.setProperty("full", text.replace("&&", "&"))
            b.setToolTip(text.split("  ", 1)[-1].replace("&&", "&"))
            lay.addWidget(b)
            self.app_btns[key] = b
        lay.addStretch(1)
        for key, text in (("mc", "▶  Monte Carlo lab"), ("train", "◆  ML training lab")):
            b = QtWidgets.QPushButton(text)
            b.setObjectName("app")
            b.clicked.connect(lambda _=False, k=key: self.open_lab(k))
            b.setProperty("full", text)
            b.setToolTip(text.split("  ", 1)[-1])
            lay.addWidget(b)
            self.app_btns[key + "_lab"] = b
        lay.addSpacing(20)
        for cam in ("chase", "top", "orbit"):
            b = QtWidgets.QPushButton(cam.title())
            b.setObjectName("app")
            b.clicked.connect(lambda _=False, c=cam: self.scene.camera(c))
            lay.addWidget(b)
        self.reset_btn = QtWidgets.QPushButton("Reset simulated car")
        self.reset_btn.setObjectName("app")
        self.reset_btn.clicked.connect(lambda: self.link.http_post("/api/sim/reset"))
        self.reset_btn.hide()
        lay.addWidget(self.reset_btn)
        f.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)   # never widen the window
        return f

    def resizeEvent(self, ev):
        """Narrow windows: the bottom bar shows icons only (tooltips carry the names) instead of forcing the window wider."""
        super().resizeEvent(ev)
        QtCore.QTimer.singleShot(0, self._place)
        compact = self.width() < 1500
        for b in self.app_btns.values():
            full = b.property("full")
            if full:
                b.setText(full.split("  ", 1)[0] if compact else full.replace("&", "&&"))

    def toggle(self, key, on):
        self.docks[key].setVisible(on)
        if on:
            self.docks[key].raise_()

    def open_lab(self, key):
        from gui.lab_windows import MonteCarloWindow, TrainingWindow
        if key not in self.labs:
            self.labs[key] = MonteCarloWindow() if key == "mc" else TrainingWindow()
        self.labs[key].show()
        self.labs[key].raise_()

    def add_event(self, text):
        self.events.insertItem(0, f"{time.strftime('%H:%M:%S')}  {text}")
        if self.events.count() > 500:
            self.events.takeItem(500)

    def _place(self):
        w, h = self.scene.width(), self.scene.height()
        self.cluster.adjustSize()
        self.cluster.move(20, 20)
        self.intent.setFixedWidth(self.cluster.width())
        self.intent.move(20, 30 + self.cluster.height())
        self.banner.move((w - self.banner.width()) // 2, 18)
        self.alert.move((w - self.alert.width()) // 2, 18 + self.banner.height() + 10)
        self.pill.move((w - self.pill.width()) // 2, h - self.pill.height() - 18)
        self.mini.move(w - self.mini.width() - 20, h - self.mini.height() - 20)

    # --- live
    def refresh(self):
        self.clock.setText(time.strftime("%H:%M"))
        st, t_rx, arrivals = self.link.snapshot()
        now = time.time()
        if st is None or now - t_rx > 1.5:
            self.mode.setText("NO DATA - is the relay / simulator running?")
            self.mode.setStyleSheet(self._chip(EV["bad"]))
            self.chips["link"].setText(f"waiting for {self.link.host}:{CTRL_PORT}")
            self.chips["link"].setStyleSheet(f"color: {EV['dim']};")
            return
        fps = (len(arrivals) - 1) / max(arrivals[-1] - arrivals[0], 1e-3) if len(arrivals) > 1 else 0.0
        self.chips["link"].setText(f"{self.link.host}  {fps:.0f} Hz")
        self.chips["link"].setStyleSheet(f"color: {EV['dim']}; padding-left: 10px;")
        self.store.add(st)
        drive, assist, gate = st.get("drive") or {}, st.get("assist") or {}, st.get("gate") or {}
        plan, nav, intent = st.get("plan") or {}, st.get("nav") or {}, st.get("intent") or {}
        world = st.get("world") or {}
        info = assist.get("info") or {}
        act = str(gate.get("action") or "")
        # mode
        if st.get("mode") == "override":
            mode, col = "OVERRIDE - NO ADAS", EV["bad"]
        elif nav.get("state") not in (None, "idle"):
            mode, col = "AUTONOMY", C["accent"]
        elif assist.get("evading") or assist.get("phase") in ("EXECUTE", "WAIT", "BACKOFF"):
            mode, col = "AUTONOMOUS MANOEUVRE", EV["cyan"]
        elif "brak" in act or act.startswith("hold") or act == "stopped":
            mode, col = "EMERGENCY BRAKE", EV["bad"]
        else:
            mode, col = "GUARDIAN", EV["ok"]
        self.mode.setText(mode.upper())
        self.mode.setStyleSheet(self._chip(col))
        hl, esp = st.get("health") or {}, st.get("esp32") or {}
        for key, ok_text, data, ok_state in (("health", "HEALTH", hl, "normal"), ("esp", "ESP32", esp, "ok")):
            s = str(data.get("state", "")) if data else ""
            if not s:
                self.chips[key].setText("")
                continue
            c = EV["ok"] if s == ok_state else EV["warn"] if s in ("limp", "connecting") else EV["bad"]
            self.chips[key].setText(ok_text + ("" if s == ok_state else f" {s.upper()}"))
            self.chips[key].setStyleSheet(self._chip(c))
            self.chips[key].setToolTip("; ".join(data.get("causes") or []) or str(data.get("detail", "")))
        sim = st.get("sim")
        self.chips["src"].setText(f"SIMULATOR \u00b7 {sim.get('world', '').upper()}" if sim else "CAR")
        self.chips["src"].setStyleSheet(self._chip(C["text2"]))
        self.reset_btn.setVisible(bool(sim))
        # cluster
        v = float(drive.get("v", 0.0) or 0.0)
        pin = float(drive.get("pwm_in", 0) or 0)
        gear = "D" if pin > 5 else "R" if pin < -5 else ("N" if abs(v) > 0.03 else "P")
        steer = float(drive.get("servo", 87)) - float(drive.get("centre", 87))
        self.cluster.set(v, gear, world.get("zone_kph"), -steer, drive.get("steer_max_deg"))
        self.cluster.tick(0.033)
        # banner: autonomy / manoeuvre
        if nav.get("state") not in (None, "idle") or nav.get("msg"):
            self.banner.set("➜", f"{nav.get('msg') or nav.get('state')}", EV["violet"])
        elif info.get("evasive"):
            self.banner.set("⤳", str(info["evasive"]), EV["cyan"])
        else:
            self.banner.set("", "")
        # alerts
        ttc = plan.get("ttc") if plan.get("hit") else None
        if hl.get("state") in ("limp", "fault"):
            self.alert.set("⚠", f"{hl['state'].upper()}: {'; '.join(hl.get('causes') or [])}", EV["warn"])
        elif "brak" in act or act.startswith("hold"):
            self.alert.set("■", "Emergency brake - obstacle on the path", EV["bad"])
        elif ttc is not None and ttc < 1.6 and abs(v) > 0.1:
            if not self.last_msgs.get("fcw"):
                QtWidgets.QApplication.beep()
                self.add_event(f"collision warning: contact in {ttc:.1f} s")
            self.alert.set("⚠", f"Collision warning - contact in {ttc:.1f} s", EV["warn"])
        elif info.get("zone"):
            self.alert.set("◷", str(info["zone"]), EV["blue"])
        else:
            self.alert.set("", "")
        self.last_msgs["fcw"] = ttc is not None and ttc < 1.6 and abs(v) > 0.1
        for k, val in list(info.items()) + [("gate", gate.get("action"))]:
            if val and self.last_msgs.get(k) != val:
                self.add_event(f"{k}: {val}")
            self.last_msgs[k] = val
        # safety pill
        if plan.get("hit") and ttc is not None:
            c = EV["bad"] if ttc < 1.0 else EV["warn"]
            self.pill.set(f"contact in {plan.get('hit_m', 0):.2f} m / {ttc:.1f} s on this path", c)
        else:
            free = gate.get("free_m")
            rss = (world or {}).get("rss_min_m")
            txt = "path clear" + (f" \u00b7 {free:.1f} m free" if free else "")
            col = EV["ok"]
            if free and rss is not None and abs(v) > 0.05:
                txt += f" \u00b7 RSS needs {rss:.2f} m"
                if free < rss:
                    col = EV["warn"]
                    txt += " (inside)"
            self.pill.set(txt, col)
        p = intent.get("p_crash")
        self.intent_txt.setText(f"driver risk {100 * p:.0f} %  -  " + ("attentive" if intent.get("attentive") else "stick idle")
                                if p is not None else "driver risk -")
        self.intent_bar.setValue(int(100 * (p or 0)))
        self.scene.car.pulse(now, "bad" if mode == "EMERGENCY BRAKE" else "warn" if hl.get("state") == "limp" else "accent")
        self.scene.show(st)
        self.mini.set(st)
        self.panels["map"][1].show_state(st)
        self.panels["assists"][1].show_state(st)
        cal = (st.get("drive") or {}).get("steer_cal") or {}
        kap = -(cal.get("k", 0.0656)) * (float(drive.get("servo", 87)) - float(drive.get("centre", 87)))
        self.panels["rear"][1].set_steering(kap)
        self.panels["rear"][1].set_hazards(polar_xy(st.get("_pts")), float(drive.get("v", 0.0) or 0.0))
        self.panels["diag"][1].add(now, st)
        self._place()
