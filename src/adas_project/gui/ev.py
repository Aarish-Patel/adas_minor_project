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
import OpenGL
OpenGL.ERROR_CHECKING = False      # PyOpenGL's glGetError after every call was ~5 % of a frame; nothing here relies on it
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from gui.controls import Card, FeatureTile, Segmented, TileGroup, ZoneRow, section_label
from gui.render import FrameClock, PoseSmoother, Smooth, SmoothAngle, StateInterpolator
from gui.dashboard import ASSISTS, CTRL_PORT, GEO, polar_xy

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CAR_SCALE = 14.0
PANEL_PORT = 8080
ZONES_FILE = os.path.join(ROOT, "gui", "zones.json")
LANE_FILE = os.path.join(ROOT, "gui", "parking_lane.json")   # the parking lane drawn with the Parking lane tool (relay / map frame)

from gui import theme
from gui.theme import C, rgb

# legacy names used through this file, mapped onto the design system (graphite + brass, not navy + cyan)
def _ev_tokens():
    return {"bg": C["bg"], "bg2": C["bg1"], "edge": C["hair"], "text": C["text"], "dim": C["dim"], "blue": C["accent"],
            "cyan": C["accent"], "ok": C["ok"], "warn": C["warn"], "bad": C["bad"], "violet": C["info"]}


EV = _ev_tokens()
theme.on_change(lambda: EV.update(_ev_tokens()))
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


def loft(sections):
    """A closed body from cross-sections along x: [(x, half_width, z_bottom, z_top, half_width_top)] - each section is a
    trapezoid (narrower at the top), consecutive sections are joined, the two ends capped. Gives the EV silhouette
    (tapered nose and tail, tumblehome cabin) instead of a box."""
    V = []
    for x, hw, zb, zt, hwt in sections:
        V += [[x, -hw, zb], [x, hw, zb], [x, hwt, zt], [x, -hwt, zt]]
    V = np.array(V, float)
    F = []
    for i in range(len(sections) - 1):
        a, b = 4 * i, 4 * (i + 1)
        for k in range(4):
            k2 = (k + 1) % 4
            F += [[a + k, a + k2, b + k2], [a + k, b + k2, b + k]]
    n = len(sections)
    F += [[0, 1, 2], [0, 2, 3], [4 * (n - 1), 4 * (n - 1) + 2, 4 * (n - 1) + 1], [4 * (n - 1), 4 * (n - 1) + 3, 4 * (n - 1) + 2]]
    return V, np.array(F)


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
    """Vertical quads over floor segments ((N, 4): x1, y1, x2, y2). Vectorised: it runs several times per packet."""
    seg = np.asarray(segments, float).reshape(-1, 4)
    if not len(seg):
        return None
    n = len(seg)
    x1, y1, x2, y2 = seg[:, 0], seg[:, 1], seg[:, 2], seg[:, 3]
    z0, z1 = np.zeros(n), np.full(n, float(h))
    V = np.stack([np.column_stack([x1, y1, z0]), np.column_stack([x2, y2, z0]),
                  np.column_stack([x2, y2, z1]), np.column_stack([x1, y1, z1])], axis=1).reshape(-1, 3)
    base = 4 * np.arange(n)[:, None]
    F = np.concatenate([base + [0, 1, 2], base + [0, 2, 3]], axis=1).reshape(-1, 3)
    return V, F


def solid_box(cx, cy, length, width, heading, height, z0=0.0):
    """A rectangular block standing on the floor, rotated by `heading` about its centre (closed, so it has a top and four
    sides and shades like an object - not a single flat quad)."""
    v, f = box(-length / 2, length / 2, -width / 2, width / 2, z0, z0 + height)
    c, s_ = math.cos(heading), math.sin(heading)
    v = v.copy()
    x, y = v[:, 0].copy(), v[:, 1].copy()
    v[:, 0], v[:, 1] = cx + c * x - s_ * y, cy + s_ * x + c * y
    return v, f


def wall_slabs(segments, height, thickness=0.05):
    """Room walls as solid slabs: each segment becomes a block `thickness` thick (both faces and the top exist)."""
    parts = []
    for x1, y1, x2, y2 in np.asarray(segments, float).reshape(-1, 4):
        length = math.hypot(x2 - x1, y2 - y1)
        if length > 1e-6:
            parts.append(solid_box((x1 + x2) / 2, (y1 + y2) / 2, length + thickness, thickness,
                                   math.atan2(y2 - y1, x2 - x1), height))
    return merge(parts) if parts else None


LIGHT_DIR = np.array([-0.35, -0.45, 0.82])            # towards the light: from above, slightly from the viewer's left


def lit_faces(V, F, colour):
    """Per-face colours: `colour` scaled by a Lambert term against LIGHT_DIR (ambient 0.55) - the cheap shading that makes
    boxes read as solids without a shader."""
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    k = 0.55 + 0.45 * np.abs(n @ (LIGHT_DIR / np.linalg.norm(LIGHT_DIR)))
    out = np.tile(np.asarray(colour, float), (len(F), 1))
    out[:, :3] = np.clip(out[:, :3] * k[:, None] * 1.15, 0, 1)
    return out


def point_walls(xy, h=0.10, half=0.02, join=0.12):
    """Obstacles as surfaces: consecutive LiDAR points closer than `join` become one wall strip (a continuous
    face, like a mapped surface), isolated points a small tile facing the car. Vectorised (no per-point Python loop)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < 2:
        return None
    d = np.hypot(xy[1:, 0] - xy[:-1, 0], xy[1:, 1] - xy[:-1, 1])
    near = d < join
    lone = np.ones(len(xy), bool)
    lone[:-1] &= ~near
    lone[1:] &= ~near
    parts = [np.hstack([xy[:-1][near], xy[1:][near]])]
    p = xy[lone]
    if len(p):
        a = np.arctan2(p[:, 1], p[:, 0]) + math.pi / 2
        dx, dy = np.cos(a) * half, np.sin(a) * half
        parts.append(np.column_stack([p[:, 0] - dx, p[:, 1] - dy, p[:, 0] + dx, p[:, 1] + dy]))
    seg = np.vstack(parts)
    return walls_mesh(seg, h) if len(seg) else None


def cyl(centre, radius, length, axis, n=18):
    """A cylinder as (V, F): centre (x, y, z), the axis 'x', 'y' or 'z', capped at both ends."""
    cx, cy, cz = centre
    k = "xyz".index(axis)
    b, c = [i for i in range(3) if i != k]
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    V = []
    for end in (-0.5, 0.5):
        for t in ang:
            v = [cx, cy, cz]
            v[k] += end * length
            v[b] += radius * math.cos(t)
            v[c] += radius * math.sin(t)
            V.append(v)
    for end in (-0.5, 0.5):
        v = [cx, cy, cz]
        v[k] += end * length
        V.append(v)
    F = []
    for i in range(n):
        j = (i + 1) % n
        F += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
    return np.array(V, float), np.array(F)


def car_geometry(colour=(0.78, 0.82, 0.90, 1.0)):
    """The car as a hot rod with the LiDAR on top, in the vehicle frame (x ahead of the rear axle, y left, z up), as a list
    of (name, (V, F), colour). Everything stays inside the real footprint - GEO rear .. front in x, +-width/2 in y - so the
    render never passes through what the LiDAR sees (tests/test_gui_smoke.py checks the bounds)."""
    r, f, w = GEO["rear"], GEO["front"], GEO["width"] / 2
    wb, lx = GEO["wheelbase"], GEO["lidar_x"]
    L = f - r
    at = lambda u: r + u * L
    glass, tyre, hub = (0.07, 0.09, 0.13, 1), (0.035, 0.035, 0.04, 1), (0.78, 0.80, 0.84, 1)
    chrome, dark = (0.88, 0.90, 0.93, 1), (0.10, 0.11, 0.14, 1)
    body = loft([(r + 0.005, w * 0.70, 0.040, 0.072, w * 0.55), (at(0.12), w * 0.90, 0.032, 0.085, w * 0.72),
                 (at(0.35), w * 0.96, 0.032, 0.088, w * 0.78), (at(0.55), w * 0.92, 0.032, 0.084, w * 0.74),
                 (at(0.70), w * 0.90, 0.034, 0.090, w * 0.62), (at(0.92), w * 0.80, 0.036, 0.082, w * 0.56),
                 (f - 0.004, w * 0.62, 0.040, 0.066, w * 0.48)])
    cabin = loft([(at(0.32), w * 0.72, 0.086, 0.090, w * 0.62), (at(0.40), w * 0.72, 0.088, 0.118, w * 0.56),
                  (at(0.58), w * 0.70, 0.088, 0.118, w * 0.54), (at(0.64), w * 0.72, 0.086, 0.095, w * 0.62)])
    wheels_t, wheels_h = [], []
    for x0, rad, wid in ((0.0, 0.034, 0.030), (wb, 0.028, 0.022)):
        for sgn in (-1, 1):
            yc = sgn * (w - wid / 2 - 0.002)                  # the tyre's outer face sits 2 mm inside the footprint
            wheels_t.append(cyl((x0, yc, rad), rad, wid, "y"))
            wheels_h.append(cyl((x0, sgn * (w - 0.002), rad), rad * 0.62, 0.004, "y"))
    pipes = [cyl((at(0.50), sgn * (w - 0.010), 0.046), 0.008, 0.28 * L, "x", 12) for sgn in (-1, 1)]
    lights_f = [cyl((f - 0.003, sgn * 0.040, 0.060), 0.012, 0.006, "x", 14) for sgn in (-1, 1)]
    lights_r = [box(r, r + 0.005, sgn * 0.055 - 0.015, sgn * 0.055 + 0.015, 0.056, 0.068) for sgn in (-1, 1)]
    scoop = box(at(0.72), at(0.86), -0.020, 0.020, 0.086, 0.103)
    lidar = [cyl((lx, 0.0, 0.126), 0.012, 0.016, "z", 14),        # mount post
             cyl((lx, 0.0, 0.150), 0.034, 0.032, "z", 24)]        # the RPLIDAR body (~7 cm across)
    cap = cyl((lx, 0.0, 0.1675), 0.028, 0.003, "z", 24)
    return [("body", body, colour), ("cabin", cabin, glass), ("tyres", merge(wheels_t), tyre), ("hubs", merge(wheels_h), hub),
            ("pipes", merge(pipes), chrome), ("head", merge(lights_f), (1.0, 0.96, 0.85, 1)),
            ("tail", merge(lights_r), (0.86, 0.16, 0.18, 1)), ("scoop", scoop, dark), ("lidar", merge(lidar), dark),
            ("lidar_cap", cap, (0.30, 0.55, 1.0, 1))]


class CarModel:
    """A small hot rod with a LiDAR puck on top, and the ground ring (pulsing). All parts hang off one parent, so the whole
    car moves with one transform (labs animate several); items[0] is the body (labs colour it by risk). The LiDAR's
    marker spins while the car is shown."""

    def __init__(self, view, colour=(0.78, 0.82, 0.90, 1.0), ring=None, scale=1.0):
        ring = ring or rgb('accent', 0.85)
        self.scale = scale
        a = np.linspace(0, 2 * math.pi, 60)
        self.parent = gl.GLLinePlotItem(pos=np.column_stack([0.1 + 0.30 * np.cos(a), 0.30 * np.sin(a),
                                                             np.full(60, 0.003)]), color=ring, width=3,
                                        antialias=True)
        view.addItem(self.parent)
        self.items = []
        for k, (name, (V, F), col) in enumerate(car_geometry(colour)):
            edge = tuple(0.78 * c for c in col[:3]) + (1.0,)
            it = gl.GLMeshItem(vertexes=V, faces=F, color=col, smooth=(name in ("tyres", "hubs", "pipes", "lidar", "lidar_cap")),
                               drawEdges=name in ("body", "cabin"), edgeColor=edge, glOptions="opaque")
            it.setParentItem(self.parent)
            self.items.append(it)
            if name == "lidar_cap":
                self.spin_item = it
        # a small bright marker on the LiDAR cap that goes round with it (the scan direction)
        lx = GEO["lidar_x"]
        mk = gl.GLMeshItem(vertexes=box(lx + 0.010, lx + 0.026, -0.004, 0.004, 0.1695, 0.1725)[0],
                           faces=box(0, 1, 0, 1, 0, 1)[1], color=(1.0, 1.0, 1.0, 1), smooth=False, glOptions="opaque")
        mk.setParentItem(self.parent)
        self.marker = mk
        self._lx = lx

    def _spin(self):
        m = QtGui.QMatrix4x4()
        m.translate(self._lx, 0, 0)
        m.rotate((time.time() * 540.0) % 360.0, 0, 0, 1)         # ~1.5 turns a second, like the real sensor at 5.5 Hz x ...
        m.translate(-self._lx, 0, 0)
        self.marker.setTransform(m)

    def pulse(self, t, colour_name="accent", speed=1.6):
        a = 0.55 + 0.35 * math.sin(t * speed)
        self.parent.setData(color=rgb(colour_name, a))
        self._spin()

    def place(self, x, y, th_rad):
        m = QtGui.QMatrix4x4()
        m.translate(x, y, 0)
        m.rotate(math.degrees(th_rad), 0, 0, 1)
        m.scale(self.scale, self.scale, self.scale)
        self.parent.setTransform(m)
        self._spin()

    def set_visible(self, on):
        self.parent.setVisible(on)


class Scene(gl.GLViewWidget):
    """Shared 3D stage: floor, camera presets, meshes that can be replaced every frame."""

    def __init__(self, grid=16):
        super().__init__()
        self.setBackgroundColor(C["scene_bg"])
        g = gl.GLGridItem()
        g.setSize(grid, grid)
        g.setSpacing(0.5, 0.5)
        g.setColor(C["grid"])
        self.addItem(g)
        self.grid = g
        self.group = gl.GLGraphicsItem.GLGraphicsItem()      # everything that lives in the frame of the newest packet: the
        self.addItem(self.group)                             # renderer moves it as one to follow the interpolated pose
        self.world_group = gl.GLGraphicsItem.GLGraphicsItem()   # objects fixed in the WORLD frame (parking): built once, only
        self.addItem(self.world_group)                          # this group's matrix changes as the car moves
        self._meshes = {}

    def mesh(self, key, data, colour, opts="translucent", lit=False, world=False):
        """lit=True shades each face by its orientation to a fixed key light (tops bright, sides darker: solid objects);
        the default is one flat colour (paths, zones, floor patches)."""
        it = self._meshes.get(key)
        if data is None:
            if it is not None:
                it.setVisible(False)
            return
        V, F = data
        face_colours = lit_faces(V, F, colour) if lit else None
        if it is None:
            it = gl.GLMeshItem(vertexes=V, faces=F, color=colour, faceColors=face_colours, smooth=False, glOptions=opts)
            it.setParentItem(self.world_group if world else self.group)
            self._meshes[key] = it
        else:
            it.setMeshData(vertexes=V, faces=F, color=colour, faceColors=face_colours)
            if face_colours is None:
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
        self.goal.setParentItem(self.group)
        self.hit = gl.GLLinePlotItem(pos=np.zeros((2, 3)), mode="lines", color=(0.94, 0.32, 0.31, 0.0), width=5)
        self.hit.setParentItem(self.group)
        self.labels, self._label_src = [], []
        self._pkt_pose = None
        self._park_keys = set()
        self.idle_rings = []                       # range rings + an expanding pulse, shown while there is no data
        for _ in range(5):
            it = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 1, 1, 0), width=2, antialias=True)
            it.setVisible(False)
            self.addItem(it)
            self.idle_rings.append(it)
        self.cam = "chase"
        self.camera("chase")

    def set_idle(self, on, t=0.0):
        """No data: three faint range rings (1, 2, 3 m) round the car and a pulse that keeps expanding from it, so the
        empty scene reads as 'listening', not as a crashed view."""
        a = np.linspace(0, 2 * math.pi, 96)
        radii = [1.0, 2.0, 3.0, 0.3 + (t * 0.9) % 2.7]
        for i, it in enumerate(self.idle_rings[:4]):
            it.setVisible(on)
            if not on:
                continue
            r = radii[i]
            alpha = 0.22 if i < 3 else 0.55 * (1 - (r - 0.3) / 2.7)
            it.setData(pos=np.column_stack([r * np.cos(a), r * np.sin(a), np.full(96, 0.01)]), color=(*rgb("accent")[:3], alpha))

    def set_view(self, pkt_pose, view_pose, align=(0.0, 0.0, 0.0)):
        """Between packets: move the newest packet's scene (obstacles, paths, zones - all in the vehicle frame of the packet)
        to where the interpolated pose says the car is, and slide the floor grid under the car so the ground visibly
        moves. Called every frame; only two 4x4 matrices and the few zone labels change."""
        s = self.grid_spacing
        x, y, th = view_pose
        c, sn = math.cos(th), math.sin(th)
        g = QtGui.QMatrix4x4()                       # world-fixed lattice seen from the car: R(-th) * T(-(xy mod spacing))
        g.rotate(math.degrees(-th), 0, 0, 1)
        g.translate(-(x % s), -(y % s), 0)
        self.grid.setTransform(g)
        wm = QtGui.QMatrix4x4()                      # world -> car: R(-th) * T(-xy)
        wm.rotate(math.degrees(-th), 0, 0, 1)
        wm.translate(-x, -y, 0)
        al = QtGui.QMatrix4x4()                      # simulator frame -> the relay's estimated frame (align = dth, tx, ty)
        al.translate(align[1], align[2], 0)
        al.rotate(math.degrees(align[0]), 0, 0, 1)
        self.world_group.setTransform(wm * al)
        if pkt_pose is None:
            self.group.resetTransform()
            return
        dx, dy = pkt_pose[0] - x, pkt_pose[1] - y
        m = QtGui.QMatrix4x4()                       # D = P_view^-1 * P_packet
        m.translate(c * dx + sn * dy, -sn * dx + c * dy, 0)
        m.rotate(math.degrees(pkt_pose[2] - th), 0, 0, 1)
        self.group.setTransform(m)
        if self._label_src:
            for t, (lx, ly, _, _) in zip(self.labels, self._label_src):
                q = m.map(QtGui.QVector3D(lx, ly, 0.02))
                t.setData(pos=(q.x(), q.y(), q.z()))

    grid_spacing = 0.5

    def camera(self, which):
        self.cam = which
        if which == "chase":
            self.chase((0.7, 0.0), 180, 2.2, 20)
        elif which == "top":
            self.setCameraPosition(pos=QtGui.QVector3D(0.8, 0, 0), distance=6.0, elevation=89, azimuth=180)
        else:
            self.setCameraPosition(pos=QtGui.QVector3D(0.4, 0, 0), distance=3.2, elevation=35, azimuth=135)

    def _labels(self, items):
        """items: [(x, y, text, rgb)] - reuses GLTextItems (drawn in the screen projection, so set_view() moves them)."""
        self._label_src = list(items)
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

    def show_parking(self, model, selected=None, hover=None):
        """The Parking Simulator objects (bay pads, parked vehicles, road, sidewalk, marker) as meshes in the WORLD frame
        (`world_group` puts them under the car every frame). Rebuilt only when the layout, the selection or the hover changes,
        not per packet. model None hides them."""
        if model is None:
            for key in self._park_keys:
                self.mesh(key, None, (0, 0, 0, 0), world=True)
            return
        from gui.parking import park_meshes
        for key, (data, col, lit) in park_meshes(model, None, selected, hover).items():
            self._park_keys.add(key)
            self.mesh(key, data, col, opts="opaque" if lit else "translucent", lit=lit, world=True)

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
        self.mesh("obstacles", point_walls(pts[idle]), C["obstacle"])
        self.mesh("obst_path", point_walls(pts[on_path & ~red]), (*rgb('accent')[:3], 0.95))
        self.mesh("obst_hit", point_walls(pts[red]), (*rgb('bad')[:3], 0.95))
        self._proximity_arcs(pts)
        state = plan.get("state", "clear")
        col = {"collision": rgb('bad')[:3], "limited": rgb('warn')[:3]}.get(state, (0.93, 0.94, 0.95))
        pred_xy = polar_xy(plan.get("pred"))
        half = max(2, len(pred_xy) // 2)
        slowing = state in ("limited", "collision")
        self.mesh("pred", ribbon(pred_xy[:half + 1], GEO["width"] * 0.9, 0.004), (*col, 0.36))
        self.mesh("pred_far", ribbon(pred_xy[half:], GEO["width"] * 0.9, 0.004) if len(pred_xy) > half + 1 else None,
                  (*col, 0.10 if slowing else 0.26))      # faded = the car will slow down there
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
        slot = polar_xy((st.get("nav") or {}).get("slot"))
        if len(slot) >= 4:                                  # parallel parking: the slot the car chose, a blue pad with its outline
            V = np.column_stack([slot[:4], np.full(4, 0.005)])
            self.mesh("slot", (V, np.array([[0, 1, 2], [0, 2, 3]])), rgb('accent', 0.28))
            self.mesh("slot_edge", ribbon(np.vstack([slot[:4], slot[:1]]), 0.02, 0.007), rgb('accent', 0.95))
        else:
            self.mesh("slot", None, rgb('accent', 0.0))
            self.mesh("slot_edge", None, rgb('accent', 0.0))
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
    """Bottom-centre status pill in the style of production cockpits: a coloured icon disc (tick / warning / cross), the
    state as a headline, and the numbers (free distance, RSS need) as a quieter second part."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame#glass {{ background: {theme.css_rgba('surface', 0.92)}; border: 1px solid {C['hair']}; "
                           f"border-radius: 24px; }} QLabel {{ background: transparent; }}")
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(10, 8, 22, 8)
        lay.setSpacing(12)
        self.disc = QtWidgets.QLabel("✓")
        self.disc.setFixedSize(30, 30)
        self.disc.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.disc)
        self.head = QtWidgets.QLabel("Path clear")
        self.head.setFont(theme.semibold(15))
        lay.addWidget(self.head)
        self.sub = QtWidgets.QLabel("")
        self.sub.setFont(theme.font(12))
        self.sub.setStyleSheet(f"color: {C['dim']};")
        lay.addWidget(self.sub)
        self._last = None

    def set(self, text, colour):
        if (text, colour) == self._last:
            return
        self._last = (text, colour)
        head, _, rest = text.partition(" · ")
        glyph = "✓" if colour == EV["ok"] else "✕" if colour == EV["bad"] else "…" if colour == C["dim"] else "!"
        r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
        self.disc.setText(glyph)
        self.disc.setStyleSheet(f"background: rgba({r},{g},{b},40); color: {colour}; border: 1px solid rgba({r},{g},{b},150); "
                                f"border-radius: 15px; font-size: 15px; font-weight: 700;")
        self.head.setText(head[:1].upper() + head[1:])
        self.sub.setText(rest)
        self.sub.setVisible(bool(rest))
        self.adjustSize()

    def fit(self, avail):
        """Drop the numbers when the scene is too narrow for the whole pill (never let it run off the edge)."""
        if self.sub.text():
            self.sub.setVisible(True)
            self.adjustSize()
            if self.width() > avail:
                self.sub.setVisible(False)
                self.adjustSize()


def qpoly(xs, ys):
    """A QPolygonF from two arrays without a Python loop over the points."""
    poly = pg.functions.create_qpolygonf(len(xs))
    arr = pg.functions.ndarray_from_qpolygonf(poly)
    arr[:, 0], arr[:, 1] = xs, ys
    return poly


class MiniMap(Glass):
    """Bottom-right: the world round the car (heading up), zones, planned path; click to open Map & Zones."""

    clicked = QtCore.Signal()

    def __init__(self, parent, store):
        super().__init__(parent)
        self.setFixedSize(230, 170)
        self.store = store
        self.st = None
        self.view_pose = (0.0, 0.0, 0.0)               # the interpolated pose the map is drawn from (set every frame)
        self.zones_w, self.man_w = [], None            # world-frame copies of the newest packet's zones and manoeuvre path
        self.setCursor(QtCore.Qt.PointingHandCursor)

    def mousePressEvent(self, _):
        self.clicked.emit()

    def set(self, st):
        """A new packet: keep its zones and path in the WORLD frame, so the frames in between can redraw them from the
        interpolated pose without a jump."""
        self.st = st
        self.zones_w, self.man_w = [], None
        if st:
            world = st.get("world") or {}
            pose = world.get("pose") or (0.0, 0.0, 0.0)
            for z in world.get("zones") or []:
                out = zone_outline(z)
                if len(out) >= 3:
                    self.zones_w.append((out, z["kph"]))
            man = polar_xy((st.get("plan") or {}).get("maneuver"))
            self.man_w = vehicle_to_world(man, pose) if len(man) >= 2 else None
        self.update()

    def set_view(self, pose):
        self.view_pose = pose
        if self.st:
            self.update()

    def paintEvent(self, ev):
        super().paintEvent(ev)
        st = self.st
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not st:
            self._paint_idle(p, w, h)
            return
        scale = 38.0                                   # px per metre
        cx, cy = w / 2, h * 0.62
        pose = self.view_pose

        def to_px(v):                                  # vehicle frame -> screen (ahead = up)
            return cx - v[:, 1] * scale, cy - v[:, 0] * scale
        for out, kph in self.zones_w:
            xs, ys = to_px(world_to_vehicle(out, pose))
            p.setBrush(QtGui.QColor.fromRgbF(*zone_colour(kph), 0.35))
            p.setPen(QtCore.Qt.NoPen)
            p.drawPolygon(qpoly(xs, ys))
        wp = self.store.points()
        if len(wp):
            xs, ys = to_px(world_to_vehicle(wp, pose))
            keep = (xs >= 0) & (xs <= w) & (ys >= 0) & (ys <= h)
            if keep.any():
                p.setPen(QtGui.QPen(theme.qcolor("text2", 190), 2))
                p.drawPoints(qpoly(xs[keep], ys[keep]))
        if self.man_w is not None:
            xs, ys = to_px(world_to_vehicle(self.man_w, pose))
            p.setPen(QtGui.QPen(theme.qcolor("accent"), 3))
            p.drawPolyline(qpoly(xs, ys))
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(QtGui.QColor(C["text"]))
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx, cy - 10), QtCore.QPointF(cx - 6, cy + 6),
                                       QtCore.QPointF(cx + 6, cy + 6)]))
        p.setPen(QtGui.QColor(EV["dim"]))
        p.setFont(theme.font(10, spacing=1.2))
        p.drawText(QtCore.QRectF(12, 8, w - 24, 14), QtCore.Qt.AlignLeft, "MAP")


    @staticmethod
    def _paint_idle(p, w, h):
        """No data yet: an empty radar - range rings round the car, a forward cone, a caption - instead of a blank box."""
        cx, cy = w / 2, h * 0.68
        p.setBrush(QtCore.Qt.NoBrush)
        for i, r in enumerate((26, 52, 78, 104)):
            p.setPen(QtGui.QPen(theme.qcolor("hair2", 150 - 25 * i), 1))
            p.drawEllipse(QtCore.QPointF(cx, cy), r, r)
        p.setPen(QtGui.QPen(theme.qcolor("hair2", 120), 1, QtCore.Qt.DashLine))
        p.drawLine(QtCore.QPointF(cx, cy), QtCore.QPointF(cx, 26))
        cone = QtGui.QPainterPath(QtCore.QPointF(cx, cy))
        cone.lineTo(cx - 44, cy - 92)
        cone.arcTo(QtCore.QRectF(cx - 104, cy - 104, 208, 208), 90 + 25, -50)
        cone.closeSubpath()
        p.setPen(QtCore.Qt.NoPen)
        p.setBrush(theme.qcolor("accent", 26))
        p.drawPath(cone)
        p.setBrush(theme.qcolor("dim"))
        p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(cx, cy - 10), QtCore.QPointF(cx - 6, cy + 6), QtCore.QPointF(cx + 6, cy + 6)]))
        p.setPen(theme.qcolor("dim"))
        p.setFont(theme.font(10, spacing=1.2))
        p.drawText(QtCore.QRectF(12, 8, w - 24, 14), QtCore.Qt.AlignLeft, "MAP")
        p.setPen(theme.qcolor("faint"))
        p.setFont(theme.font(11))
        p.drawText(QtCore.QRectF(12, 8, w - 24, 14), QtCore.Qt.AlignRight, "waiting for scan")


class WorldStore:
    """The room as the car has seen it: LiDAR points in the world frame on a 5 cm grid, from the relay's scan-matched pose
    (a simple map that builds up while driving). Shared by the mini-map and the Map & Zones drawer.

    Each cell keeps a clamped hit score, as in an occupancy grid (Moravec & Elfes 1985; clamping as OctoMap, Hornung et al.
    2013): +1 for every scan that hits it, up to HIT_MAX. Visibility-based removal (Kim & Kim 2020, "Removert"): a cell the
    current scan sees past - the beam in its direction returns from more than MARGIN_M farther away - loses SEE_THROUGH and
    is dropped at zero, so points left by pose error or by things that moved disappear once the LiDAR looks there again.
    Cells the LiDAR cannot see now keep their score: closer than its minimum range, hidden behind a nearer return, or in a
    direction without any return."""

    HIT_MAX = 6              # clamp: a cell seen many times still goes after 3 scans that see past it
    SEE_THROUGH = 2
    MARGIN_M = 0.10          # 2 cells + pose noise: a return must be this much farther than the cell to count as "seen past"
    BIN_DEG = 1.0            # the scan as a range per 1 deg of bearing (the 1-D range image)
    MIN_RANGE_M = 0.20       # the A3 sees nothing closer (tuning mount.min_valid_range_m)
    MAX_RANGE_M = 8.0        # no removal further out (bearing bins grow to 14 cm there)

    def __init__(self, cell=0.05, max_cells=40000):
        self.cell, self.max = cell, max_cells
        self.keys = np.empty((0, 2), np.int64)
        self.score = np.empty(0)
        self.seen = np.empty(0, np.int64)     # the add() count when the cell was last hit (oldest go first past max_cells)
        self.index = {}
        self.n = 0
        self.version = 0                      # bumped whenever the cells change; views redraw only then
        self._cache = (-1, 2, np.empty((0, 2)))

    def add(self, st):
        self.n += 1
        if self.n % 3:
            return
        world = st.get("world")
        if not world or "pose" not in world:
            return
        polar = np.asarray(st.get("_pts") if st.get("_pts") is not None else [], float).reshape(-1, 2)
        polar = polar[polar[:, 1] >= self.MIN_RANGE_M]
        if not len(polar):
            return
        pose = world["pose"]
        hit = self._hit(np.round(vehicle_to_world(polar_xy(polar), pose) / self.cell).astype(np.int64))
        self._see_past(polar, pose, hit)
        if len(self.score) > self.max:
            self._keep(np.argsort(self.seen)[len(self.score) - self.max:])
        self.version += 1

    def _hit(self, keys):
        """+1 (clamped) for every cell this scan hits, new cells appended; returns their indices."""
        keys = np.unique(keys, axis=0)
        idx = np.fromiter((self.index.get(k, -1) for k in map(tuple, keys.tolist())), np.int64, len(keys))
        new = idx < 0
        if new.any():
            n0 = len(self.score)
            idx[new] = np.arange(n0, n0 + new.sum())
            self.keys = np.vstack([self.keys, keys[new]])
            self.score = np.concatenate([self.score, np.zeros(new.sum())])
            self.seen = np.concatenate([self.seen, np.zeros(new.sum(), np.int64)])
            self.index.update(zip(map(tuple, keys[new].tolist()), idx[new].tolist()))
        self.score[idx] = np.minimum(self.score[idx] + 1, self.HIT_MAX)
        self.seen[idx] = self.n
        return idx

    def _see_past(self, polar, pose, hit):
        """Lower the score of every cell the scan sees through (adas/visibility.py: only beams within the cell's own angular size
        count, weighed by range and by how many beams should hit it - a thin object far away is not judged); drop cells at zero."""
        if not len(self.score):
            return
        from adas.visibility import evidence, width_from_grid
        lx, ly = vehicle_to_world([[GEO["lidar_x"], 0.0]], pose)[0]
        w = self.keys * self.cell
        d = w - (lx, ly)
        r = np.hypot(d[:, 0], d[:, 1])
        near = r <= self.MAX_RANGE_M
        if not near.any():
            return
        cw = (math.degrees(pose[2]) - np.degrees(np.arctan2(d[near, 1], d[near, 0]))) % 360     # clockwise from ahead, like the scan
        _, miss = evidence(r[near], cw, polar[:, 1], polar[:, 0] % 360, min_range=self.MIN_RANGE_M,
                           half_width=width_from_grid(self.keys, self.cell)[near], max_beams=24)
        loss = np.zeros(len(self.score))
        loss[near] = self.SEE_THROUGH * miss
        loss[hit] = 0.0
        if loss.any():
            self.score = self.score - loss
            gone = self.score <= 0
            if gone.any():
                self._keep(np.flatnonzero(~gone))

    def _keep(self, idx):
        idx = np.sort(idx)
        self.keys, self.score, self.seen = self.keys[idx], self.score[idx], self.seen[idx]
        self.index = dict(zip(map(tuple, self.keys.tolist()), range(len(idx))))

    def points(self, min_hits=2):
        """Occupied cells as world points; cached per version (rebuilding 40 000 cells every frame was a main cost)."""
        if self._cache[0] == self.version and self._cache[1] == min_hits:
            return self._cache[2]
        out = self.keys[self.score >= min_hits] * self.cell
        out = out.astype(float) if len(out) else np.empty((0, 2))
        self._cache = (self.version, min_hits, out)
        return out

    def clear(self):
        self.keys, self.score, self.seen, self.index = np.empty((0, 2), np.int64), np.empty(0), np.empty(0, np.int64), {}
        self.version += 1


# ====================================================================== drawers
class MetricGrid(pg.GraphicsObject):
    """A metric floor grid drawn in one pass over the visible range (about 40 lines, no text). pyqtgraph's own axes rebuild
    their tick labels every time the view moves - with a following camera that was the largest single cost of the map (about
    8 ms a frame) - so the map hides them and uses this instead. `spacing_m` is the current line spacing (1, 2, 5 ... x 0.1 m
    chosen so lines stay at least 34 px apart); the map shows it as a scale caption."""

    def __init__(self):
        super().__init__()
        self.spacing_m = 0.5
        self.setZValue(-100)

    def boundingRect(self):
        vb = self.getViewBox()
        return vb.viewRect() if vb is not None else QtCore.QRectF()

    def viewRangeChanged(self):
        self.prepareGeometryChange()
        self.update()

    def paint(self, p, *_):
        vb = self.getViewBox()
        if vb is None:
            return
        r = vb.viewRect()
        px = vb.viewPixelSize()[0]
        target = 34.0 * px                                   # metres per 34 px
        step = 0.1
        for m in (1, 2, 5, 10, 20, 50, 100):
            step = 0.1 * m
            if step >= target:
                break
        self.spacing_m = step
        x0, x1 = math.floor(r.left() / step) * step, r.right()
        y0, y1 = math.floor(r.top() / step) * step, r.bottom()
        minor, major = [], []
        i = round(x0 / step)
        x = x0
        while x <= x1:
            (major if i % 5 == 0 else minor).append(QtCore.QLineF(x, r.top(), x, r.bottom()))
            x += step
            i += 1
        j = round(y0 / step)
        y = y0
        while y <= y1:
            (major if j % 5 == 0 else minor).append(QtCore.QLineF(r.left(), y, r.right(), y))
            y += step
            j += 1
        pen = QtGui.QPen(theme.qcolor("hair", 110), 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawLines(minor)
        pen = QtGui.QPen(theme.qcolor("hair2", 150), 1)
        pen.setCosmetic(True)
        p.setPen(pen)
        p.drawLines(major)


class ParkingLayer:
    """The parking scene on the map plot: road and sidewalk, the parked vehicles, the strip line, the ArUco marker and - in the bay room -
    every bay as a filled object (free green, occupied red, selected blue, hover a light preview). On the street (one continuous lane,
    no bays) the hover and the click are car-sized ghosts on the lane. Items are rebuilt only when the layout changes; selection and
    hover just recolour / move them."""

    def __init__(self, plot):
        self.plot, self.model, self.items, self.pads, self.labels = plot, None, [], {}, []
        self.label_base = []                         # (x, y) of every label in the simulator frame
        self.sel = self.hover = None
        self.align = (0.0, 0.0, 0.0)

    @staticmethod
    def _poly(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return QtGui.QPolygonF([QtCore.QPointF(-y, x) for x, y in xy])     # map convention: screen right = world -y, up = x

    def _add(self, item, z):
        item.setZValue(z)
        self.plot.addItem(item, ignoreBounds=True)
        self.items.append(item)
        return item

    def clear(self):
        for it in self.items + self.labels:
            self.plot.removeItem(it)
        self.items, self.labels, self.label_base, self.pads, self.model = [], [], [], {}, None

    def _polygon(self, xy, fill, edge, z, width=1.2):
        it = QtWidgets.QGraphicsPolygonItem(self._poly(xy))
        pen = QtGui.QPen(edge, width)
        pen.setCosmetic(True)
        it.setPen(pen)
        it.setBrush(QtGui.QBrush(fill))
        return self._add(it, z)

    def rebuild(self, model):
        from gui.parking import corners
        self.clear()
        self.model = model
        if model.road:
            self._polygon(model.road, theme.qcolor("bg1", 120), QtGui.QColor(0, 0, 0, 0), -60)
        if model.sidewalk:
            self._polygon(model.sidewalk, theme.qcolor("raised", 230), theme.qcolor("hair2", 160), -55)
        for sp in model.spaces:
            self.pads[sp["id"]] = self._polygon(corners(sp), QtGui.QColor(0, 0, 0, 0), QtGui.QColor(0, 0, 0, 0), -40)
            t = pg.TextItem(sp["id"], color=C["text2"], anchor=(0.5, 0.5))
            t.setFont(theme.font(11))
            t.setZValue(-38)
            self.plot.addItem(t, ignoreBounds=True)
            self.labels.append(t)
            self.label_base.append((sp["cx"], sp["cy"]))
        self.ghosts = {}
        if model.street:                                                    # street: a ghost for the hover and one for the click
            for key in ("hover", "sel"):
                self.ghosts[key] = self._polygon(np.zeros((4, 2)), QtGui.QColor(0, 0, 0, 0), QtGui.QColor(0, 0, 0, 0), -39)
        for v in model.vehicles:                                            # the parked vehicles: solid grey bodies
            self._polygon(corners(v), theme.qcolor("s3", 235), theme.qcolor("text", 200), -30, 1.5)
        line = pg.PlotCurveItem(-model.pts[:, 1], model.pts[:, 0], pen=pg.mkPen(C["accent_dim"], width=2, style=QtCore.Qt.DashLine))
        line.setZValue(-35)
        self.plot.addItem(line, ignoreBounds=True)
        self.items.append(line)
        kerb = model.layout.get("kerb")
        if kerb:                                                            # a drawn lane: its kerb as a solid line
            k = np.asarray(kerb, float)
            kl = pg.PlotCurveItem(-k[:, 1], k[:, 0], pen=pg.mkPen(C["text2"], width=3))
            kl.setZValue(-34)
            self.plot.addItem(kl, ignoreBounds=True)
            self.items.append(kl)
        m = model.marker
        if m:                                                               # the ArUco marker on the rear wall, seen from above
            u = np.array([-math.sin(m["yaw"]), math.cos(m["yaw"])])
            a, b = np.array([m["x"], m["y"]]) - u * m["size"] / 2, np.array([m["x"], m["y"]]) + u * m["size"] / 2
            ln = pg.PlotCurveItem([-a[1], -b[1]], [a[0], b[0]], pen=pg.mkPen(C["text"], width=5))
            ln.setZValue(-28)
            self.plot.addItem(ln, ignoreBounds=True)
            self.items.append(ln)
            t = pg.TextItem(f"ArUco {m['id']}", color=C["text"], anchor=(0.5, 1.2))
            t.setFont(theme.font(11))
            self.plot.addItem(t, ignoreBounds=True)
            self.labels.append(t)
            self.label_base.append((m["x"], m["y"]))
        self.set_align(*self.align, force=True)
        self.paint_state(self.sel, self.hover)

    def set_align(self, dth, tx, ty, force=False):
        """Put the layout (simulator frame) into the relay's estimated world frame: rotate by dth, move by (tx, ty). Items get a
        QTransform (screen convention: right = -y, up = x, so the rotation is unchanged and the shift is (-ty, tx)); the text
        labels ignore transforms and are placed by hand."""
        if not force and max(abs(dth - self.align[0]), abs(tx - self.align[1]), abs(ty - self.align[2])) < 1e-4:
            return
        self.align = (dth, tx, ty)
        tf = QtGui.QTransform()
        tf.translate(-ty, tx)
        tf.rotate(math.degrees(dth))
        for it in self.items:
            it.setTransform(tf)
        c, sn = math.cos(dth), math.sin(dth)
        for t, (x, y) in zip(self.labels, self.label_base):
            t.setPos(-(sn * x + c * y + ty), c * x - sn * y + tx)

    def paint_state(self, sel, hover):
        self.sel, self.hover = sel, hover
        if self.model is None:
            return
        if self.model.street:
            from gui.parking import corners
            for key, s, fill, edge, w in (("hover", hover if hover != sel else None, theme.qcolor("text", 40), theme.qcolor("text", 190), 1.6),
                                          ("sel", sel, theme.qcolor("accent", 110), theme.qcolor("accent"), 2.6)):
                it = self.ghosts.get(key)
                if it is None:
                    continue
                it.setVisible(s is not None)
                if s is not None:
                    it.setPolygon(self._poly(corners(self.model.ghost(s))))
                    pen = QtGui.QPen(edge, w)
                    pen.setCosmetic(True)
                    if key == "hover":
                        pen.setStyle(QtCore.Qt.DashLine)
                    it.setPen(pen)
                    it.setBrush(QtGui.QBrush(fill))
            return
        for sp in self.model.spaces:
            it = self.pads[sp["id"]]
            if sp["id"] == sel:
                fill, edge, w = theme.qcolor("accent", 150), theme.qcolor("accent"), 3.0
            elif sp["id"] == hover:
                fill, edge, w = theme.qcolor("text", 50), theme.qcolor("text", 200), 2.0
            elif sp["occupied"]:
                fill, edge, w = theme.qcolor("bad", 85), theme.qcolor("bad", 190), 1.2
            else:
                fill, edge, w = theme.qcolor("ok", 80), theme.qcolor("ok", 190), 1.2
            pen = QtGui.QPen(edge, w)
            pen.setCosmetic(True)
            if sp["id"] == hover and sp["id"] != sel:
                pen.setStyle(QtCore.Qt.DashLine)
            it.setPen(pen)
            it.setBrush(QtGui.QBrush(fill))


class MapZonesPanel(QtWidgets.QWidget):
    """World map (builds up while driving), click-to-go with a heading, the speed-limit zone editor and, in the simulator,
    the Parking Simulator: room switch, clickable parking strip, occupancy."""

    parking_selected = QtCore.Signal(dict)         # {clicked_world_pose, selected_space_id, occupancy} - see interfaces.md 16b
    parking_changed = QtCore.Signal()              # selection or hover changed: redraw the 3D parking objects

    def __init__(self, link, store, on_event):
        super().__init__()
        self.link, self.store, self.on_event = link, store, on_event
        self.zones = self._load()
        self.pose = (0.0, 0.0, 0.0)
        self._press, self._poly = None, []
        self._sent_n = None
        self._map_version, self._cam, self._grid_step = -1, None, None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 14)
        lay.setSpacing(8)
        lay.addWidget(section_label("Map tool"))
        tools = QtWidgets.QGridLayout()
        tools.setSpacing(8)
        self.mode = {}
        for i, (key, glyph, text, tip) in enumerate((
                ("goto", "\u2316", "Go to", "click = drive there; press-drag-release = arrive facing that way"),
                ("park", "\u24c5", "Park", "click a spot on the parking lane - the car finds the nearest gap it fits into"),
                ("lane", "\u25a4", "Parking lane", "drag a rectangle along the parked cars: its far side from the car is the kerb"),
                ("erase", "\u2715", "Erase", "click a zone to delete it"),
                ("rect", "\u25ad", "Rectangle", "drag a rectangle"), ("circle", "\u25ef", "Circle", "drag from the centre"),
                ("poly", "\u2b20", "Polygon", "click the corners, right-click to finish"))):
            t = FeatureTile(glyph, text, tip, state=False, compact=True)
            self.mode[key] = t
            tools.addWidget(t, i // 4, i % 4)
        self._group = TileGroup(list(self.mode.values()))
        self.mode["goto"].setChecked(True)
        lay.addLayout(tools)
        lim = Card(padding=12)
        row = QtWidgets.QHBoxLayout()
        lab = QtWidgets.QVBoxLayout()
        lab.setSpacing(0)
        t1 = QtWidgets.QLabel("Speed limit for new zones")
        t1.setFont(theme.semibold(13))
        self.kph_car = QtWidgets.QLabel("")
        self.kph_car.setFont(theme.font(11))
        self.kph_car.setStyleSheet(f"color: {C['dim']};")
        lab.addWidget(t1)
        lab.addWidget(self.kph_car)
        row.addLayout(lab, 1)
        self.kph = QtWidgets.QSpinBox()
        self.kph.setRange(5, 120)
        self.kph.setValue(20)
        self.kph.setSuffix(" km/h")
        self.kph.setFont(theme.light(20))
        self.kph.setMinimumWidth(130)
        self.kph.valueChanged.connect(lambda v: self.kph_car.setText(f"= {v / 3.6 / CAR_SCALE:.2f} m/s on the car (1:{CAR_SCALE:.0f})"))
        self.kph.valueChanged.emit(20)
        row.addWidget(self.kph)
        lim.body.addLayout(row)
        lay.addWidget(lim)
        self.parking, self.park_sel, self.park_hover, self.park_world = None, None, None, "room"
        self.park_align = (0.0, 0.0, 0.0)             # simulator frame -> relay frame (dth rad, tx m, ty m)
        self.park_head = section_label("Parking simulator")
        lay.addWidget(self.park_head)
        self.park_card = Card(padding=12)
        self.park_seg = Segmented((("room", "ROOM"), ("park_perp", "BAYS"), ("park_street", "STREET")))
        self.park_seg.changed.connect(self._switch_world)
        self.park_card.body.addWidget(self.park_seg)
        self.park_desc = QtWidgets.QLabel("")
        self.park_desc.setFont(theme.font(12))
        self.park_desc.setStyleSheet(f"color: {C['dim']};")
        self.park_desc.setWordWrap(True)
        self.park_card.body.addWidget(self.park_desc)
        self.marker_row = QtWidgets.QWidget()
        mrow = QtWidgets.QHBoxLayout(self.marker_row)
        mrow.setContentsMargins(0, 0, 0, 0)
        mlab = QtWidgets.QLabel("ArUco marker ID")
        mlab.setFont(theme.semibold(13))
        mrow.addWidget(mlab, 1)
        self.marker_id = QtWidgets.QSpinBox()
        self.marker_id.setRange(0, 49)
        self.marker_id.setValue(23)
        self.marker_id.setMinimumWidth(90)
        self.marker_timer = QtCore.QTimer(self)
        self.marker_timer.setSingleShot(True)
        self.marker_timer.timeout.connect(lambda: self._switch_world("park_perp"))
        self.marker_id.valueChanged.connect(lambda _: self.marker_timer.start(450))
        mrow.addWidget(self.marker_id)
        self.park_card.body.addWidget(self.marker_row)
        self.park_go = QtWidgets.QPushButton("Park at the marker")
        self.park_go.setToolTip("reverse into the bay of the ArUco marker the rear camera sees (hold the throttle as the dead-man switch)")
        self.park_go.clicked.connect(self._park_marker)
        self.park_card.body.addWidget(self.park_go)
        from gui.controls import LegendPill
        self.park_legend = LegendPill([("free", C["ok"]), ("occupied", C["bad"]), ("target", C["accent"])])
        self.park_card.body.addWidget(self.park_legend, 0, QtCore.Qt.AlignLeft)
        self.park_info = QtWidgets.QLabel("")
        self.park_info.setFont(theme.font(12))
        self.park_info.setStyleSheet(f"color: {C['text2']};")
        self.park_info.setWordWrap(True)
        self.park_card.body.addWidget(self.park_info)
        lay.addWidget(self.park_card)
        self.park_head.hide()
        self.park_card.hide()
        lay.addWidget(section_label("Parking lane"))
        self.lane_card = Card(padding=12)
        self.lane_info = QtWidgets.QLabel("")
        self.lane_info.setFont(theme.font(12))
        self.lane_info.setStyleSheet(f"color: {C['text2']};")
        self.lane_info.setWordWrap(True)
        self.lane_card.body.addWidget(self.lane_info)
        lrow = QtWidgets.QGridLayout()
        lrow.setSpacing(8)
        self.lane_flip = FeatureTile("\u21c5", "Flip kerb side", "the kerb is the other long side of the lane", state=False,
                                     momentary=True, compact=True)
        self.lane_flip.clicked.connect(self.flip_lane)
        self.lane_del = FeatureTile("\u2300", "Delete lane", "back to the simulator street's own lane, if any", state=False,
                                    momentary=True, compact=True, danger=True)
        self.lane_del.clicked.connect(lambda: self.set_lane(None))
        lrow.addWidget(self.lane_flip, 0, 0)
        lrow.addWidget(self.lane_del, 0, 1)
        self.lane_card.body.addLayout(lrow)
        lay.addWidget(self.lane_card)
        self.lane = self._load_lane()
        self.hint = QtWidgets.QLabel("")
        self.hint.setFont(theme.font(12))
        self.hint.setStyleSheet(f"color: {C['dim']}; padding: 0 2px;")
        self.hint.setWordWrap(True)
        self.hint.setMinimumHeight(36)
        lay.addWidget(self.hint)
        frame = Card(padding=6)
        self.plot = theme.style_plot(pg.PlotWidget())
        self.plot.setMinimumHeight(300)
        self.plot.setAspectLocked(True)
        self.plot.getPlotItem().hideButtons()
        for ax in ("left", "bottom"):
            self.plot.hideAxis(ax)                                   # their tick labels cost ~8 ms a frame when following
        self.grid = MetricGrid()
        self.plot.addItem(self.grid, ignoreBounds=True)
        self.scale_lbl = QtWidgets.QLabel("", self.plot)
        self.scale_lbl.setFont(theme.font(11))
        self.scale_lbl.setStyleSheet(f"color: {C['dim']}; background: transparent;")
        self.scale_lbl.setMinimumHeight(16)                          # (a set minimum keeps the drawer's word-wrap pass off it)
        self.scale_lbl.move(10, 8)
        self.plot.getPlotItem().vb.setMouseEnabled(x=False, y=False)      # zoom / pan are handled in eventFilter
        self.plot.setXRange(-3, 3)
        self.plot.setYRange(-2, 4)
        self.plot.setBackground(C["bg1"])
        self.map_pts = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(C["dim"]), pen=None)
        self.live = pg.ScatterPlotItem(size=3, brush=pg.mkBrush(C["accent"]), pen=None)
        self.path = pg.PlotCurveItem(pen=pg.mkPen(C["accent"], width=4))
        self.car = pg.PlotCurveItem(pen=pg.mkPen(C["text"], width=2))
        self.draft = pg.PlotCurveItem(pen=pg.mkPen(C["warn"], width=2, style=QtCore.Qt.DashLine))
        self.slot = pg.PlotCurveItem(pen=pg.mkPen(C["accent"], width=3))        # the parking slot the car chose (relay's nav.slot)
        for it in (self.map_pts, self.live, self.path, self.car, self.draft, self.slot):
            self.plot.addItem(it)
        self.zone_items = []
        self.park_layer = ParkingLayer(self.plot)
        frame.body.addWidget(self.plot)
        lay.addWidget(frame, 1)
        vrow = QtWidgets.QGridLayout()
        vrow.setSpacing(8)
        for i, (glyph, text, fn) in enumerate((("\uff0b", "Zoom in", lambda: self.zoom(1.4)),
                                               ("\uff0d", "Zoom out", lambda: self.zoom(1 / 1.4)),
                                               ("\u26f6", "Fit all", self.fit))):
            t = FeatureTile(glyph, text, state=False, momentary=True, compact=True)
            t.clicked.connect(fn)
            vrow.addWidget(t, 0, i)
        self.follow_btn = FeatureTile("\u25ce", "Follow car", "keeps the car in the middle of the map", compact=True)
        self.follow_btn.setChecked(True)
        vrow.addWidget(self.follow_btn, 0, 3)
        lay.addLayout(vrow)
        tip = QtWidgets.QLabel("Ctrl + mouse wheel = zoom at the cursor \u00b7 middle-drag or Shift + drag = pan \u00b7 the plain wheel scrolls this "
                               "panel \u00b7 Fit all frames the map, the zones and the car")
        tip.setWordWrap(True)
        tip.setFont(theme.font(11))
        tip.setStyleSheet(f"color: {C['dim']};")
        lay.addWidget(tip)
        lay.addWidget(section_label("Drive"))
        g1 = QtWidgets.QGridLayout()
        g1.setSpacing(8)
        for i, (glyph, text, fn, danger) in enumerate((
                ("\u2302", "Return to start", lambda: (self.link.send("HOME"), self.on_event("return to start")), False),
                ("\u25ce", "Centre on car", self.centre, False),
                ("\u2b58", "Cancel autonomy", lambda: self.link.send("GOTO CANCEL"), True))):
            t = FeatureTile(glyph, text, state=False, momentary=True, compact=True, danger=danger)
            t.clicked.connect(fn)
            g1.addWidget(t, 0, i)
        lay.addLayout(g1)
        lay.addWidget(section_label("Autonomy"))
        g3 = QtWidgets.QGridLayout()
        g3.setSpacing(8)
        self.explore_tile = FeatureTile("\u2734", "Explore the room", "the car maps the room by itself, frontier by frontier "
                                        "(hold the throttle as the dead-man switch)", compact=True)
        self.explore_tile.toggled.connect(lambda on: (self.link.send("EXPLORE ON" if on else "EXPLORE OFF"),
                                                      self.on_event("exploring" if on else "exploration stopped")))
        park = FeatureTile("\u24c5", "Auto-park", "finds a bay beside the car and backs into it (hold the throttle)",
                           state=False, momentary=True, compact=True)
        park.clicked.connect(lambda: (self.link.send("PARK"), self.on_event("auto-park requested")))
        g3.addWidget(self.explore_tile, 0, 0)
        g3.addWidget(park, 0, 1)
        lay.addLayout(g3)
        lay.addWidget(section_label("Map and zones"))
        g2 = QtWidgets.QGridLayout()
        g2.setSpacing(8)
        for i, (glyph, text, fn) in enumerate((
                ("\u21e7", "Send zones", self.send_zones), ("\u2300", "Delete zones", self.clear_zones),
                ("\u2299", "Reset origin", self.reset_origin), ("\u267b", "Clear map", self.store.clear))):
            t = FeatureTile(glyph, text, state=False, momentary=True, compact=True)
            t.clicked.connect(fn)
            g2.addWidget(t, i // 2, i % 2)
        lay.addLayout(g2)
        lay.addWidget(section_label("Speed-limit zones"))
        self.zone_box = QtWidgets.QVBoxLayout()
        self.zone_box.setSpacing(6)
        lay.addLayout(self.zone_box)
        self.zone_empty = QtWidgets.QLabel("No zones yet. Pick Rectangle, Circle or Polygon above and draw on the map; "
                                           "where there is no zone the car may go at full speed.")
        self.zone_empty.setWordWrap(True)
        self.zone_empty.setFont(theme.font(12))
        self.zone_empty.setStyleSheet(f"color: {C['dim']};")
        lay.addWidget(self.zone_empty)
        for key, b in self.mode.items():
            b.toggled.connect(self._update_hint)
        self._update_hint()
        self.plot.scene().installEventFilter(self)
        self._wheel = WheelToScroll(self.plot, self._wheel_zoom)      # plain wheel scrolls the drawer, Ctrl + wheel zooms
        self._draw_zones()
        self._show_lane()

    # --- the drawn parking lane (relay / map frame), on disk like the zones
    def _load_lane(self):
        try:
            lane = json.load(open(LANE_FILE))
            return lane if lane and all(k in lane for k in ("x0", "y0", "x1", "y1", "road")) else None
        except Exception:
            return None

    def set_lane(self, lane, note=True):
        """The parking lane (None = none): saved, shown in the card; EVWindow turns it into the parking layout of the Park tool."""
        self.lane = lane
        try:
            if lane is None:
                if os.path.exists(LANE_FILE):
                    os.remove(LANE_FILE)
            else:
                json.dump(lane, open(LANE_FILE, "w"))
        except OSError:
            pass
        self._show_lane()
        if note:
            if lane is None:
                self.on_event("parking lane deleted")
            else:
                from gui.parking import lane_geometry
                _, _, depth, _, _ = lane_geometry(lane)
                length = max(abs(lane["x1"] - lane["x0"]), abs(lane["y1"] - lane["y0"]))
                self.on_event(f"parking lane {length:.2f} m x {100 * depth:.0f} cm drawn - pick Park and click where to park")
        self.parking_changed.emit()

    def flip_lane(self):
        if self.lane is not None:
            from gui.parking import flip_lane
            self.set_lane(flip_lane(self.lane), note=False)
            self.on_event("parking lane: the kerb is the other side now")

    def _show_lane(self):
        lane = self.lane
        self.lane_flip.setEnabled(lane is not None)
        self.lane_del.setEnabled(lane is not None)
        if lane is None:
            self.lane_info.setText("No lane drawn. Pick Parking lane and drag a rectangle along the parked cars (or boxes), at least "
                                   "30 cm deep: the long side nearer the car is the road edge, the far side the kerb. Then pick Park and "
                                   "click where to park - in any simulator room or on the car.")
            return
        from gui.parking import lane_geometry
        _, _, depth, _, _ = lane_geometry(lane)
        length = max(abs(lane["x1"] - lane["x0"]), abs(lane["y1"] - lane["y0"]))
        warn = " - too shallow for the car (needs about 30 cm)" if depth < 0.28 else ""
        self.lane_info.setText(f"Lane {length:.2f} m long, {100 * depth:.0f} cm deep, kerb on the "
                               f"{'far' if self._kerb_is_far(lane) else 'near'} side from the car{warn}. Pick Park and click where to "
                               f"park; it replaces the simulator street's own lane while it exists.")

    def _kerb_is_far(self, lane):
        from gui.parking import make_lane
        a, b = (lane["x0"], lane["y0"]), (lane["x1"], lane["y1"])
        return make_lane(a, b, self.pose[:2])["road"] == lane["road"]

    def user_lane_layout(self):
        """The drawn lane as a parking layout in the simulator frame (moved by the current alignment), or None."""
        if self.lane is None:
            return None
        from gui.parking import lane_layout
        return lane_layout(self.lane, self.park_align)

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
                           "lane": "Drag a rectangle along the parked cars, as deep as the lane (at least 30 cm): the long side nearer the "
                                   "car is the road edge, the far side the kerb (flip it in the Parking lane card). Then pick Park.",
                           "circle": "Press at the centre and drag out the radius.",
                           "poly": "Click the corners; right-click to close the shape.",
                           "erase": "Click inside a zone to delete it.",
                           "park": "Click a spot on the parking lane - the simulator street's or one you drew - and hold the "
                                   "throttle to drive. The car looks for the "
                                   "nearest gap between the parked cars it fits into, drives there and parks; the slot it chose is "
                                   "drawn in blue. Bays: the nearest free bay is highlighted. Right-click clears."}[self.current_mode()])

    # --- coordinates: screen up = world x, screen right = world -y (as the car's own view)
    def _world(self, scene_pos):
        p = self.plot.getPlotItem().vb.mapSceneToView(scene_pos)
        return float(p.y()), float(-p.x())

    @staticmethod
    def _sc(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return -xy[:, 1], xy[:, 0]

    def zoom(self, factor, centre=None):
        vb = self.plot.getPlotItem().vb
        c = vb.viewRect().center() if centre is None else centre
        vb.scaleBy((1 / factor, 1 / factor), center=c)
        self.follow_btn.setChecked(False)

    def _wheel_zoom(self, ev):
        vb = self.plot.getPlotItem().vb
        pos = self.plot.mapToScene(ev.position().toPoint())
        self.zoom(1.25 if ev.angleDelta().y() > 0 else 0.8, vb.mapSceneToView(pos))

    def fit(self):
        """Frame the map points, the zones and the car."""
        pts = [self._sc(self.store.points())] if len(self.store.points()) else []
        for z in self.zones:
            out = zone_outline(z)
            if len(out) >= 3:
                pts.append(self._sc(out))
        x, y, _ = self.pose
        pts.append(self._sc(np.array([[x, y]])))
        if self.parking is not None:
            from gui.parking import corners
            dth, tx, ty = self.park_align
            c, sn = math.cos(dth), math.sin(dth)

            def to_relay(xy):
                xy = np.asarray(xy, float).reshape(-1, 2)
                return np.column_stack([c * xy[:, 0] - sn * xy[:, 1] + tx, sn * xy[:, 0] + c * xy[:, 1] + ty])
            for sp in self.parking.spaces + self.parking.vehicles:
                pts.append(self._sc(to_relay(corners(sp))))
            if self.parking.road:
                pts.append(self._sc(to_relay(self.parking.road)))
        xs = np.concatenate([np.asarray(p[0], float) for p in pts])
        ys = np.concatenate([np.asarray(p[1], float) for p in pts])
        pad = 0.6
        self.plot.setXRange(xs.min() - pad, xs.max() + pad)
        self.plot.setYRange(ys.min() - pad, ys.max() + pad)

    def eventFilter(self, obj, ev):
        t = ev.type()
        vb = self.plot.getPlotItem().vb
        if t == QtCore.QEvent.GraphicsSceneMousePress and (ev.button() == QtCore.Qt.MiddleButton or
                                                          (ev.button() == QtCore.Qt.LeftButton and
                                                           ev.modifiers() & QtCore.Qt.ShiftModifier)):
            self._pan = ev.scenePos()
            self.follow_btn.setChecked(False)
            return True
        if t == QtCore.QEvent.GraphicsSceneMouseMove and getattr(self, "_pan", None) is not None:
            a, b = vb.mapSceneToView(self._pan), vb.mapSceneToView(ev.scenePos())
            vb.translateBy(x=-(b.x() - a.x()), y=-(b.y() - a.y()))
            self._pan = ev.scenePos()
            return True
        if t == QtCore.QEvent.GraphicsSceneMouseRelease and getattr(self, "_pan", None) is not None:
            self._pan = None
            return True
        if self.current_mode() == "park":
            if t == QtCore.QEvent.GraphicsSceneMouseMove:
                self._park_hover(self._world(ev.scenePos()))
                return False
            if t == QtCore.QEvent.GraphicsSceneMousePress:
                if ev.button() == QtCore.Qt.RightButton:
                    self._park_clear()
                else:
                    self._park_click(self._world(ev.scenePos()))
                return True
            if t == QtCore.QEvent.GraphicsSceneMouseRelease:
                return True
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
            if mode in ("rect", "lane"):
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
            elif mode == "lane":
                if max(abs(w[0] - x0), abs(w[1] - y0)) >= 0.5 and min(abs(w[0] - x0), abs(w[1] - y0)) >= 0.15:
                    from gui.parking import make_lane
                    self.set_lane(make_lane((x0, y0), w, self.pose[:2]))
                else:
                    self.on_event("parking lane: drag a longer rectangle (at least 0.5 m long and 15 cm deep)")
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

    def _delete_zone(self, i):
        if 0 <= i < len(self.zones):
            del self.zones[i]
            self._changed()

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
        if self.lane is not None:
            self.set_lane(None, note=False)                  # (drawn in the old world frame: it would point somewhere else now)
        self.on_event("world origin reset at the car's position - zones are relative to this")

    def centre(self):
        x, y, _ = self.pose
        vb = self.plot.getPlotItem().vb
        r = vb.viewRect()
        vb.translateBy(x=-y - r.center().x(), y=x - r.center().y())

    def _draw_zones(self):
        for it in self.zone_items:
            self.plot.removeItem(it)
        self.zone_items = []
        while self.zone_box.count():
            w = self.zone_box.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self.zone_empty.setVisible(not self.zones)
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
            hexc = "#%02x%02x%02x" % (r, g, b)
            shape = {"rect": "Rectangle", "circle": "Circle", "poly": "Polygon"}.get(z["kind"], z["kind"])
            row = ZoneRow(hexc, f"{shape} \u00b7 {z['kph']:.0f} km/h", f"{z['kph'] / 3.6 / CAR_SCALE:.2f} m/s on the car")
            row.delete.connect(lambda i=i: self._delete_zone(i))
            self.zone_box.addWidget(row)

    # --- Parking Simulator
    PARK_DESC = {"room": "The plain simulator room (no parking objects).",
                 "park_perp": "Five perpendicular bays along the rear wall, two occupied; an ArUco marker marks the target bay.",
                 "park_street": "A street: one parking lane along the kerb, cars parked at different spacing - no marked spaces. "
                                "Click where you want to park; the car finds a gap it fits into."}

    def _switch_world(self, name):
        """Ask the simulator for another room (POST /api/sim/world/<name>[/<marker id>]), then re-origin the relay's world frame
        at the new start pose and forget the old map."""
        self.park_world = name
        self.park_seg.select(name)
        path = f"/api/sim/world/{name}" + (f"/{self.marker_id.value()}" if name == "park_perp" else "")
        self.park_sel = self.park_hover = None

        def work():
            self.link.http_post(path)
            time.sleep(0.5)
            self.link.send("ORIGIN")
        threading.Thread(target=work, daemon=True).start()
        self.store.clear()
        self._cam = None
        if self.lane is not None:
            self.set_lane(None, note=False)                  # (a lane drawn in the old room)
        self.on_event(f"simulator room: {name}" + (f" (marker {self.marker_id.value()})" if name == "park_perp" else ""))

    def set_parking(self, model, sim_world=None):
        """The layout of the newest packet (None outside the Parking Simulator): shows / hides the card, rebuilds the map layer
        when the layout changed, keeps a selection only while that space is still free."""
        seen = self.parking.signature if self.parking else None
        self.parking = model
        show = sim_world is not None
        self.park_head.setVisible(show)
        self.park_card.setVisible(show)
        if sim_world is not None and sim_world != self.park_world:
            self.park_world = sim_world
            self.park_seg.select(sim_world if sim_world in self.park_seg.btns else "room")
        self.park_desc.setText(self.PARK_DESC.get(self.park_world, ""))
        self.marker_row.setVisible(self.park_world == "park_perp")
        self.park_go.setVisible(self.park_world == "park_perp")
        self.park_legend.setVisible(model is not None and not model.street)
        self.plot.setMinimumHeight(520 if model is not None else 300)
        if model is None:
            if seen is not None:
                self.park_layer.clear()
            self.park_sel = self.park_hover = None
            return
        if model.signature != seen:
            if not model.street and (self.park_sel not in model.by_id or model.by_id[self.park_sel]["occupied"]):
                self.park_sel = None
            self.park_layer.rebuild(model)
            self.park_layer.paint_state(self.park_sel, self.park_hover)
            self._update_park_info()
            QtCore.QTimer.singleShot(200, self.fit)

    def set_park_align(self, align):
        self.park_align = align
        self.park_layer.set_align(*align)

    def _to_sim(self, w):
        """A click in the relay's frame -> the simulator frame the layout is stored in."""
        dth, tx, ty = self.park_align
        c, sn = math.cos(dth), math.sin(dth)
        dx, dy = w[0] - tx, w[1] - ty
        return (c * dx + sn * dy, -sn * dx + c * dy)

    def _update_park_info(self, preview=None):
        m = self.parking
        if m is None:
            return
        if m.street and m.user_lane:
            txt = "Your parking lane \u00b7 click where you want to park; the car finds the parked cars with its LiDAR"
            if self.park_sel is not None:
                txt += f" \u00b7 clicked at {self.park_sel:.2f} m"
            self.park_info.setText(txt)
            self.lane_info.setText(self.lane_info.text().split(" \u00b7 clicked")[0] +
                                   (f" \u00b7 clicked at {self.park_sel:.2f} m" if self.park_sel is not None else ""))
            return
        if m.street:
            txt = f"{len(m.vehicles)} cars parked along the lane \u00b7 click where you want to park"
            if self.park_sel is not None:
                txt += f" \u00b7 clicked at {self.park_sel:.2f} m"
            self.park_info.setText(txt)
            return
        free = sum(1 for sp in m.spaces if not sp["occupied"])
        txt = f"{free} of {len(m.spaces)} spaces free"
        sp = m.by_id.get(self.park_sel)
        if sp:
            txt += f" \u00b7 target {sp['id']} ({100 * sp['l']:.0f} x {100 * sp['w']:.0f} cm)"
        if preview and preview != self.park_sel:
            txt += f" \u00b7 hover: {preview}"
        self.park_info.setText(txt)

    def _park_hover(self, w):
        m = self.parking
        if m is None:
            return
        ws = self._to_sim(w)
        if m.street:
            hid = round(m.project(ws)[0], 2) if m.in_strip(ws) else None
        else:
            sp = m.nearest_free(ws) if m.in_strip(ws) else None
            hid = sp["id"] if sp else None
        if hid != self.park_hover:
            self.park_hover = hid
            self.park_layer.paint_state(self.park_sel, hid)
            self._update_park_info(hid)
            self.parking_changed.emit()

    def _park_click(self, w):
        m = self.parking
        if m is None:
            self.on_event("parking: no parking layout - choose BAYS or STREET in the simulator")
            return
        msg = m.click(self._to_sim(w))
        if msg is None:
            self.on_event("parking: that click is not on the parking strip")
            return
        # published in the relay's world frame (what the map and the planner use): the click as made, the strip heading turned by
        # the alignment
        msg["clicked_world_pose"] = [round(float(w[0]), 3), round(float(w[1]), 3),
                                     round(math.remainder(msg["clicked_world_pose"][2] + self.park_align[0], 2 * math.pi), 4)]
        self.park_sel = msg["s"] if m.street else msg["selected_space_id"]
        self.park_layer.paint_state(self.park_sel, self.park_hover)
        self._update_park_info()
        self.parking_selected.emit(msg)
        self.parking_changed.emit()
        if m.street:
            self._park_target(msg)
        else:
            self.on_event(f"parking: bay {self.park_sel} selected" if self.park_sel else "parking: every bay is occupied")

    def _park_target(self, msg):
        """The click on the parking strip -> the relay's PARK_TARGET (docs/interfaces.md 15-16b): strip, spaces and the click in the
        world frame; the car plans and drives the manoeuvre while the throttle is held (the HMI itself does not plan)."""
        from gui.parking import park_target_command
        cmd = park_target_command(self.parking.layout, msg, self.park_align)
        self.on_event(f"parking: looking for a gap near {msg['s']:.2f} m along the lane - hold the throttle to drive")
        threading.Thread(target=self.link.send, args=(cmd,), daemon=True).start()

    def _park_marker(self):
        self.on_event("parking: reversing into the bay of the marker - hold the throttle to drive")
        threading.Thread(target=self.link.send, args=("PARK_MARKER",), daemon=True).start()

    def _park_clear(self):
        self.park_sel = None
        self.park_layer.paint_state(None, self.park_hover)
        self._update_park_info()
        self.parking_selected.emit({"clicked_world_pose": None, "selected_space_id": None,
                                    "occupancy": {s["id"]: s["occupied"] for s in self.parking.spaces} if self.parking else {}})
        self.parking_changed.emit()

    def show_state(self, st):
        world = st.get("world") or {}
        self.pose = tuple(world.get("pose") or (0.0, 0.0, 0.0))
        if world and len(world.get("zones") or []) != len(self.zones) and self._sent_n != len(self.zones):
            self._sent_n = len(self.zones)             # the relay restarted without our zones: send them again
            self.send_zones()
        if not self.isVisible():
            return
        if self._map_version != self.store.version:                  # the map cells only change now and then
            self._map_version = self.store.version
            wp = self.store.points()
            self.map_pts.setData(*self._sc(wp)) if len(wp) else self.map_pts.setData([], [])
        v = polar_xy(st.get("_pts"))
        self.live.setData(*self._sc(vehicle_to_world(v, self.pose))) if len(v) else self.live.setData([], [])
        man = polar_xy((st.get("plan") or {}).get("maneuver"))
        self.path.setData(*self._sc(vehicle_to_world(man, self.pose))) if len(man) >= 2 else self.path.setData([], [])
        slot = polar_xy((st.get("nav") or {}).get("slot"))
        if len(slot) >= 4:
            ring = vehicle_to_world(np.vstack([slot[:4], slot[:1]]), self.pose)
            self.slot.setData(*self._sc(ring))
        else:
            self.slot.setData([], [])

    def animate(self, pose, dt):
        """Every frame: the car outline at the interpolated pose and, when following, a camera that eases towards it (so
        neither the car nor the view steps at the 20 Hz packet rate)."""
        if not self.isVisible():
            self._cam = None
            return
        hw = GEO["width"] / 2
        body = np.array([[GEO["rear"], -hw], [GEO["front"], -hw], [GEO["front"] + 0.05, 0], [GEO["front"], hw],
                         [GEO["rear"], hw], [GEO["rear"], -hw]])
        self.car.setData(*self._sc(vehicle_to_world(body, pose)))
        step = self.grid.spacing_m
        if step != self._grid_step:
            self._grid_step = step
            self.scale_lbl.setText(f"grid {step:g} m")
            self.scale_lbl.adjustSize()
        if not self.follow_btn.isChecked():
            self._cam = None
            return
        cx, cy = -pose[1], pose[0]
        vb = self.plot.getPlotItem().vb
        r = vb.viewRect()
        if self._cam is None:
            self._cam = (Smooth(r.center().x(), 0.10), Smooth(r.center().y(), 0.10))
        sx, sy = self._cam
        vb.translateBy(x=sx.step(dt, cx) - r.center().x(), y=sy.step(dt, cy) - r.center().y())


class RearCameraPanel(QtWidgets.QWidget):
    """Reverse-camera view: the rear webcam (pi/rear_camera.py, MJPEG on :8091) with the steering-dependent reverse
    guidelines drawn on the floor, objects the vision pipeline found (with looming time to contact) and its status."""


    def __init__(self, host, port=8091):
        super().__init__()
        self.url, self.state_url = f"http://{host}:{port}/stream", f"http://{host}:{port}/state"
        self.kappa, self.running, self.have = 0.0, False, False
        self.hazards, self.speed = None, 0.0
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 14)
        lay.setSpacing(8)
        self.card = Card(padding=6)
        self.view = QtWidgets.QLabel()
        self.view.setMinimumSize(320, 240)
        self.view.setAlignment(QtCore.Qt.AlignCenter)
        self.view.setStyleSheet(f"background: {C['bg']}; border-radius: 10px; color: {C['dim']};")
        self.placeholder()
        self.card.body.addWidget(self.view)
        lay.addWidget(self.card, 1)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setFont(theme.font(12))
        self.status.setStyleSheet(f"color: {C['text2']};")
        lay.addWidget(self.status)
        lay.addWidget(section_label("View"))
        row = QtWidgets.QGridLayout()
        row.setSpacing(8)
        self.guides = FeatureTile("\u2261", "Reverse guidelines", "steering-dependent wheel tracks and distance bars", compact=True)
        self.guides.setChecked(True)
        self.ghost = FeatureTile("\u25a3", "Ghost car and assist", "where the car will be if you keep reversing; objects and puddles in the path",
                                 compact=True)
        self.ghost.setChecked(True)
        row.addWidget(self.guides, 0, 0)
        row.addWidget(self.ghost, 0, 1)
        lay.addLayout(row)
        from gui.controls import ThreadSignal
        self.frame_ready = ThreadSignal(self._on_frame)          # emitted by the reader / poller threads
        self.state_ready = ThreadSignal(self._on_state)

    def placeholder(self):
        self.view.setText("\u25c9\n\nNO REAR CAMERA\n\nConnect the webcam to the Pi and start\npi/rear_camera.py  (or  --sim  to try it)")
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

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.snap = None
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 14)
        lay.setSpacing(8)
        self.card = Card(padding=14)
        head = QtWidgets.QHBoxLayout()
        self.dot = QtWidgets.QLabel("\u25cf")
        self.dot.setStyleSheet(f"color: {C['faint']}; font-size: 18px;")
        head.addWidget(self.dot)
        self.status_title = QtWidgets.QLabel("Connecting to the car")
        self.status_title.setFont(theme.semibold(15))
        head.addWidget(self.status_title, 1)
        self.card.body.addLayout(head)
        self.status = QtWidgets.QLabel("connecting to the car's control panel ...")
        self.status.setWordWrap(True)
        self.status.setFont(theme.font(12))
        self.status.setStyleSheet(f"color: {C['dim']};")
        self.card.body.addWidget(self.status)
        lay.addWidget(self.card)
        lay.addWidget(section_label("Drive"))
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(8)
        self.mode_btns = {}
        glyphs = {"drive": "\u25b6", "lidar": "\u25c9", "center": "\u2316", "speed": "\u25d4", "turn": "\u21bb",
                  "oa": "\u2934", "logdrive": "\u25cf"}
        for i, (key, text) in enumerate(self.MODES):
            t = FeatureTile(glyphs.get(key, "\u25cf"), text, state=False, momentary=True, compact=True)
            t.clicked.connect(lambda k=key: self.start(k))
            grid.addWidget(t, i // 2, i % 2)
            self.mode_btns[key] = t
        lay.addLayout(grid)
        stop = FeatureTile("\u23fb", "STOP - zero the motors", "sends zero throttle and steering centre to the car",
                           danger=True, state=False, momentary=True, compact=True)
        stop.clicked.connect(lambda: self.post("/api/stop", {}))
        lay.addWidget(stop)
        lay.addWidget(section_label("Calibration results"))
        self.res_card = Card(padding=12)
        self.results = QtWidgets.QLabel("no results yet")
        self.results.setWordWrap(True)
        self.results.setFont(theme.font(12))
        self.results.setStyleSheet(f"color: {C['text2']};")
        self.res_card.body.addWidget(self.results)
        lay.addWidget(self.res_card)
        note = QtWidgets.QLabel("Nothing is written to the car until you press Apply (a backup of the tuning file is kept).")
        note.setWordWrap(True)
        note.setFont(theme.font(11))
        note.setStyleSheet(f"color: {C['dim']};")
        lay.addWidget(note)
        ga = QtWidgets.QGridLayout()
        ga.setSpacing(8)
        for i, (kind, text) in enumerate((("lidar", "Apply LiDAR"), ("servo_center", "Apply centre"), ("speed", "Apply speed"),
                                          ("turn", "Apply turning"))):
            t = FeatureTile("\u2713", text, state=False, momentary=True, compact=True)
            t.clicked.connect(lambda k=kind: self.apply(k))
            ga.addWidget(t, i // 2, i % 2)
        lay.addLayout(ga)
        lay.addWidget(section_label("Activity"))
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(110)
        self.log.setMaximumHeight(160)
        self.log.setPlaceholderText("Job output appears here")
        lay.addWidget(self.log)
        lay.addStretch(1)
        from gui.controls import ThreadSignal
        self.state_ready = ThreadSignal(self._show)              # emitted by the poll / post threads
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
            self.dot.setStyleSheet(f"color: {C['warn']}; font-size: 18px;")
            self.status_title.setText("Control panel not reachable")
            self.status.setText("The Pi's rc-panel service (port 8080) does not answer. In the laptop simulator there is none - "
                                "calibrations and drive modes need the car.")
            self.results.setText("no results yet")
            return
        self.snap = s
        run = "running" if s.get("running") else "idle"
        self.dot.setStyleSheet(f"color: {C['ok'] if s.get('running') or s.get('relay') else C['dim']}; font-size: 18px;")
        self.status_title.setText(f"{s.get('mode') or 'No job'} \u00b7 {run}")
        self.status.setText(f"{s.get('message', '')}\nrelay (drive mode): {'on' if s.get('relay') else 'off'}")
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


class WheelToScroll(QtCore.QObject):
    """Plots inside a scrolling drawer must not eat the mouse wheel (the drawer would stop scrolling over them): the wheel
    goes to the enclosing scroll area; with Ctrl held it is passed to `on_ctrl(event)` instead (map zoom)."""

    def __init__(self, plot, on_ctrl=None):
        super().__init__(plot)
        self.plot, self.on_ctrl = plot, on_ctrl
        plot.viewport().installEventFilter(self)

    def eventFilter(self, obj, ev):
        if ev.type() != QtCore.QEvent.Wheel:
            return False
        if ev.modifiers() & QtCore.Qt.ControlModifier and self.on_ctrl is not None:
            self.on_ctrl(ev)
            return True
        w = self.plot.parentWidget()
        while w is not None and not isinstance(w, QtWidgets.QScrollArea):
            w = w.parentWidget()
        if w is not None:
            sb = w.verticalScrollBar()
            sb.setValue(sb.value() - ev.angleDelta().y())
            return True
        return False


class StatCard(Card):
    """A small status card: caption, a big value and a detail line (Diagnostics header row)."""

    def __init__(self, caption):
        super().__init__(padding=12)
        cap = QtWidgets.QLabel(caption.upper())
        cap.setFont(theme.semibold(10, spacing=1.4))
        cap.setStyleSheet(f"color: {C['dim']};")
        self.value = QtWidgets.QLabel("\u2014")
        self.value.setFont(theme.light(22))
        self.detail = QtWidgets.QLabel("")
        self.detail.setWordWrap(True)
        self.detail.setFont(theme.font(11))
        self.detail.setStyleSheet(f"color: {C['dim']};")
        self.body.setSpacing(2)
        self.body.addWidget(cap)
        self.body.addWidget(self.value)
        self.body.addWidget(self.detail)

    def set(self, value, detail="", colour=None):
        self.value.setText(str(value))
        self.value.setStyleSheet(f"color: {colour or C['text']};")
        self.detail.setText(detail)


class EmptyList(QtWidgets.QListWidget):
    """A list that says so when it is empty instead of showing a blank box."""

    def __init__(self, text):
        super().__init__()
        self.empty_text = text

    def paintEvent(self, ev):
        super().paintEvent(ev)
        if self.count() == 0:
            p = QtGui.QPainter(self.viewport())
            p.setPen(QtGui.QColor(C["dim"]))
            p.setFont(theme.font(13))
            p.drawText(self.viewport().rect(), QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap, self.empty_text)


class DiagnosticsPanel(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 8, 14, 14)
        lay.setSpacing(8)
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        self.c_health, self.c_lidar, self.c_esp = StatCard("Health"), StatCard("LiDAR and link"), StatCard("ESP32")
        for c in (self.c_health, self.c_lidar, self.c_esp):
            row.addWidget(c, 1)
        lay.addLayout(row)
        trip = QtWidgets.QHBoxLayout()
        trip.setSpacing(8)
        self.t_dist, self.t_time, self.t_top, self.t_brake = (StatCard("Trip distance"), StatCard("Driving time"),
                                                             StatCard("Top speed"), StatCard("Brake events"))
        for c in (self.t_dist, self.t_time, self.t_top, self.t_brake):
            trip.addWidget(c, 1)
        lay.addLayout(trip)
        self.trip = {"dist": 0.0, "time": 0.0, "top": 0.0, "brakes": 0, "last": None, "braking": False}
        self.hist = collections.deque(maxlen=900)
        self.curves = {}
        for title, specs in (("Speed (m/s): estimate vs throttle model", (("v", "cyan", 2), ("vm", "dim", 1))),
                             ("Throttle PWM: driver vs sent", (("in", "dim", 1), ("out", "ok", 2))),
                             ("Free way on the path (m) and crash risk", (("free", "warn", 2), ("risk", "bad", 1)))):
            lay.addWidget(section_label(title))
            card = Card(padding=6)
            pw = theme.style_plot(pg.PlotWidget())
            pw.setMinimumHeight(170)
            pw.setBackground(C["bg1"])
            pw.getPlotItem().hideButtons()
            pw.getPlotItem().vb.setMouseEnabled(x=False, y=False)
            for key, col, wd in specs:
                self.curves[key] = pw.plot(pen=pg.mkPen(EV[col], width=wd))
            WheelToScroll(pw)
            card.body.addWidget(pw)
            lay.addWidget(card)
        lay.addStretch(1)

    def add(self, now, st):
        d = st.get("drive") or {}
        g = st.get("gate") or {}
        p = (st.get("intent") or {}).get("p_crash")
        self.hist.append((now, float(d.get("v", 0) or 0), float(d.get("v_model", 0) or 0), float(d.get("pwm_in", 0) or 0),
                          float(d.get("pwm_out", 0) or 0), g.get("free_m") if g.get("free_m") is not None else np.nan,
                          p if p is not None else np.nan))
        tp = self.trip
        if tp["last"] is not None:
            dt = min(0.5, now - tp["last"])
            v = abs(float(d.get("v", 0) or 0))
            tp["dist"] += v * dt
            tp["time"] += dt if v > 0.03 else 0.0
            tp["top"] = max(tp["top"], v)
            braking = "brak" in str(g.get("action") or "") or str(g.get("action") or "").startswith("hold")
            if braking and not tp["braking"]:
                tp["brakes"] += 1
            tp["braking"] = braking
        tp["last"] = now
        if not self.isVisible() or len(self.hist) < 3:
            return
        self.t_dist.set(f"{tp['dist']:.1f} m", f"{tp['dist'] * CAR_SCALE / 1000:.2f} km at full size")
        self.t_time.set(f"{int(tp['time'] // 60)}:{int(tp['time'] % 60):02d}", "time the car was moving")
        self.t_top.set(f"{kmh(tp['top']):.0f} km/h", f"{tp['top']:.2f} m/s on the car")
        self.t_brake.set(str(tp["brakes"]), "brake / hold interventions", EV["warn"] if tp["brakes"] else None)
        h = np.array(self.hist, float)
        t = h[:, 0] - h[-1, 0]
        for key, i in (("v", 1), ("vm", 2), ("in", 3), ("out", 4), ("free", 5), ("risk", 6)):
            self.curves[key].setData(t, np.nan_to_num(h[:, i], nan=0.0))
        hl, esp = st.get("health") or {}, st.get("esp32") or {}
        state = hl.get("state", "?")
        self.c_health.set(str(state).upper(), "; ".join(hl.get("causes") or []) or "all systems normal",
                          {"normal": EV["ok"], "limp": EV["warn"], "fault": EV["bad"]}.get(state))
        self.c_lidar.set(f"{hl.get('lidar_hz') or '\u2014'} Hz", f"link {hl.get('link_hz') or '\u2014'} packets/s \u00b7 Pi "
                         f"{hl.get('temp_c') or '\u2014'} \u00b0C")
        self.c_esp.set(str(esp.get("state", "?")).upper(), f"{esp.get('detail', '')} \u00b7 reboots {esp.get('reboots', 0)}",
                       EV["ok"] if esp.get("state") == "ok" else None)


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
        self.fps_lbl = QtWidgets.QLabel("- FPS", self.scene)              # frame-rate readout, top right of the scene
        self.fps_lbl.setFont(theme.font(12, spacing=0.6))
        self.fps_lbl.setStyleSheet(f"color: {C['dim']}; background: transparent;")
        self.fps_lbl.setMinimumHeight(18)
        self.mini = MiniMap(self.scene, self.store)
        self.mini.clicked.connect(lambda: self.toggle("map", True))
        self.parking_model, self.parking_published, self._park_drawn = None, None, None
        self._align_s = (SmoothAngle(0.0, 0.35), Smooth(0.0, 0.35), Smooth(0.0, 0.35))   # eased simulator -> relay frame
        self._align_seen = False
        self.intent = Glass(self.scene)
        il = QtWidgets.QVBoxLayout(self.intent)
        il.setContentsMargins(14, 8, 14, 8)
        self.intent_txt = QtWidgets.QLabel("DRIVER RISK")
        self.intent_txt.setFont(theme.font(11, spacing=1.2))
        self.intent_bar = QtWidgets.QProgressBar()
        self.intent_bar.setRange(0, 100)
        self.intent_bar.setTextVisible(False)
        self.intent_bar.setFixedHeight(6)
        self.intent_bar.setStyleSheet(f"QProgressBar {{ background: {C['hair']}; border: none; border-radius: 3px; }} "
                                      f"QProgressBar::chunk {{ background: {C['accent']}; border-radius: 3px; }}")
        il.addWidget(self.intent_txt)
        il.addWidget(self.intent_bar)
        # drawers
        self.events = EmptyList("No events yet.\nCommands you send from the map and the car's notices appear here.")
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
        mp = self.panels["map"][1]
        mp.parking_selected.connect(self._on_parking_selected)
        mp.parking_changed.connect(self._redraw_parking)
        # 60 Hz renderer: packets (20 Hz) are processed when they arrive, everything visual is redrawn every frame from the
        # interpolated pose (gui/render.py)
        self.clock, self.interp = FrameClock(1.0 / 60.0), StateInterpolator()
        self.pose_filter = PoseSmoother()      # the displayed pose: scan-match noise must not shake the car or the camera (AC16)
        self._rx_seen, self._no_data, self._pulse, self._fps_t, self._idle_t = None, True, ("accent", 0.0), 0.0, 0.0
        self.timer = QtCore.QTimer(self)
        self.timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.timer.timeout.connect(self.frame)
        self.fps_cap, self._slow_s, self._frame_i = "auto", 0, 0
        self.timer.start(16)
        # frames paced by the display's vertical sync: the 3D view reports every buffer swap and the next frame is scheduled from
        # there. A free-running 16 ms timer beat against the 16.7 ms refresh - every few frames a swap waited a whole refresh,
        # 30 ms frames among 14 ms ones (measured: sd 6.3 ms), which is the judder. The timer stays as a slow fallback (no swaps
        # when the 3D view is hidden) and as the 30 FPS eco cap.
        self._vsync, self._swap_i = False, 0
        self._pace = QtCore.QTimer(self)                  # one-shot, precise: the wait from a swap to the next frame's start
        self._pace.setSingleShot(True)
        self._pace.setTimerType(QtCore.Qt.PreciseTimer)
        self._pace.timeout.connect(self.frame)
        if hasattr(self.scene, "frameSwapped"):
            self.scene.frameSwapped.connect(self._swapped)
            self._vsync = True
            self.timer.setInterval(100)

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
        self.clock_lbl = QtWidgets.QLabel("")
        self.clock_lbl.setFont(theme.font(13))
        self.clock_lbl.setStyleSheet(f"color: {EV['dim']}; padding-left: 16px;")
        lay.addWidget(self.clock_lbl)
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
        self.theme_btn = QtWidgets.QPushButton("◐  " + ("LIGHT" if theme.MODE == "dark" else "DARK"))
        self.theme_btn.setObjectName("app")
        self.theme_btn.setToolTip("Switch between the dark and the light theme")
        self.theme_btn.clicked.connect(self.toggle_theme)
        lay.addWidget(self.theme_btn)
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
        self.reset_btn.clicked.connect(self._reset_sim)
        self.reset_btn.hide()
        lay.addWidget(self.reset_btn)
        f.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)   # never widen the window
        return f

    def resizeEvent(self, ev):
        """Narrow windows: the bottom bar shows icons only (tooltips carry the names) instead of forcing the window wider."""
        super().resizeEvent(ev)
        QtCore.QTimer.singleShot(0, self._place)
        compact = self.centralWidget().width() < 1420 if self.centralWidget() is not None else self.width() < 1500
        for b in self.app_btns.values():
            full = b.property("full")
            if full:
                b.setText(full.split("  ", 1)[0] if compact else full.replace("&", "&&"))

    def toggle_theme(self):
        """Dark <-> light: switch the palette and rebuild the window (every widget's colours are set at construction)."""
        theme.set_mode("light" if theme.MODE == "dark" else "dark")
        theme.save_mode()
        app = QtWidgets.QApplication.instance()
        app.setStyleSheet(theme.STYLE)
        was_open = [k for k, d in self.docks.items() if d.isVisible()]
        geo = self.geometry()
        new = EVWindow(self.link)
        new.setGeometry(geo)
        if self.isMaximized():
            new.showMaximized()
        elif self.isFullScreen():
            new.showFullScreen()
        else:
            new.show()
        for k in was_open:
            new.toggle(k, True)
        for lab in self.labs.values():
            lab.close()
        app._ev_window = new
        self.timer.stop()
        self.close()

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
        self.pill.fit(w - 300 if w < 1000 else w - 40)          # (the mini-map sits at the right on narrow scenes)
        self.pill.move(max(10, (w - self.pill.width()) // 2), h - self.pill.height() - 18)
        self.mini.move(w - self.mini.width() - 20, h - self.mini.height() - 20)
        self.fps_lbl.adjustSize()
        self.fps_lbl.move(w - self.fps_lbl.width() - 18, 10)

    # --- live
    def refresh(self):
        """Process the newest packet now and draw one frame (tests and tools; the running window uses frame())."""
        self._process(force=True)
        self._animate(1.0 / 60.0, 1)

    MIN_FRAME_MS, ECO_FRAME_MS = 11.0, 33.0   # at most ~90 FPS (30 in eco): smooth on a fast display without taking the CPU

    def _swapped(self):
        """A frame reached the screen: start the next one at once - the display's own pace sets the cadence (60 FPS on a 60 Hz
        panel) - but not sooner than MIN_FRAME_MS after the last one started (a 165 Hz panel would otherwise be chased at full CPU).
        Measured on the twin (1707 x 1067 @ 165 Hz): frame-time sd 6.3 -> ~4 ms and the car's per-frame jump p95 3.0 -> ~2.3 mm
        against the free-running 16 ms timer, which beat against the refresh."""
        eco = self.fps_cap == 30 or getattr(self, "_eco", False)
        since = (time.perf_counter() - self.clock.t_last) * 1000.0
        wait = (self.ECO_FRAME_MS if eco else self.MIN_FRAME_MS) - since
        self._pace.start(int(max(0.0, wait)))

    def set_fps_cap(self, cap):
        """cap: 60, 30 or "auto". Auto starts at 60 and drops to 30 when the machine cannot keep up (3 s below 48 FPS), so a
        display that is too heavy never starves the relay / twin running on the same computer (its control loop must stay
        within budget - the health monitor goes LIMP otherwise)."""
        self.fps_cap = cap
        self._slow_s = 0
        self._eco = cap == 30
        self.timer.setInterval(100 if self._vsync else (33 if cap == 30 else 16))

    def frame(self):
        """The 60 Hz tick: take a new packet if there is one, then redraw everything that moves from the interpolated pose."""
        self.clock.begin()
        dt, n = self.clock.tick()
        self._process(force=False)
        self.clock.mark("packet")
        self._animate(dt, n)
        self.clock.mark("animate")
        now = time.time()
        if now - self._fps_t > 0.5:
            self._fps_t = now
            if self.fps_cap == "auto" and not getattr(self, "_eco", False) and self.clock.fps < 48 and len(self.clock.frames) > 60:
                self._slow_s += 0.5
                if self._slow_s >= 3.0:
                    self._eco = True                       # eco: 30 FPS
                    if not self._vsync:
                        self.timer.setInterval(33)
                    self.add_event(f"display could only reach {self.clock.fps:.0f} FPS: limited to 30 FPS to leave the CPU to the relay")
            else:
                self._slow_s = 0
            self.fps_lbl.setText(f"{self.clock.fps:.0f} FPS" + (" · eco" if getattr(self, "_eco", False) else ""))
            self.fps_lbl.adjustSize()
            self.fps_lbl.move(self.scene.width() - self.fps_lbl.width() - 18, 10)
            prof = self.clock.profile()
            lines = [f"{k}: {v:.2f} ms" for k, v in prof.items()]
            self.fps_lbl.setToolTip(chr(10).join([f"frame {self.clock.frame_ms:.1f} ms"] + lines))

    def _reset_sim(self):
        """Put the simulated car back at its start and re-origin the relay there, so the relay's world frame (and with it
        the parking layout, which is published in the start frame) stays aligned."""
        def work():
            self.link.http_post("/api/sim/reset")
            time.sleep(0.5)
            self.link.send("ORIGIN")
        threading.Thread(target=work, daemon=True).start()
        self.store.clear()
        self.add_event("simulated car reset to its start; world origin re-set there")

    def _update_align(self, sim, world):
        """The Parking Simulator layout is stored in the simulator's frame, but the relay works in its own estimated frame
        (dead reckoning + scan matching drift: 20 degrees after a few turns in the twin). The frame change is measured from the pair
        (true pose in `sim.pose`, the relay's `world.pose`) and eased, so the bays stay on the walls the LiDAR draws."""
        sp, wp = sim.get("pose"), world.get("pose")
        if not (sp and wp):
            return
        ths = math.radians(sp[2])
        dth = math.remainder(wp[2] - ths, 2 * math.pi)
        c, sn = math.cos(dth), math.sin(dth)
        tx, ty = wp[0] - (c * sp[0] - sn * sp[1]), wp[1] - (sn * sp[0] + c * sp[1])
        if not self._align_seen:
            for f, v in zip(self._align_s, (dth, tx, ty)):
                f.snap(v)
            self._align_seen = True
        for f, v in zip(self._align_s, (dth, tx, ty)):
            f.target = f.value + math.remainder(v - f.value, 2 * math.pi) if isinstance(f, SmoothAngle) else v

    def _on_parking_selected(self, msg):
        """The HMI's parking outputs (docs/interfaces.md section 16b): clicked_world_pose, selected_space_id and the occupancy
        map. Kept on the window (`parking_published`) and logged; planning reads them, the HMI never plans."""
        self.parking_published = msg

    def _redraw_parking(self):
        """Redraw the 3D parking objects, but only when the layout, the target or the hover really changed."""
        mp = self.panels["map"][1]
        m = self.parking_model
        key = (m.signature if m else None, mp.park_sel, mp.park_hover)
        if key == self._park_drawn:
            return
        self._park_drawn = key
        self.scene.show_parking(m, mp.park_sel, mp.park_hover)

    def start_measure(self):
        """Record every frame's time and displayed pose (gui.dashboard --measure): FPS, frame-time spread and how smoothly the car
        moves on screen."""
        self._measure = []

    def measure_report(self):
        """Frame statistics of the recording: FPS, frame intervals (mean / p50 / p95 / p99 / max / sd), and shake = the second
        difference of the displayed position per frame (0 for smooth motion; a car that jumps at the packet rate shows up here)."""
        rec = np.array(getattr(self, "_measure", []) or [(0.0, np.nan, np.nan)], float)
        t = rec[:, 0]
        dt = np.diff(t) * 1000
        out = {"frames": int(len(rec)), "seconds": round(float(t[-1] - t[0]), 1) if len(t) > 1 else 0.0}
        if len(dt) > 10:
            out.update(fps=round(len(dt) / (t[-1] - t[0]), 1), frame_ms_mean=round(float(dt.mean()), 2),
                       frame_ms_p50=round(float(np.percentile(dt, 50)), 2), frame_ms_p95=round(float(np.percentile(dt, 95)), 2),
                       frame_ms_p99=round(float(np.percentile(dt, 99)), 2), frame_ms_max=round(float(dt.max()), 1),
                       frame_ms_sd=round(float(dt.std()), 2))
            xy = rec[:, 1:3]
            ok = np.isfinite(xy).all(axis=1)
            xy = xy[ok]
            if len(xy) > 10:
                d2 = np.hypot(*np.diff(xy, n=2, axis=0).T) * 1000
                sp = np.hypot(*np.diff(xy, axis=0).T) / np.maximum(np.diff(t[ok]), 1e-3)
                out.update(shake_mm_p50=round(float(np.percentile(d2, 50)), 3), shake_mm_p95=round(float(np.percentile(d2, 95)), 3),
                           shake_mm_max=round(float(d2.max()), 2), shown_speed_mps_mean=round(float(sp.mean()), 3))
        out["stages_ms"] = {k: round(v, 2) for k, v in self.clock.profile().items()}
        return out

    def _animate(self, dt, n):
        now = time.time()
        for _ in range(n):
            self.cluster.tick(self.clock.step)
        if self._no_data:
            self.scene.set_idle(True, now)
        pose = self.interp.pose_at(now)
        if pose is None:
            self.pose_filter.reset()
        else:
            pose = self.pose_filter.update(pose, dt)
        if getattr(self, "_measure", None) is not None:
            self._measure.append((time.perf_counter(), *(pose[:2] if pose is not None else (np.nan, np.nan))))
        if pose is not None:
            align = tuple(f.step(dt) for f in self._align_s)
            self.scene.set_view(self.scene._pkt_pose, pose, align)
            self.panels["map"][1].set_park_align(align)
            self._frame_i += 1
            if self._frame_i % 2 == 0:                      # the mini-map is a translucent overlay: 30 Hz is plenty
                self.mini.set_view(pose)
            else:                                           # the map drawer repaints its whole point cloud when it follows the
                self.panels["map"][1].animate(pose, 2 * dt) # car: every other frame (measured: 52 -> 64 FPS with it open)
        self.scene.car.pulse(now, *self._pulse)
        self.clock.mark("scene")

    def _process(self, force):
        st, t_rx, arrivals = self.link.snapshot()
        now = time.time()
        stale = st is None or now - t_rx > 2.5
        if not force and not stale and t_rx == self._rx_seen:
            return                              # nothing new: the frame only animates
        if not force and stale and self._no_data and now - self._idle_t < 0.5:
            return
        self._rx_seen = t_rx
        self._idle_t = now
        self.clock_lbl.setText(time.strftime("%H:%M"))
        if stale:      # (2.5 s: a busy planner or a WiFi hiccup must not flash 'NO DATA', TODO D6)
            self._no_data = True
            self.interp.reset()
            self.scene._pkt_pose = None
            self.mode.setText("NO DATA - is the relay / simulator running?")
            self.mode.setStyleSheet(self._chip(EV["bad"]))
            self.chips["link"].setText(f"waiting for {self.link.host}:{CTRL_PORT}")
            self.chips["link"].setStyleSheet(f"color: {EV['dim']};")
            self.pill.set("waiting for the car · no data yet", C["dim"])
            self.intent_txt.setText("DRIVER RISK   —   no data")
            self.intent_bar.setValue(0)
            self.mini.set(None)
            return
        self._no_data = False
        self.scene.set_idle(False)
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
        self.intent_txt.setText(f"DRIVER RISK   {100 * p:.0f} %   \u00b7   " + ("attentive" if intent.get("attentive") else "stick idle")
                                if p is not None else "DRIVER RISK   \u2014")
        self.intent_bar.setValue(int(100 * (p or 0)))
        self._pulse = ("bad" if mode == "EMERGENCY BRAKE" else "warn" if hl.get("state") == "limp" else "accent", 1.6)
        self.scene.show(st)
        sim = st.get("sim") or {}
        lay = sim.get("parking")
        user_lane = self.panels["map"][1].user_lane_layout()      # a lane the user drew takes the place of the simulator's (AD1)
        if user_lane is not None:
            if lay:                                               # in the simulator its cars, pavement and marker stay on screen
                user_lane = dict(user_lane, vehicles=lay.get("vehicles") or [], sidewalk=lay.get("sidewalk"), marker=lay.get("marker"),
                                 road=lay.get("road") or user_lane["road"], sig=(user_lane["sig"], len(lay.get("vehicles") or [])))
            lay = user_lane
        if lay and (lay.get("spaces") or lay.get("strip")):
            from gui.parking import ParkingModel
            sig = ParkingModel.signature_of(lay)
            if self.parking_model is None or self.parking_model.signature != sig:
                self.parking_model = ParkingModel(lay)
        else:
            self.parking_model = None
        self._update_align(sim, world)
        self.panels["map"][1].set_parking(self.parking_model, sim.get("world") if sim else None)
        pose = world.get("pose")
        if pose:
            self.interp.push(t_rx, pose)
            self.scene._pkt_pose = tuple(pose)
        else:
            self.interp.reset()
            self.scene._pkt_pose = None
        self.mini.set(st)
        self._redraw_parking()
        self.panels["map"][1].show_state(st)
        self.panels["assists"][1].show_state(st)
        cal = (st.get("drive") or {}).get("steer_cal") or {}
        kap = -(cal.get("k", 0.0656)) * (float(drive.get("servo", 87)) - float(drive.get("centre", 87)))
        self.panels["rear"][1].set_steering(kap)
        self.panels["rear"][1].set_hazards(polar_xy(st.get("_pts")), float(drive.get("v", 0.0) or 0.0))
        self.panels["diag"][1].add(now, st)
        self._place()
