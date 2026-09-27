"""Worlds for the laptop car (sim/hw_sim.py). World frame: metres, x forward from the car's start, y left.

  doorway    the case from the real car: a box in front of a doorway in a dividing wall
  room       a cluttered room: boxes, a chair (thin legs), a dark box the LiDAR barely sees, a doorway
  corridor   an 80 cm corridor with a box in it
  gap        two boxes with a 35 cm gap (narrower than the car + margins) and a wider one beside
  open       an empty 6 x 4 m room
  log:<file> the room as the car saw it: walls rebuilt from the first scan of a drive log
"""
import math

import numpy as np

from .world import Box, Cone, Wall, World


def _finish(w):
    """Mark which segments belong to dark objects (the LiDAR gets no return from them)."""
    flags = []
    for o in w.objects:
        s = o.segments()
        if len(s):
            flags.append(np.full(len(s), bool(getattr(o, "dark", False))))
    w.dark_segments = np.concatenate(flags) if flags else np.zeros(0, bool)
    return w


def room_walls(w, x0, x1, y0, y1):
    for a, b, c, d in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        w.add(Wall(a, b, c, d))


def doorway():
    w = World()
    room_walls(w, -1.0, 5.0, -1.5, 1.5)
    w.add(Wall(2.8, -1.5, 2.8, -0.36))            # dividing wall with a 72 cm doorway on the car's line
    w.add(Wall(2.8, 0.36, 2.8, 1.5))
    w.add(Box(1.7, 0.0, 0.26, 0.26))              # the box in front of it
    return _finish(w), (0.0, 0.0, 0.0)


def room():
    w = World()
    room_walls(w, -1.0, 4.5, -1.6, 1.6)
    w.add(Box(1.5, 0.35, 0.30, 0.30))
    w.add(Box(2.6, -0.7, 0.40, 0.25, 0.3))
    for dx, dy in ((0, 0), (0.38, 0), (0, 0.38), (0.38, 0.38)):   # a chair: four 2.4 cm legs
        w.add(Cone(3.2 + dx, 0.5 + dy, 0.012))
    dark = Box(0.9, -0.9, 0.35, 0.30)
    dark.dark = True                                             # black/absorbing: no LiDAR return
    w.add(dark)
    w.add(Wall(3.9, -1.6, 3.9, -0.4))
    w.add(Wall(3.9, 0.35, 3.9, 1.6))
    return _finish(w), (0.0, 0.0, 0.0)


def corridor():
    w = World()
    w.add(Wall(-1.0, 0.40, 6.0, 0.40))
    w.add(Wall(-1.0, -0.40, 6.0, -0.40))
    w.add(Wall(6.0, -0.40, 6.0, 0.40))
    w.add(Wall(-1.0, -0.40, -1.0, 0.40))
    w.add(Box(3.0, -0.2, 0.25, 0.2))
    return _finish(w), (0.0, 0.0, 0.0)


def gap():
    w = World()
    room_walls(w, -1.0, 5.0, -1.6, 1.6)
    w.add(Box(2.0, 0.30, 0.3, 0.25))
    w.add(Box(2.0, -0.30, 0.3, 0.25))             # 35 cm gap between these two
    w.add(Box(2.0, -1.10, 0.3, 0.30))             # a wider (0.5 m) gap between these two
    return _finish(w), (0.0, 0.0, 0.0)


def open_room():
    w = World()
    room_walls(w, -2.0, 4.0, -2.0, 2.0)
    return _finish(w), (0.0, 0.0, 0.0)


def from_log(path, max_gap=0.08):
    """Walls from the first scan of a drive log: consecutive returns closer than max_gap are joined."""
    from pi.drive_log import load
    r = load(path)
    t, ang, dist, _q = r["scans"][0]
    yaw = r["meta"]["tuning"]["mount"]["yaw_offset_deg"]
    pts = []
    for a, d in sorted(zip(ang, dist)):
        if d < 0.2:
            continue
        cw = math.radians((a - yaw) % 360.0)
        pts.append((0.12 + d * math.cos(cw), -d * math.sin(cw)))   # vehicle frame, LiDAR 12 cm ahead of the axle
    w = World()
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        if math.hypot(x2 - x1, y2 - y1) < max_gap:
            w.add(Wall(x1, y1, x2, y2))
    return _finish(w), (0.0, 0.0, 0.0)


WORLDS = {"doorway": doorway, "room": room, "corridor": corridor, "gap": gap, "open": open_room}


def build(name):
    if name.startswith("log:"):
        return from_log(name[4:])
    return WORLDS[name]()
