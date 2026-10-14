"""Parking Simulator worlds for the laptop car (TODO G3-2): two rooms built from the twin's own object classes, plus the
layout the HMI draws (spaces as objects, parked vehicles, road, sidewalk, ArUco marker).

  park_perp    perpendicular bays: a room whose rear wall carries an ArUco marker over the target bay; five 46 cm bays in a
               row, two of them occupied by parked vehicles; the car starts facing away from the wall (reverse-in)
  park_street  a parallel-parking street: road, one continuous parking lane along the kerb with cars parked at arbitrary
               spacing (no spaces or bays: the car finds the gaps itself), sidewalk and facade

Only what a 2D LiDAR could see is a physical object (walls, parked vehicles); the road paint, the kerb / sidewalk slab and the
ArUco marker are flat and live in the layout only (the marker is also a `Marker` for the camera simulator). Sizes are for the
1:14 car (0.36 x 0.20 m). The layout is stored in the simulator frame; `layout_for_hmi()` converts it to the relay's world frame
(origin and heading of the car's start pose). The street's strip (road-edge polyline, side, lane depth) is what the HMI publishes
clicks against; the perpendicular room keeps its bays (spaces) for display.
"""
import math

import numpy as np

from .hw_worlds import _finish, room_walls
from .world import Box, FloorLine, Marker, Wall, World

CAR_L, CAR_W = 0.36, 0.20
BAY_W, BAY_D = 0.46, 0.62
MARKER_SIZE = 0.16      # m; a 10 cm marker is under 18 px (undetectable) beyond ~2.4 m, and the car starts 3 m from the wall


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def _space(sid, cx, cy, yaw, length, width, occupied, s0=0.0, s1=0.0, depth=0.0):
    return {"id": sid, "cx": cx, "cy": cy, "yaw": yaw, "l": length, "w": width, "occupied": bool(occupied),
            "s0": s0, "s1": s1, "depth": depth}


def _vehicle(cx, cy, yaw, tone, length=CAR_L, width=CAR_W, height=0.12):
    return {"cx": cx, "cy": cy, "yaw": yaw, "l": length, "w": width, "h": height, "tone": tone}


def perpendicular(marker_id=23):
    """Five perpendicular bays along the rear wall (x = -3.0), the car at the origin facing +x (its rear towards the wall)."""
    w = World()
    x_wall = -3.0
    room_walls(w, x_wall, 2.0, -1.6, 1.6)
    ys = [(i - 2) * BAY_W for i in range(5)]
    occupied = {1, 3}
    spaces, vehicles = [], []
    strip_y0 = ys[0] - BAY_W / 2
    for i, y in enumerate(ys):
        cx = x_wall + BAY_D / 2
        spaces.append(_space(f"P{i}", cx, y, 0.0, BAY_D, BAY_W, i in occupied,
                             s0=i * BAY_W, s1=(i + 1) * BAY_W, depth=BAY_D))
        if i in occupied:                                                  # a parked vehicle, nose out, off the wall by 6 cm
            vx = x_wall + 0.06 + CAR_L / 2
            w.add(Box(vx, y, CAR_L, CAR_W + 0.02, 0.0, height=0.12, style="car", colour="#8fa3bf"))
            vehicles.append(_vehicle(vx, y, 0.0, i % 3))
    lines = []
    for k in range(6):                                                     # bay separators and the entrance line
        yy = strip_y0 + k * BAY_W
        lines.append([[x_wall, yy], [x_wall + BAY_D, yy]])
    lines.append([[x_wall + BAY_D, strip_y0], [x_wall + BAY_D, strip_y0 + 5 * BAY_W]])
    for pts in lines:
        w.add(FloorLine(pts, width=0.02))
    w.add(Marker(marker_id, x_wall, 0.0, 0.0, z=0.09, size=MARKER_SIZE))
    w.layout = {
        "scenario": "perpendicular",
        "strip": {"points": [[x_wall + BAY_D, strip_y0], [x_wall + BAY_D, strip_y0 + 5 * BAY_W]], "side": 1},
        "spaces": spaces, "vehicles": vehicles, "lines": lines,
        "road": None, "sidewalk": None,
        "marker": {"id": int(marker_id), "x": x_wall, "y": 0.0, "yaw": 0.0, "z": 0.09, "size": MARKER_SIZE, "space": "P2"},
    }
    return _finish(w), (0.0, 0.0, 0.0)


# The parallel street: ONE continuous parking lane along the kerb - no spaces, no bays, no paint between cars (as on a real street).
# Parked cars of different lengths stand at arbitrary spacing, a few cm off the kerb and a little skewed:
# (rear end x m, length m, gap to the kerb m, yaw deg). The free gaps between them are (the 0.33 m laptop car needs 0.63 m:
# adas/park_select.min_space_length): 0.55 no, 1.30 comfortable, 0.50 no, 1.02 normal, 1.90 long, 0.86 tight, then the lane runs on
# 1.3 m to its end (the road continues past it).
STREET_CARS = [(-0.95, 0.40, 0.05, 1.0), (0.00, 0.38, 0.04, -1.0), (1.68, 0.40, 0.06, 2.0), (2.58, 0.36, 0.03, 0.0),
               (3.96, 0.42, 0.05, -2.0), (6.28, 0.36, 0.04, 1.0), (7.50, 0.40, 0.06, 0.0)]
LANE_DEPTH = 0.38       # road edge to kerb (adas/park_select.LANE_DEPTH_M)


def parallel_street():
    """A street along +x: road y in [-0.85, 0.55], a continuous parking lane on the left (0.55 .. 0.93), a sidewalk behind it and a
    facade wall. Cars parked at arbitrary spacing (STREET_CARS); the car starts on the road beside the second parked car."""
    w = World()
    x0, x1, x_end, y_road0, y_edge, y_face = -1.2, 9.2, 10.7, -0.85, 0.55, 1.45      # the lane ends at x1, the road at x_end
    y_kerb = y_edge + LANE_DEPTH
    w.add(Wall(x0, y_road0, x_end, y_road0))                                # the far side of the road (a fence)
    w.add(Wall(x0, y_face, x_end, y_face))                                  # the building facade behind the sidewalk
    w.add(Wall(x0, y_road0, x0, y_face))
    w.add(Wall(x_end, y_road0, x_end, y_face))
    vehicles = []
    for k, (xr, length, off, yaw_deg) in enumerate(STREET_CARS):
        cx, cy, yaw = xr + length / 2, y_kerb - off - CAR_W / 2, math.radians(yaw_deg)
        w.add(Box(cx, cy, length, CAR_W, yaw, height=0.12, style="car", colour="#8fa3bf"))
        vehicles.append(_vehicle(cx, cy, yaw, k % 3, length=length))
    lines = [[[x0, y_edge], [x1, y_edge]]]                                  # the lane's edge line; the kerb is the sidewalk's edge
    for pts in lines:
        w.add(FloorLine(pts, width=0.02))
    w.layout = {
        "scenario": "parallel",
        "strip": {"points": [[x0, y_edge], [x1, y_edge]], "side": 1, "depth": LANE_DEPTH},
        "spaces": [], "vehicles": vehicles, "lines": lines,
        "road": [[x0, y_road0], [x_end, y_road0], [x_end, y_edge], [x0, y_edge]],
        "sidewalk": [[x0, y_kerb], [x_end, y_kerb], [x_end, y_face], [x0, y_face]],
        "marker": None,
    }
    return _finish(w), (0.0, -0.15, 0.0)


def layout_for_hmi(world, start):
    """The layout in the relay's world frame (origin = the car's start pose, x ahead of it) as a JSON-ready dict, or None."""
    lay = getattr(world, "layout", None)
    if not lay:
        return None
    x0, y0, th0 = start
    c, s = math.cos(-th0), math.sin(-th0)

    def pt(p):
        dx, dy = p[0] - x0, p[1] - y0
        return [round(c * dx - s * dy, 3), round(s * dx + c * dy, 3)]

    def pose(d):
        out = dict(d)
        out["cx"], out["cy"] = pt((d["cx"], d["cy"]))
        out["yaw"] = round(_wrap(d["yaw"] - th0), 4)
        return out
    out = {"scenario": lay["scenario"], "strip": {"points": [pt(p) for p in lay["strip"]["points"]],
                                                 "side": lay["strip"]["side"], "depth": lay["strip"].get("depth")},
           "spaces": [pose(d) for d in lay["spaces"]], "vehicles": [pose(d) for d in lay["vehicles"]],
           "lines": [[pt(p) for p in ln] for ln in lay["lines"]],
           "road": [pt(p) for p in lay["road"]] if lay["road"] else None,
           "sidewalk": [pt(p) for p in lay["sidewalk"]] if lay["sidewalk"] else None, "marker": None}
    m = lay["marker"]
    if m:
        mx, my = pt((m["x"], m["y"]))
        out["marker"] = dict(m, x=mx, y=my, yaw=round(_wrap(m["yaw"] - th0), 4))
    return out


def build(name, marker_id=23):
    if name == "park_perp":
        return perpendicular(marker_id)
    return parallel_street()
