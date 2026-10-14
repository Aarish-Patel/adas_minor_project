"""Parking Simulator view model (TODO G3-2..G3-4, AB1): what the HMI knows about the parking area, and what a click on it sends.

The simulator publishes `state["sim"]["parking"]` (docs/interfaces.md section 15, simulator-only key): the scenario, the strip
polyline (the road edge of the parking lane, the lane on `side` of it, `depth` deep to the kerb), the parked vehicles, road /
sidewalk polygons, the floor paint and, in the perpendicular room, the bays (spaces) and the ArUco marker.

Street (parallel): one continuous lane with cars parked at arbitrary spacing - no spaces. A click on the lane sends the strip and
the click (`PARK_TARGET`); the CAR finds the gaps with its LiDAR and chooses the slot nearest to the click that it fits into
(adas/park_select.py); the HMI shows a car-sized ghost where the user points and, once the relay reports it, the slot it chose.
Bays (perpendicular): the nearest free bay to a click is highlighted (a visual aid); parking there uses the ArUco marker.
A drawn lane (the Parking lane tool: a rectangle on the map, `lane_layout`) is a street of its own in any room or on the car (ADR-043).
This module is presentation only; it never plans.
"""
import math

import numpy as np


def corners(sp):
    """The four corners (4, 2) of a space or vehicle dict {cx, cy, yaw, l, w} in world frame, counter-clockwise."""
    c, s = math.cos(sp["yaw"]), math.sin(sp["yaw"])
    hl, hw = sp["l"] / 2.0, sp["w"] / 2.0
    loc = np.array([[-hl, -hw], [hl, -hw], [hl, hw], [-hl, hw]])
    return np.column_stack([sp["cx"] + c * loc[:, 0] - s * loc[:, 1], sp["cy"] + s * loc[:, 0] + c * loc[:, 1]])


def marker_bits(marker_id, n=4):
    """The n x n payload cells of the drawn marker (visual only: a deterministic pattern of the id, not the ArUco dictionary)."""
    v = (int(marker_id) * 2654435761 + 12345) & 0xFFFFFFFF
    return [[(v >> (r * n + c)) & 1 for c in range(n)] for r in range(n)]


class ParkingModel:
    """The strip and its spaces, with click geometry. Immutable per layout; selection state lives in the callers."""

    def __init__(self, layout):
        self.layout = layout
        self.scenario = layout.get("scenario", "")
        self.spaces = layout.get("spaces") or []
        self.vehicles = layout.get("vehicles") or []
        self.marker = layout.get("marker")
        self.road, self.sidewalk = layout.get("road"), layout.get("sidewalk")
        self.lines = layout.get("lines") or []
        strip = layout.get("strip") or {"points": [[0, 0], [1, 0]], "side": 1}
        self.pts = np.asarray(strip["points"], float).reshape(-1, 2)
        self.side = 1 if strip.get("side", 1) >= 0 else -1
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.heading = np.arctan2(seg[:, 1], seg[:, 0])
        self.by_id = {sp["id"]: sp for sp in self.spaces}
        self.depth = float(strip.get("depth") or max([sp.get("depth", 0.3) for sp in self.spaces] or [0.38]))
        self.depth_max = self.depth
        self.street = not self.spaces                   # a continuous lane: no bays, the car finds the gaps
        self.signature = self.signature_of(layout)

    @staticmethod
    def signature_of(layout):
        return (layout.get("scenario", ""), tuple((sp["id"], sp["occupied"]) for sp in layout.get("spaces") or []),
                len(layout.get("vehicles") or []), (layout.get("marker") or {}).get("id"), layout.get("sig"))

    @property
    def user_lane(self):
        """A lane the user drew (Parking lane tool), not the simulator's street."""
        return self.scenario == "user_lane"

    @classmethod
    def from_state(cls, st):
        lay = ((st or {}).get("sim") or {}).get("parking")
        return cls(lay) if lay and (lay.get("spaces") or lay.get("strip")) else None

    def point_at(self, s):
        """(x, y, heading) of the road edge at s along the strip."""
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1))
        t = (s - self.cum[i]) / max(self.seg_len[i], 1e-9)
        p = self.pts[i] + t * (self.pts[i + 1] - self.pts[i])
        return float(p[0]), float(p[1]), float(self.heading[i])

    def ghost(self, s, car_l=0.33, car_w=0.20):
        """A car-sized rectangle {cx, cy, yaw, l, w} on the lane at s, kerb-side as the planner parks (adas/park_select.KERB_GAP_M)."""
        from adas.park_select import KERB_GAP_M
        x, y, th = self.point_at(float(np.clip(s, 0.0, self.cum[-1])))
        off = max(self.depth / 2, self.depth - KERB_GAP_M - car_w / 2)
        nx, ny = -math.sin(th) * self.side, math.cos(th) * self.side
        return {"cx": x + nx * off, "cy": y + ny * off, "yaw": th, "l": car_l, "w": car_w}

    # --- strip geometry
    def project(self, xy):
        """(s, lateral, heading) of a world point on the strip: s along it (m), lateral = signed distance towards the spaces
        (positive on the space side), heading of the strip there (rad)."""
        p = np.asarray(xy, float)
        best = None
        for i in range(len(self.pts) - 1):
            a, b = self.pts[i], self.pts[i + 1]
            d = b - a
            L2 = float(d @ d)
            t = 0.0 if L2 <= 1e-12 else float(np.clip((p - a) @ d / L2, 0.0, 1.0))
            q = a + t * d
            dist = float(np.hypot(*(p - q)))
            if best is None or dist < best[0]:
                nrm = np.array([-d[1], d[0]]) / max(math.sqrt(L2), 1e-9) * self.side
                best = (dist, self.cum[i] + t * self.seg_len[i], float((p - q) @ nrm), float(self.heading[i]))
        return best[1], best[2], best[3]

    def in_strip(self, xy, margin=0.12):
        """True for a click on the strip: along its length and within the band that holds the spaces (plus a margin, and a little
        onto the road so the road edge itself counts)."""
        s, lat, _ = self.project(xy)
        return -margin <= s <= self.cum[-1] + margin and -0.25 <= lat <= self.depth_max + margin

    def nearest_free(self, xy):
        """The free space closest to the click along the strip (0 inside the space), or None if none is free."""
        s, _, _ = self.project(xy)
        best = None
        for sp in self.spaces:
            if sp["occupied"]:
                continue
            d = max(sp["s0"] - s, 0.0, s - sp["s1"])
            key = (round(d, 6), abs(0.5 * (sp["s0"] + sp["s1"]) - s))
            if best is None or key < best[0]:
                best = (key, sp)
        return best[1] if best else None

    def click(self, xy):
        """A click at world (x, y): the dict published to planning, or None when the click is not on the strip. Street: the click
        and its position `s` along the lane (the car chooses the slot); bays: also the nearest free bay (`selected_space_id`, None if
        every bay is occupied) and the occupancy."""
        if not self.in_strip(xy):
            return None
        s, _, th = self.project(xy)
        out = {"clicked_world_pose": [round(float(xy[0]), 3), round(float(xy[1]), 3), round(th, 4)], "s": round(float(s), 3),
               "selected_space_id": None, "occupancy": {}}
        if not self.street:
            sp = self.nearest_free(xy)
            out["selected_space_id"] = sp["id"] if sp else None
            out["occupancy"] = {q["id"]: q["occupied"] for q in self.spaces}
        return out


# ---------------------------------------------------------------------------------------------- 3D objects (drive scene)
def _fan(pts_xy, z):
    """A convex polygon (n, 2) as a triangle fan at height z -> (V, F)."""
    n = len(pts_xy)
    V = np.column_stack([pts_xy, np.full(n, z)])
    return V, np.array([[0, i, i + 1] for i in range(1, n - 1)])


def _quad3(centre, u, half_u, half_v, n_off=0.0, nrm=None):
    """A vertical quad centred at centre = (x, y, z), spanning `half_u` along the 2D unit vector u and `half_v` in z, pushed
    `n_off` along nrm (2D) off the wall."""
    cx, cy, cz = centre
    ox, oy = (nrm[0] * n_off, nrm[1] * n_off) if nrm is not None else (0.0, 0.0)
    V = np.array([[cx + ox - u[0] * half_u, cy + oy - u[1] * half_u, cz - half_v],
                  [cx + ox + u[0] * half_u, cy + oy + u[1] * half_u, cz - half_v],
                  [cx + ox + u[0] * half_u, cy + oy + u[1] * half_u, cz + half_v],
                  [cx + ox - u[0] * half_u, cy + oy - u[1] * half_u, cz + half_v]])
    return V, np.array([[0, 1, 2], [0, 2, 3]])


def park_meshes(model, pose, selected=None, hover=None):
    """The parking scene as meshes: in the world frame when pose is None (the drive scene's world group), else in the car's
    frame for `pose` (x, y, th): {key: (data, rgba, lit)} with data =
    (V, F) or None. Floor pads: free = green, occupied = red, selected = blue, hover = a light preview; parked vehicles
    are lit solids, the sidewalk a low slab, the marker a black plate with its white payload cells."""
    from gui import theme
    from gui.ev import merge, ribbon, solid_box, world_to_vehicle
    rgb = theme.rgb
    out = {}

    def tv(xy):                                            # pose None: stay in the world frame (the scene moves the group)
        xy = np.asarray(xy, float).reshape(-1, 2)
        return xy if pose is None else world_to_vehicle(xy, pose)

    def to_car(V):                                         # world-frame vertices (n, 3) -> the drawing frame
        V = np.array(V, float)
        V[:, :2] = tv(V[:, :2])
        return V

    def pads(items, z):
        parts = [_fan(tv(corners(sp)), z) for sp in items]
        return merge(parts) if parts else None
    free = [sp for sp in model.spaces if not sp["occupied"] and sp["id"] not in (selected, hover)]
    occ = [sp for sp in model.spaces if sp["occupied"] and sp["id"] not in (selected, hover)]
    out["park_road"] = (_fan(tv(model.road), 0.001) if model.road else None, rgb("bg1", 0.45), False)
    if model.sidewalk:
        sw = np.asarray(model.sidewalk, float)
        (x0, y0), (x1, y1) = sw.min(axis=0), sw.max(axis=0)
        slab = solid_box((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0.0, 0.02)
        out["park_sidewalk"] = ((to_car(slab[0]), slab[1]), rgb("raised", 1.0), True)
    else:
        out["park_sidewalk"] = (None, rgb("raised", 1.0), True)
    out["park_free"] = (pads(free, 0.004), rgb("ok", 0.30), False)
    out["park_occ"] = (pads(occ, 0.004), rgb("bad", 0.30), False)
    if model.street:                                       # street: selected / hover are positions along the lane -> car-sized ghosts
        sel = model.ghost(selected) if selected is not None else None
    else:
        sel = model.by_id.get(selected)
    out["park_sel"] = (pads([sel], 0.006) if sel else None, rgb("accent", 0.55), False)
    if model.street:
        hov = model.ghost(hover) if hover is not None and hover != selected else None
    else:
        hov = model.by_id.get(hover) if hover != selected else None
    out["park_hover"] = (pads([hov], 0.006) if hov else None, rgb("text", 0.20), False)
    edge = []
    for sp in ([sel] if sel else []):
        c = tv(corners(sp))
        edge.append(ribbon(np.vstack([c, c[:1]]), 0.02, 0.008))
    out["park_sel_edge"] = (merge([e for e in edge if e is not None]) if edge else None, rgb("accent", 1.0), False)
    lines = [ribbon(tv(ln), 0.012, 0.005) for ln in model.lines if len(ln) >= 2]
    lines = [ln for ln in lines if ln is not None]
    out["park_lines"] = (merge(lines) if lines else None, rgb("text2", 0.55), False)
    tones = ("s1", "s3", "s4")
    for k, name in enumerate(tones):
        blocks = [solid_box(v["cx"], v["cy"], v["l"], v["w"], v["yaw"], v["h"]) for i, v in enumerate(model.vehicles)
                  if i % 3 == k]
        data = merge([(to_car(V), F) for V, F in blocks]) if blocks else None
        out[f"park_veh{k}"] = (data, rgb(name, 1.0), True)
    m = model.marker
    if m:
        u = np.array([-math.sin(m["yaw"]), math.cos(m["yaw"])])
        nrm = np.array([math.cos(m["yaw"]), math.sin(m["yaw"])])
        half = m["size"] / 2.0
        centre = (m["x"], m["y"], m["z"])
        black = _quad3(centre, u, half, half, 0.003, nrm)
        cell = m["size"] / 6.0
        bits = marker_bits(m["id"])
        whites = []
        for r in range(4):
            for c in range(4):
                if bits[r][c]:
                    cu = (c - 1.5) * cell
                    cz = (1.5 - r) * cell
                    whites.append(_quad3((m["x"] + u[0] * cu, m["y"] + u[1] * cu, m["z"] + cz), u, cell / 2, cell / 2, 0.006, nrm))
        bl = (to_car(black[0]), black[1])
        out["park_marker"] = (bl, (0.05, 0.06, 0.08, 1.0), False)
        out["park_marker_bits"] = ((merge([(to_car(V), F) for V, F in whites]) if whites else None), (0.96, 0.97, 0.98, 1.0), False)
    else:
        out["park_marker"] = (None, (0, 0, 0, 1), False)
        out["park_marker_bits"] = (None, (0, 0, 0, 1), False)
    return out


def sim_to_relay(pt, align):
    """A point in the simulator frame -> the relay's estimated frame; align = (dth rad, tx m, ty m) as the HMI measures it."""
    dth, tx, ty = align
    c, sn = math.cos(dth), math.sin(dth)
    return [c * pt[0] - sn * pt[1] + tx, sn * pt[0] + c * pt[1] + ty]


def relay_to_sim(pt, align):
    """The inverse of sim_to_relay: a point in the relay's frame -> the simulator frame."""
    dth, tx, ty = align
    c, sn = math.cos(dth), math.sin(dth)
    dx, dy = pt[0] - tx, pt[1] - ty
    return [c * dx + sn * dy, -sn * dx + c * dy]


def lane_geometry(lane):
    """A parking lane drawn as a rectangle (the HMI's Parking lane tool; relay / map frame): {x0, y0, x1, y1, road} with `road` the long
    side that is the road edge ("x0" / "x1" / "y0" / "y1"; the opposite long side is the kerb). Returns (road-edge points [[x, y], [x, y]]
    along the long axis, side +-1 of the lane relative to that direction, depth m, kerb points, the rectangle's corners)."""
    x0, x1 = sorted((float(lane["x0"]), float(lane["x1"])))
    y0, y1 = sorted((float(lane["y0"]), float(lane["y1"])))
    road = lane.get("road", "y0")
    corners_ = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    if road in ("y0", "y1"):                                   # the lane runs along x
        yr, yk = (y0, y1) if road == "y0" else (y1, y0)
        pts, kerb = [[x0, yr], [x1, yr]], [[x0, yk], [x1, yk]]
        return pts, (1 if yk > yr else -1), y1 - y0, kerb, corners_
    xr, xk = (x0, x1) if road == "x0" else (x1, x0)          # the lane runs along y; the left of +y is -x
    pts, kerb = [[xr, y0], [xr, y1]], [[xk, y0], [xk, y1]]
    return pts, (1 if xk < xr else -1), x1 - x0, kerb, corners_


def make_lane(a, b, car_xy):
    """The lane for a rectangle dragged from a to b (relay frame): the long side nearer the car (it stands on the road) is the road edge."""
    x0, x1 = sorted((a[0], b[0]))
    y0, y1 = sorted((a[1], b[1]))
    if x1 - x0 >= y1 - y0:
        road = "y0" if abs(car_xy[1] - y0) <= abs(car_xy[1] - y1) else "y1"
    else:
        road = "x0" if abs(car_xy[0] - x0) <= abs(car_xy[0] - x1) else "x1"
    return {"x0": round(x0, 3), "y0": round(y0, 3), "x1": round(x1, 3), "y1": round(y1, 3), "road": road}


def flip_lane(lane):
    """The same rectangle with the kerb on the other long side."""
    other = {"x0": "x1", "x1": "x0", "y0": "y1", "y1": "y0"}
    return dict(lane, road=other[lane.get("road", "y0")])


def lane_layout(lane, align=(0.0, 0.0, 0.0)):
    """A drawn lane (relay frame) as a parking layout in the simulator frame the HMI keeps layouts in (moved by the inverse of `align`;
    on the car there is no simulator and align is the identity): a street with no parked cars known to the HMI - the car finds them with
    its LiDAR - so the Park tool, the map / 3D drawing and PARK_TARGET work on it exactly as on the simulator's street."""
    pts, side, depth, kerb, rect = lane_geometry(lane)
    to_s = lambda P: [[round(v, 4) for v in relay_to_sim(p, align)] for p in P]
    sig = ("user_lane", tuple(round(float(lane[k]), 3) for k in ("x0", "y0", "x1", "y1")), lane.get("road"),
           tuple(round(v, 3) for v in align))
    return {"scenario": "user_lane", "sig": sig, "strip": {"points": to_s(pts), "side": side, "depth": round(depth, 3)},
            "spaces": [], "vehicles": [], "marker": None, "road": to_s(rect), "sidewalk": None, "lines": [to_s(kerb)],
            "kerb": to_s(kerb)}


def align_from_poses(sim_pose, world_pose):
    """The frame change simulator -> relay from the pair (`sim.pose` x, y, deg; `world.pose` x, y, rad), as EVWindow._update_align."""
    dth = math.remainder(world_pose[2] - math.radians(sim_pose[2]), 2 * math.pi)
    c, sn = math.cos(dth), math.sin(dth)
    return (dth, world_pose[0] - (c * sim_pose[0] - sn * sim_pose[1]), world_pose[1] - (sn * sim_pose[0] + c * sim_pose[1]))


def park_target_command(layout, msg, align=(0.0, 0.0, 0.0)):
    """The relay command for a click on the parking lane (docs/interfaces.md 15, 16b): `PARK_TARGET <json>` with the strip (road-edge
    polyline, lane side, lane depth) and the click, both in the RELAY's frame: the layout is stored in the simulator's frame and moved
    by `align`; `msg["clicked_world_pose"]` is already in the relay's frame. No spaces: the car finds the gaps itself."""
    import json
    from adas.park_select import LANE_DEPTH_M
    st = layout["strip"]
    strip = {"points": [[round(v, 4) for v in sim_to_relay(p, align)] for p in st["points"]], "side": st["side"],
             "depth": float(st.get("depth") or LANE_DEPTH_M)}
    return "PARK_TARGET " + json.dumps({"click": msg["clicked_world_pose"][:2], "strip": strip}, separators=(",", ":"))
