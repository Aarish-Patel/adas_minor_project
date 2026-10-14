"""Parking selection and reverse-parking trajectories (planning only: no GUI, no hardware).

Two modes, both returning a `ParkingPlan(selected_space_id, parking_goal, planned_path)`:

Perpendicular (`plan_perpendicular`): an ArUco marker on the bay's rear wall gives the bay's centre line (adas/markers.py:
solvePnP on the four corners, Lepetit et al. EPnP/IPPE family). The goal is the rear-axle pose on that line, nose out
(reverse-in), with the rear bumper `stop_gap` from the wall. The path is a reverse Dubins curve (Dubins 1957; adas/dubins.py) to
a point `align_m` out on the centre line, then a straight reverse: the car arrives aligned with the marker. If the car
cannot reverse straight to that line (marker not behind it), it first drives forward to a staging pose on the line.

Parallel (`plan_parallel`): the parking lane is one continuous strip along the kerb (road-edge polyline + lane depth) with other
cars parked on it at arbitrary spacing - there are no predefined spaces. The free gaps between the parked cars are found from
the obstacle points in the lane band (`find_gaps`: projection onto the strip, 1-D clustering along it); a gap shorter than the
car plus the manoeuvre room (`min_space_length`) is not viable; in every viable gap a slot is placed as near to the user's click
as the gap allows (`choose_slots`) and the slots are tried nearest first. The trajectory is the classic reverse S of two equal
arcs after a pull-up alongside the slot (Paromtchik & Laugier 1996; d = 2R(1 - cos theta)), planned below the tightest curvature
for a control margin; a gap too short for it gets a multi-stroke manoeuvre planned as the EXIT from the parked pose and driven
backwards (tiny-spot method of Vorobieva, Glaser, Minoiu-Enache & Mammar, IEEE T-ITS 2015): deterministic, a few ms, exact
rectangle clearance. As in production automated
parking, the gap is measured again when the car is alongside it (short range, no accumulated drift) and the manoeuvre is
re-planned for the gap found in the close-range scan (`refine_parallel`).

Frames: everything is in one working frame (the vehicle frame when the request is made, unless `pose` is given). Space frame
F: x along the strip, y from the road toward the kerb, origin at the space's rear end on the road edge.
Goals are (x, y, heading_deg) of the REAR AXLE (as adas/park.park_goal); paths are (N, 4) x, y, theta rad, direction +-1.
"""
import math
from typing import NamedTuple, Optional, Sequence

import numpy as np

from . import dubins
from .hybrid_astar import Grid, HybridAStar

KAPPA_PARK = 1.5          # 1/m, the planner's steering limit (adas/assists.plan_kappa_max)
CLEAR_M = 0.04            # body clearance kept to neighbours, kerb and wall
COMFORT_CLEAR_M = 0.13    # tried first for the S and the approach; CLEAR_M is the fallback
END_MARGIN_M = 0.04       # a space must be at least the car's length + 2 x this
STOP_GAP_M = 0.06         # perpendicular: rear bumper to the wall at the goal
ALIGN_M = 0.30            # perpendicular: straight reverse on the centre line at the end
DS = 0.02                 # path sampling
LANE_DEPTH_M = 0.38       # parking lane: road edge to kerb (1:14 street; the HMI sends the strip's own depth)
GAP_JOIN_M = 0.12         # obstacle returns closer than this along the strip belong to one parked object
MIN_GAP_EXTRA_M = 0.30    # a gap must be the car's length + this (min_space_length)
STROKE_CLEAR_M = 0.07     # a stroke inside a tight gap ends this far (rectangle to obstacle) before a neighbour: where the brake gate still
                          # lets the car creep (CREEP_MIN_M 6 cm + 1.2 cm) - at 5 cm the steering noise left strokes unfinished, stalled
PULL_OUT_CLEAR_M = 0.08   # the pull-out's strokes first try to end this far from a neighbour: where the brake gate still lets the car creep
                          # (pi/path_gate.py CREEP_MIN_M 6 cm + its 1.2 cm margin at a crawl) - 5 cm when that finds no way out
LANE_GAPS = (0.10, 0.16, 0.24, 0.32)   # road-edge-to-body gaps tried for the pull-up alongside the slot (m)
KERB_GAP_M = 0.05         # the parked car's kerb-side body edge this far from the kerb (the street's cars stand 3-6 cm off it; 3 cm clipped it)
KERB_CLEAR_M = 0.03       # clearance kept to the kerb along the path - the rear corner swings toward it in the S (1.5 cm: the wheels clipped it)
KERB_MIN_M = 0.035        # in line with the neighbours, but the body's kerb side never nearer the (believed) kerb than this
KERB_FAR_M = 0.06         # ... nor further from it than this (a row parked far out is not followed: user, 30 Sep 'parks too far out')
KERB_SEEN_GAP_M, KERB_CLEAR_SEEN_M = 0.06, 0.05   # a kerb the LiDAR sees (returns along the kerb line): the parked car stays this far off it
                          # and the path this clear - outside the brake gate's envelope (1.2 cm + 2 cm path tolerance at a crawl), or the
                          # gate holds the car on its last stroke (30 Sep: a lane drawn along a wall stalled at 3 cm)
KERB_OBJ_M = 0.08         # returns within this of the kerb line are not used to find the parked cars (a seen kerb, drifted); cars show further out
KERB_SEEN_M = 0.03        # returns within this of the kerb line are the kerb itself (a wall / board kerb), not a parked car: 3.5 x the
                          # LiDAR's 8 mm range noise (1.5 cm let noisy wall returns split a drawn lane into gaps too short) and a drawing
                          # error of 3 cm
REFINE_REACH_M = 1.5      # alongside the gap, gaps this far from the planned slot are measured well enough to choose again
SLOT_MAX_M = 1.6          # a slot is never longer than this: in a longer gap the car parks where the user clicked
SLOT_STEP_M, SLOT_CANDIDATES = 0.25, 24   # other positions tried along a longer gap (0.25 m apart), and at most this many slots in all
MIN_DUBINS_M = 0.45       # a forward Dubins approach needs at least this distance to the pull-up point
RUN_IN_M = 0.40           # straight forward run before the reversing point: the heading settles on the strip's
S_MARGIN = 0.8            # the S is planned at this share of the tightest curvature first: control margin for the steering lag


class ParkingPlan(NamedTuple):
    """The only outputs: None everywhere when nothing can be planned."""
    selected_space_id: Optional[object]
    parking_goal: Optional[tuple]          # (x, y, heading_deg), rear axle, working frame
    planned_path: Optional[np.ndarray]     # (N, 4) x, y, theta, direction
    slot: Optional[np.ndarray] = None        # (4, 2) corners of the chosen slot in the working frame (parallel; display, re-plan)
    request: Optional[dict] = None           # parallel: {"strip": (points, side, depth), "click": (x, y)} in the working frame, so the
                                             # car can measure the gap again when it is alongside (refine_parallel); bay:
                                             # {"wall": (point, normal), "stop_gap": m} - the LiDAR sets the final stop (AutoNav)
    pull_up: Optional[tuple] = None          # parallel: the pull-up pose (x, y, heading rad) where the manoeuvre begins
    starts_at_car: bool = True               # False: planned_path starts at pull_up - the car first drives there with the click-to-go
                                             # planner (AutoNav), as production park assists drive to the start of the manoeuvre
    pull_out: Optional[np.ndarray] = None    # parallel: the car stands parked - these poses (N, 4) take it out onto the road first


NO_PLAN = ParkingPlan(None, None, None)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ------------------------------------------------------------------------------------------------ strip and spaces
class ParkingSpace:
    """A slot on the strip: [s0, s1] along it (m), `depth` from the road edge to the kerb (m). `gap` = (g0, g1): the free stretch it lies
    in (the parked neighbours begin there; default = the slot itself)."""

    def __init__(self, space_id, s0, s1, depth, occupied=False, gap=None):
        self.id, self.s0, self.s1, self.depth, self.occupied = space_id, float(s0), float(s1), float(depth), bool(occupied)
        self.gap = (self.s0, self.s1) if gap is None else (float(gap[0]), float(gap[1]))
        self.face = None                     # the neighbours' road-side faces (y from the road edge) as scanned, when known
        self.kerb_seen = False               # the scan shows returns along the kerb line (a wall / board kerb)

    @property
    def length(self):
        return self.s1 - self.s0

    @property
    def mid(self):
        return 0.5 * (self.s0 + self.s1)


class ParkingStrip:
    """The road edge of the parking lane as a polyline in the working frame, in the direction of travel; the lane lies on `side`
    (+1 left, -1 right) of it and is `depth` deep (road edge to kerb)."""

    def __init__(self, points, side=1, depth=LANE_DEPTH_M):
        self.pts = np.asarray(points, float).reshape(-1, 2)
        self.side = 1 if side >= 0 else -1
        self.depth = float(depth)
        seg = np.diff(self.pts, axis=0)
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.heading = np.arctan2(seg[:, 1], seg[:, 0])

    @property
    def length(self):
        return float(self.cum[-1])

    def project(self, p):
        """Closest point on the strip to p: (s along the strip, distance)."""
        best = (math.inf, 0.0)
        for i, ln in enumerate(self.seg_len):
            a, d = self.pts[i], self.pts[i + 1] - self.pts[i]
            t = 0.0 if ln < 1e-9 else float(np.clip(np.dot(np.asarray(p, float) - a, d) / (ln * ln), 0.0, 1.0))
            dist = float(np.hypot(*(a + t * d - np.asarray(p, float))))
            if dist < best[0]:
                best = (dist, self.cum[i] + t * ln)
        return best[1], best[0]

    def to_strip(self, points):
        """Points (N, 2) of the working frame -> (s along the strip, y from the road edge toward the kerb), nearest segment each."""
        P = np.asarray(points, float).reshape(-1, 2)
        best_d = np.full(len(P), np.inf)
        s_out, y_out = np.zeros(len(P)), np.zeros(len(P))
        for i, ln in enumerate(self.seg_len):
            a, d = self.pts[i], self.pts[i + 1] - self.pts[i]
            u = d / max(ln, 1e-9)
            rel = P - a
            t = np.clip(rel @ u, 0.0, ln)
            q = a + t[:, None] * u
            dist = np.hypot(*(P - q).T)
            y = self.side * (u[0] * rel[:, 1] - u[1] * rel[:, 0])
            better = dist < best_d
            best_d = np.where(better, dist, best_d)
            s_out = np.where(better, self.cum[i] + rel @ u, s_out)
            y_out = np.where(better, y, y_out)
        return s_out, y_out

    def point_at(self, s):
        i = int(np.clip(np.searchsorted(self.cum, s, side="right") - 1, 0, len(self.seg_len) - 1))
        t = (s - self.cum[i]) / max(self.seg_len[i], 1e-9)
        p = self.pts[i] + t * (self.pts[i + 1] - self.pts[i])
        return float(p[0]), float(p[1]), float(self.heading[i])

    def frame(self, space):
        """Origin (rear end of the space on the road edge) and strip heading at the space's middle."""
        ox, oy, _ = self.point_at(space.s0)
        return ox, oy, self.point_at(space.mid)[2]

    def to_frame(self, space, poses):
        """Poses (N, 3) of the working frame -> space frame F."""
        ox, oy, t = self.frame(space)
        poses = np.atleast_2d(poses)
        c, s = math.cos(t), math.sin(t)
        dx, dy = poses[:, 0] - ox, poses[:, 1] - oy
        out = np.column_stack([c * dx + s * dy, self.side * (-s * dx + c * dy), self.side * np.array([_wrap(a - t) for a in poses[:, 2]])])
        return out

    def from_frame(self, space, poses):
        ox, oy, t = self.frame(space)
        poses = np.atleast_2d(poses)
        c, s = math.cos(t), math.sin(t)
        yl = self.side * poses[:, 1]
        out = np.column_stack([ox + c * poses[:, 0] - s * yl, oy + s * poses[:, 0] + c * yl,
                               [_wrap(t + self.side * a) for a in poses[:, 2]]])
        return out


def find_gaps(strip, points, min_points=3):
    """Free stretches of the parking lane: the obstacle points (working frame) inside the lane band are projected onto the strip and
    clustered along it (a new object starts after GAP_JOIN_M of nothing); what lies between the objects and the strip's ends is free.
    Returns [(s0, s1)] along the strip. Only what the sensor has seen is known: a stretch it has not seen yet counts as free here and is
    checked again when the car is alongside (AutoNav re-measures the gap)."""
    L = strip.length
    pts = np.asarray(points if points is not None else [], float).reshape(-1, 2)
    if not len(pts):
        return [(0.0, L)]
    s, y = strip.to_strip(pts)
    # (up to KERB_OBJ_M short of the kerb: a kerb the LiDAR does see - a wall, a skirting board, its returns drifted a little in the
    # obstacle memory - is not one long parked car; a parked car is found by its road-side face and ends, further out)
    m = (y > -0.01) & (y < strip.depth - KERB_OBJ_M) & (s > -0.3) & (s < L + 0.3)
    order = np.argsort(s[m])
    s, ys = s[m][order], y[m][order]
    objs = []
    if len(s):
        cut = np.where(np.diff(s) > GAP_JOIN_M)[0]
        starts = np.concatenate([[0], cut + 1])
        ends = np.concatenate([cut, [len(s) - 1]])
        objs = [(s[i], s[j]) for i, j in zip(starts, ends) if j - i + 1 >= min_points]
    gaps, prev = [], 0.0
    for a, b in objs:
        if a > prev:
            gaps.append((prev, float(min(a, L))))
        prev = max(prev, float(b))
    if prev < L:
        gaps.append((prev, L))
    return [(float(a), float(b)) for a, b in gaps if b - a > 0.05]


def choose_slots(gaps, s_click, car, depth=LANE_DEPTH_M):
    """Slots in the viable gaps, nearest to the click first: in a gap at least `min_space_length` long the slot is as long as the gap
    (at most SLOT_MAX_M) and slides along it to be as near to the click as the gap allows. Returns [ParkingSpace]."""
    need = min_space_length(car)
    if depth < car.width + 2 * CLEAR_M:                   # the lane itself is too shallow for the car
        return []
    out = []
    for g0, g1 in gaps:
        if g1 - g0 < need:
            continue
        L = min(g1 - g0, SLOT_MAX_M)
        lo, hi = g0 + L / 2, g1 - L / 2
        c = float(np.clip(s_click, lo, hi))
        # in a gap longer than the slot, more positions along it (nearest to the click first): the one at the click may have no way
        # in (30 Sep: a chair beside the pull-up point in a room refused the whole lane) - the nearest spot that can be parked counts
        cand = [c] + [v for k in range(1, 1 + int((hi - lo) / SLOT_STEP_M)) for v in (c - k * SLOT_STEP_M, c + k * SLOT_STEP_M)
                      if lo - 1e-9 <= v <= hi + 1e-9]
        for v in cand:
            out.append((abs(v - s_click), ParkingSpace(f"strip {v:.2f} m", v - L / 2, v + L / 2, depth, gap=(g0, g1))))
    return [sp for _, sp in sorted(out, key=lambda o: o[0])][:SLOT_CANDIDATES]


# ------------------------------------------------------------------------------------------------ body helpers
def _body(car):
    front, rear, w = car.front_x, car.rear_x, car.width
    return front, rear, w, front - rear, 0.5 * (front + rear)


def min_space_length(car, kappa=KAPPA_PARK):
    """Shortest gap the car parks in: its length + MIN_GAP_EXTRA_M. Measured with the multi-stroke planner (`_tight_gap`, exact rectangle,
    5 cm stroke clearance) on the 0.33 m car at 1.5 1/m: 0.55 m needs 9 gear changes, 0.65 m 5, 0.70-0.80 m 3, 0.9-1.2 m 1 (reverse in +
    pull forward), from 1.4 m a single reverse S. The limit keeps it to about 6 changes."""
    _, _, _, length, _ = _body(car)
    return length + MIN_GAP_EXTRA_M


def fits(space, car):
    """Necessary size: long enough (min_space_length) and a lane deep enough for the car's width + 4 cm on each side."""
    _, _, w, _, _ = _body(car)
    return space.length >= min_space_length(car) - 1e-9 and space.depth >= w + 2 * CLEAR_M


def _edge(p0, p1, step=0.02):
    n = max(2, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step))
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(p0, float) + t * (np.asarray(p1, float) - np.asarray(p0, float))


def _obstacles(space):
    """Obstacle points in F: the parked neighbours where the gap ends (blocks 0.6 m long, as deep as the lane + 0.1; a gap that runs to
    the strip's end is closed there as well). The kerb (the line y = depth in F) is checked on its own (`_kerb_room`)."""
    D = space.depth + 0.10
    r, f = space.gap[0] - space.s0, space.gap[1] - space.s0          # the gap's ends in F (r <= 0, f >= L)
    pts = []
    for x0, x1, face in ((r - 0.6, r, r), (f, f + 0.6, f)):
        pts += [_edge((x0, 0.0), (x1, 0.0)), _edge((face, 0.0), (face, D)), _edge((x0, D), (x1, D))]
    return np.vstack(pts)


def _kerb_room(poses, car, kerb):
    """Distance (m) from the body's kerb-most corner to the kerb line y = `kerb` in F, per pose (N,); +inf without a kerb. The kerb is a
    few cm high: the parked cars stand 3-6 cm off it, so the car may come closer to it than to a neighbour (KERB_CLEAR_M)."""
    poses = np.atleast_2d(poses)
    if kerb is None:
        return np.full(len(poses), np.inf)
    front, rear, w, _, _ = _body(car)
    c, sn = np.cos(poses[:, 2]), np.sin(poses[:, 2])
    y = np.max([poses[:, 1] + lx * sn + ly * c for lx in (front, rear) for ly in (-w / 2, w / 2)], axis=0)
    return kerb - y


def _arc(pose, kappa, dist, ds=DS):
    """Poses along a circular arc travelled a SIGNED distance (negative = reverse), excluding the start."""
    n = max(1, int(math.ceil(abs(dist) / ds)))
    s = np.linspace(dist / n, dist, n)
    x, y, th = pose
    if abs(kappa) < 1e-9:
        return np.column_stack([x + s * math.cos(th), y + s * math.sin(th), np.full(n, th)])
    ths = th + kappa * s
    return np.column_stack([x + (np.sin(ths) - math.sin(th)) / kappa, y - (np.cos(ths) - math.cos(th)) / kappa, ths])


def _clear(ha, pts, poses, margin, pad=0.6, kerb=None):
    """True when the car's exact rectangle (`_obb_clearance`) stays `margin` off every point along `poses` (points cropped to the
    poses' surroundings) and KERB_CLEAR_M off the kerb line y = `kerb` (F; None = no kerb). The planner's grid + circle cover over-states
    the body by ~6 cm at the bumpers and quantises to 5 cm - fine for route search, too coarse for parking between cars."""
    if not (_kerb_room(poses, ha.p, kerb) >= KERB_CLEAR_M).all():
        return False
    pts = np.asarray(pts, float).reshape(-1, 2)
    if not len(pts):
        return True
    x0, x1 = poses[:, 0].min() - pad, poses[:, 0].max() + pad
    y0, y1 = poses[:, 1].min() - pad, poses[:, 1].max() + pad
    near = pts[(pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)]
    return bool((_obb_clearance(poses, near, ha.p) >= margin).all())


def _with_dir(poses, d):
    return np.column_stack([poses, np.full(len(poses), d)])


# ------------------------------------------------------------------------------------------------ parallel parking
def _neighbour_face(space, pts_f, band=0.25, min_points=6):
    """The road-side faces of the parked neighbours at the gap's ends, measured in the scan (F: x along the slot, y from the road edge
    toward the kerb): for each neighbour, its returns within `band` of the gap end and inside the lane band; the face is the median of
    the lowest third of their y (the face toward the road; its end face runs from there toward the kerb). Returns the mean face y of the
    neighbours seen well enough, or None."""
    P = np.asarray(pts_f, float).reshape(-1, 2)
    if not len(P):
        return None
    r, f = space.gap[0] - space.s0, space.gap[1] - space.s0
    faces = []
    for lo, hi in ((r - band, r + 0.02), (f - 0.02, f + band)):
        y = np.sort(P[(P[:, 0] >= lo) & (P[:, 0] <= hi) & (P[:, 1] > -0.03) & (P[:, 1] < space.depth - KERB_SEEN_M), 1])
        if len(y) >= min_points:
            faces.append(float(np.median(y[:max(3, len(y) // 3)])))
    return float(np.mean(faces)) if faces else None


def _goal_in_space(space, car):
    """Rear-axle goal in F, the body centred along the slot. Across it: in line with the parked neighbours - the body's road side on their
    road-side faces as the scan measures them at the gap (space.face) - as production parallel-park systems place the target from the
    detected vehicle contours (Pohl, Sethsson & Ekmark, "A semi-automated parallel parking system for passenger cars", Proc. IMechE D
    220, 2006): measured at short range beside the gap it carries no odometry drift, the kerb (invisible to a 2D LiDAR) does. Never nearer
    the believed kerb than KERB_MIN_M, never nearer the road than the lane's middle; without neighbours KERB_GAP_M off the kerb."""
    front, rear, w, length, centre = _body(car)
    kmin = KERB_SEEN_GAP_M if getattr(space, "kerb_seen", False) else KERB_MIN_M   # a kerb the brake gate sees needs its envelope
    y = space.depth - max(KERB_GAP_M, kmin) - w / 2
    face = getattr(space, "face", None)
    if face is not None and space.depth / 2 - 0.05 <= face + w / 2 <= space.depth:
        y = min(max(face + w / 2, space.depth - KERB_FAR_M - w / 2), space.depth - kmin - w / 2)
    return space.length / 2 - centre, max(space.depth / 2, y), 0.0


def _reverse_s(space, car, ha, pts, kappa, lane_gaps=None, margin=CLEAR_M, kerb=None):
    lane_gaps = LANE_GAPS if lane_gaps is None else lane_gaps
    for k in (S_MARGIN * kappa, kappa):
        out = _reverse_s_k(space, car, ha, pts, k, lane_gaps, margin, kerb)
        if out is not None:
            return out
    return None


def _reverse_s_k(space, car, ha, pts, kappa, lane_gaps, margin, kerb=None):
    """Pull-up pose + two equal reverse arcs (in F). Returns (pre-park pose, poses (N,4)) or None."""
    xg, yg, _ = _goal_in_space(space, car)
    R = 1.0 / kappa
    for gap in lane_gaps:
        ys = -(gap + car.width / 2)
        dy = yg - ys
        if dy > 2 * R:
            continue
        th = math.acos(1.0 - dy / (2 * R))
        xs = xg + 2 * R * math.sin(th)
        a1 = _arc((xs, ys, 0.0), kappa, -R * th)
        a2 = _arc(tuple(a1[-1]), -kappa, -R * th)
        poses = np.vstack([a1, a2])
        if _clear(ha, pts, np.vstack([[xs, ys, 0.0], poses]), margin, kerb=kerb):
            return (xs, ys, 0.0), _with_dir(poses, -1)
    return None


def _obb_clearance(poses, pts, car):
    """Exact clearance of the car's RECTANGLE (not the planner's circle cover, which adds ~6 cm at each bumper) to points, per pose:
    distance from the body outline to the nearest point, negative inside. poses (N, 3), pts (M, 2) -> (N,)."""
    from .hybrid_astar import WORK
    front, rear, w, length, centre = _body(car)
    hl, hw = length / 2, w / 2
    if not len(pts):
        return np.full(len(poses), np.inf)
    WORK["pose_points"] += len(poses) * len(pts)
    c, sn = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx = pts[None, :, 0] - (poses[:, 0][:, None] + c * centre)
    dy = pts[None, :, 1] - (poses[:, 1][:, None] + sn * centre)
    ex = np.abs(c * dx + sn * dy) - hl
    ey = np.abs(-sn * dx + c * dy) - hw
    d = np.hypot(np.maximum(ex, 0.0), np.maximum(ey, 0.0)) + np.minimum(np.maximum(ex, ey), 0.0)
    return d.min(axis=1)


def _stroke(pose, kappa, direction, pts, car, clear, max_len=0.8, ds=0.01, max_turn=math.radians(80), kerb=None):
    """Drive from `pose` along a full-lock arc (or straight, kappa 0) in `direction` until the body would come closer than `clear` to an
    obstacle (KERB_CLEAR_M to the kerb), or the heading passes `max_turn`; returns the poses (possibly empty)."""
    P = _arc(pose, kappa, direction * max_len, ds)
    ok = (_obb_clearance(P, pts, car) >= clear) & (np.abs(P[:, 2]) <= max_turn) & (_kerb_room(P, car, kerb) >= KERB_CLEAR_M)
    n = len(P) if ok.all() else int(np.argmin(ok))
    return P[:n]


def _exit_arcs(pose, e, k, y_lane, pts, car, clear, kerb=None):
    """Leave the gap from `pose` with two full-lock arcs (the second straightening the car) in direction e (+1 forward, -1 backward),
    ending on the lane at y_lane with heading 0; a straight piece first when the car is already turned further than needed.
    Arc geometry: dy = (cos th1 - cos th0) / kappa -> the middle heading from cos th_m = (k (y_lane - y) + cos th + 1) / 2.
    Returns the poses or None when not clear."""
    x, y, th = pose
    cm = (k * (y_lane - y) + math.cos(th) + 1.0) / 2.0
    if not 0.05 <= cm <= 1.0:
        return None
    if cm <= math.cos(th) + 1e-9:
        thm = -e * math.acos(cm)
        a1 = _arc(pose, -k, e * abs(thm - th) / k) if abs(thm - th) > 1e-6 else np.empty((0, 3))
        p1 = tuple(a1[-1]) if len(a1) else pose
    else:                                                  # turned more than needed: straight on, then straighten
        if abs(math.sin(th)) < 1e-3:
            return None
        ln = (y_lane - y - (math.cos(th) - 1.0) / k) / (e * math.sin(th))
        if ln < 0:
            return None
        a1 = _arc(pose, 0.0, e * ln)
        p1, thm = tuple(a1[-1]), th
    a2 = _arc(p1, k, e * abs(thm) / k) if abs(thm) > 1e-6 else np.empty((0, 3))
    P = np.vstack([q for q in (a1, a2) if len(q)])
    return P if (_obb_clearance(P, pts, car) >= clear).all() and (_kerb_room(P, car, kerb) >= KERB_CLEAR_M).all() else None


def _exit_strokes(pose, car, pts, k, e, y_lane, clear=STROKE_CLEAR_M, kerb=None, max_strokes=16):
    """Leave a gap from `pose` (F) toward the lane at y_lane: two arcs straight out if they are clear, otherwise back up to the neighbour
    behind the exit first and alternate full-lock strokes (each ending `clear` before a neighbour / KERB_CLEAR_M before the kerb) until
    they are. Returns [(poses, direction)] in driving order, or None."""
    segs = []
    ex = _exit_arcs(pose, e, k, y_lane, pts, car, CLEAR_M, kerb)
    if ex is None:
        back = _stroke(pose, 0.0, -e, pts, car, clear, max_len=0.6, kerb=kerb)   # make room: back up to the neighbour behind the exit
        if len(back):
            segs.append((back, -e))
            pose = tuple(back[-1])
        d = e
        for _ in range(max_strokes):
            ex = _exit_arcs(pose, e, k, y_lane, pts, car, CLEAR_M, kerb)
            if ex is not None:
                break
            st = _stroke(pose, -d * e * k, d, pts, car, clear, kerb=kerb)
            if len(st) < 3:
                return None
            segs.append((st, d))
            pose, d = tuple(st[-1]), -d
    if ex is None:
        return None
    segs.append((ex, e))
    return segs


def _tight_gap(space, car, pts, kappa, e=1, clear=STROKE_CLEAR_M, lane_gaps=None, max_strokes=16, kerb=None):
    """Multi-stroke parking into a gap too short for the reverse S (Vorobieva, Glaser, Minoiu-Enache & Mammar, "Automatic parallel
    parking in tiny spots", IEEE T-ITS 2015): plan the EXIT from the parked pose - back up to the rear neighbour, then alternate
    full-lock forward and reverse strokes that each end `clear` before a neighbour / the kerb, until two arcs lead out to the lane - and
    drive it backwards. e = +1 exits forward (the usual reverse entry), e = -1 exits backward (forward entry, e.g. a dead end ahead).
    Exact rectangle clearance. Returns (pull-up pose, entry poses (N, 4)) in F, or None."""
    k = kappa
    xg, yg, _ = _goal_in_space(space, car)
    goal = (xg, yg, 0.0)
    for gap in (LANE_GAPS if lane_gaps is None else lane_gaps):
        y_lane = -(gap + car.width / 2)
        segs = _exit_strokes(goal, car, pts, k, e, y_lane, clear, kerb, max_strokes)
        if segs is None:
            continue
        ex = segs[-1][0]
        out = []                                          # the exit driven backwards: pieces in reverse order, each reversed, direction flipped
        for idx in range(len(segs) - 1, -1, -1):
            P, d = segs[idx]
            start = goal if idx == 0 else tuple(segs[idx - 1][0][-1])     # each piece ends where the exit piece began
            out.append(_with_dir(np.vstack([P[::-1][1:], [start]]), -d))
        E = np.vstack(out)
        return tuple(ex[-1]), E
    return None


def _approach(vehicle, pre, ha, pts, kappa, margin=CLEAR_M):
    """Forward Dubins curve from the vehicle to a point `RUN_IN_M` short of the pull-up pose, then a straight run-in to it (both in F):
    the follower arrives at the reversing point aligned with the strip, so the first arc of the S starts from the planned pose.
    Poses (N,4) or None."""
    if abs(vehicle[1] - pre[1]) < 0.02 and abs(math.remainder(vehicle[2] - pre[2], 2 * math.pi)) < math.radians(3):
        d = 1 if pre[0] >= vehicle[0] else -1                # level with the pull-up pose already: straight on / back to it
        n = max(2, int(abs(pre[0] - vehicle[0]) / DS))
        P = np.column_stack([np.linspace(vehicle[0], pre[0], n + 1)[1:], np.linspace(vehicle[1], pre[1], n + 1)[1:], np.full(n, pre[2])])
        return _with_dir(P, d) if _clear(ha, pts, np.vstack([vehicle, P]), margin) else None
    if vehicle[0] > pre[0] + 0.05:                      # already past the pull-up point (a driver just reverses back to it)
        P = dubins.sample_reverse(tuple(vehicle), tuple(pre), 1.0 / kappa)
        if P is None or not _clear(ha, pts, np.vstack([vehicle, P]), margin):
            return None
        turn = np.abs(np.diff(np.unwrap(np.concatenate([[vehicle[2]], P[:, 2]])))).sum()
        return _with_dir(P, -1) if turn <= math.radians(270) else None
    if pre[0] - vehicle[0] < MIN_DUBINS_M:            # nearly there: a Dubins curve over so short a distance loops - drive straight in, the follower settles the heading
        n = max(2, int((pre[0] - vehicle[0]) / DS))
        P = np.column_stack([np.linspace(vehicle[0], pre[0], n + 1)[1:], np.linspace(vehicle[1], pre[1], n + 1)[1:], np.full(n, pre[2])])
        return _with_dir(P, 1) if _clear(ha, pts, np.vstack([vehicle, P]), margin) else None
    run = float(np.clip(pre[0] - vehicle[0] - MIN_DUBINS_M, 0.0, RUN_IN_M))
    aim = (pre[0] - run, pre[1], pre[2])
    P = dubins.sample(tuple(vehicle), aim, 1.0 / kappa)
    if P is None:
        return None
    if run > 1e-3:
        n = max(2, int(run / DS))
        line = np.column_stack([np.linspace(aim[0], pre[0], n + 1)[1:], np.full(n, pre[1]), np.full(n, pre[2])])
        P = np.vstack([P, line])
    turn = np.abs(np.diff(np.unwrap(np.concatenate([[vehicle[2]], P[:, 2]])))).sum()
    if turn > math.radians(270) or not _clear(ha, pts, np.vstack([vehicle, P]), margin):
        return None
    return _with_dir(P, 1)


def _space_obstacles(strip, space, points_w):
    """Obstacle points in F: the synthetic neighbour blocks plus the scan; sets space.face (the neighbours' road-side faces, from the
    scan alone)."""
    space.face, space.kerb_seen = None, False
    f = None
    if points_w is not None and len(points_w):
        f = strip.to_frame(space, np.column_stack([points_w, np.zeros(len(points_w))]))
        # a kerb the LiDAR sees: returns along the kerb line inside the gap (not at its ends: a neighbour's kerb-side face lies there);
        # where it runs inside the drawn lane, it is the kerb (a lane drawn a few cm too deep along a wall parked into the wall)
        g0, g1 = space.gap[0] - space.s0 + 0.05, space.gap[1] - space.s0 - 0.05
        along = (f[:, 0] > g0) & (f[:, 0] < g1) & (np.abs(f[:, 1] - space.depth) < KERB_SEEN_M + 0.03)
        space.kerb_seen = int(along.sum()) >= 10
        if space.kerb_seen:
            space.depth = float(min(space.depth, np.median(f[along, 1])))
        space.face = _neighbour_face(space, f[:, :2])
    pts = _obstacles(space)
    if f is not None:
        # returns at / behind the kerb line are the kerb itself (a wall or board the LiDAR sees): the kerb clearance applies to them, not
        # a neighbour's (a lane drawn along a wall found no manoeuvre - 4 cm to the wall and the S's swing)
        keep = f[:, 1] < space.depth - KERB_SEEN_M
        pts = np.vstack([pts, f[keep, :2]])
    return pts


def _plan_space(strip, space, car, vehicle_w, points_w, kappa, approach=True):
    """The manoeuvre into `space` (reverse S, the roomier clearance first; a tiny gap: the multi-stroke exit plan reversed) and, with
    `approach`, the way from the car to its pull-up pose. Returns (poses (N, 3), directions (N,), goal (x, y) in F, pull-up) in the
    working frame, the pull-up pose (working frame) and whether the path starts at the car (False: it starts at the pull-up pose - no
    approach asked for, or none that a single curve from one scan can give; the car then gets there with the click-to-go planner).
    None = no manoeuvre."""
    ha = HybridAStar(car, kappa_max=kappa, margin=CLEAR_M)
    veh = strip.to_frame(space, np.asarray([vehicle_w], float))[0]
    pts = _space_obstacles(strip, space, points_w)
    xg, yg, _ = _goal_in_space(space, car)
    # the kerb line the clearance checks use: the path keeps KERB_CLEAR_M to it, or - a goal in line with neighbours standing closer - as
    # much as the goal itself less 5 mm (the kerb is moved out by the difference), never less
    goal_gap = space.depth - yg - car.width / 2
    kclear = KERB_CLEAR_SEEN_M if space.kerb_seen else KERB_CLEAR_M
    kerb = space.depth + KERB_CLEAR_M - min(kclear, goal_gap - 0.005)   # (the checks keep KERB_CLEAR_M to this line)
    lanes = LANE_GAPS
    own = -veh[1] - car.width / 2                         # the car's own gap to the road edge, if it drives alongside already
    if approach and 0.06 <= own <= 0.45 and abs(veh[2]) < math.radians(10):
        lanes = (own,) + LANE_GAPS
    cands = []
    for margin in (COMFORT_CLEAR_M, CLEAR_M):            # a roomier S first: the brake gate stops a car that passes a neighbour closely
        sol = _reverse_s(space, car, ha, pts, kappa, lane_gaps=lanes, margin=margin, kerb=kerb)
        if sol is not None:
            cands.append((sol, margin))
            if not approach:
                break
    if not cands:
        for e in (1, -1):                                  # tiny spot: multi-stroke exit plan, reversed (usual), then forward entry
            sol = _tight_gap(space, car, pts, kappa, e, lane_gaps=lanes, kerb=kerb)
            if sol is not None:
                cands.append((sol, CLEAR_M))
                if not approach:
                    break
    if not cands:
        return None
    if approach:
        for (pre, park), margin in cands:
            appr = _approach(veh, pre, ha, pts, kappa, margin)
            if appr is not None:
                path = np.vstack([appr, park])
                return strip.from_frame(space, path[:, :3]), path[:, 3], (xg, yg), tuple(strip.from_frame(space, np.array([pre]))[0]), True
    (pre, park), _ = cands[0]
    return strip.from_frame(space, park[:, :3]), park[:, 3], (xg, yg), tuple(strip.from_frame(space, np.array([pre]))[0]), False


def _plan_slot(strip, sp, car, pose, points, kappa, click, approach=True):
    """ParkingPlan into slot `sp`, or None when there is no manoeuvre."""
    out = _plan_space(strip, sp, car, pose, points, kappa, approach)
    if out is None:
        return None
    poses, dirs, (xg, yg), pull_up, from_car = out
    gw = strip.from_frame(sp, np.array([[xg, yg, 0.0]]))[0]
    box = strip.from_frame(sp, np.array([[0.0, 0.0, 0.0], [sp.length, 0.0, 0.0], [sp.length, sp.depth, 0.0],
                                          [0.0, sp.depth, 0.0]]))[:, :2]
    L = car.front_x - car.rear_x
    # the planned free distances from the parked car's bumpers to the neighbours (rear, front): the final position is measured against them
    req = {"strip": (strip.pts.copy(), strip.side, strip.depth), "click": (float(click[0]), float(click[1])),
           "gaps_planned": (float(sp.mid - L / 2 - sp.gap[0]), float(sp.gap[1] - sp.mid - L / 2))}
    pu = tuple(float(v) for v in pull_up)
    return ParkingPlan(sp.id, (float(gw[0]), float(gw[1]), math.degrees(gw[2])), np.column_stack([poses, dirs]), box, req, pu, from_car)


def plan_parallel(strip, click, car, pose=(0.0, 0.0, 0.0), points=None, kappa=KAPPA_PARK, tries=None, approach=True):
    """Find the free gaps on the strip from `points` (LiDAR returns, working frame), place a slot in each viable gap as near to `click`
    (any point; it is projected onto the strip) as the gap allows, and plan the reverse parallel park into the nearest slot that has a
    manoeuvre (the next nearest otherwise; `tries` limits how many). A click that cannot be parked at (on a car, in a gap too short)
    gets the nearest viable slot - never a refusal while one exists. approach=False: the path starts at the slot's pull-up pose
    (ParkingPlan.pull_up, starts_at_car False) and the car gets there with the click-to-go planner. Returns a ParkingPlan (NO_PLAN when
    no viable gap)."""
    s_click, _ = strip.project(click)
    for sp in choose_slots(find_gaps(strip, points), s_click, car, strip.depth)[:tries]:
        plan = _plan_slot(strip, sp, car, pose, points, kappa, click, approach)
        if plan is not None:
            return plan
    return NO_PLAN


def plan_pull_out(strip, car, points=None, kappa=KAPPA_PARK, lane_gaps=(0.16, 0.24, 0.10, 0.32)):
    """The car (working frame = the vehicle now) stands parked in the lane: the way out onto the road - the exit strokes of the tiny-gap
    planner (Vorobieva et al. 2015) run forward from where it stands, as a driver pulls out of a space (forward first, backward if the
    front is boxed in). Returns poses (N, 4) x, y, theta, direction in the working frame, or None when the car is not in the lane (its
    body centre on the road side of the road edge) or no exit is found."""
    front, rear, w, length, centre = _body(car)
    s_c, y_c = strip.to_strip(np.array([[centre, 0.0]]))
    if y_c[0] <= 0.0 or not -0.3 <= s_c[0] <= strip.length + 0.3:
        return None
    sp = ParkingSpace("pull-out", s_c[0] - 0.01, s_c[0] + 0.01, strip.depth)
    if abs(strip.to_frame(sp, np.zeros((1, 3)))[0, 2]) > math.pi / 2:      # parked facing against the strip: work on it reversed
        strip = ParkingStrip(strip.pts[::-1], -strip.side, strip.depth)
        s_c, _ = strip.to_strip(np.array([[centre, 0.0]]))
    gaps = find_gaps(strip, points)
    g = min(gaps, key=lambda q: 0.0 if q[0] <= s_c[0] <= q[1] else min(abs(q[0] - s_c[0]), abs(q[1] - s_c[0])))
    sp = ParkingSpace("pull-out", g[0], g[1], strip.depth, gap=g)
    veh = tuple(strip.to_frame(sp, np.zeros((1, 3)))[0])
    pts = _space_obstacles(strip, sp, points)
    # standing nearer the kerb than KERB_CLEAR_M already (parked in line with neighbours close to it): the strokes may keep the car's
    # own clearance to it, never less (30 Sep: parked 2.5 cm off the kerb, no pull-out was found)
    r0 = float(_kerb_room(np.array([veh]), car, sp.depth)[0])
    kerb = sp.depth + max(0.0, KERB_CLEAR_M - (r0 - 0.005))
    for clear in (PULL_OUT_CLEAR_M, STROKE_CLEAR_M):
        for e in (1, -1):
            for gap in lane_gaps:
                segs = _exit_strokes(veh, car, pts, kappa, e, -(gap + w / 2), clear=clear, kerb=kerb)
                if segs is not None:
                    P = np.vstack([_with_dir(Q, d) for Q, d in segs])
                    return np.column_stack([strip.from_frame(sp, P[:, :3]), P[:, 3]])
    return None


def refine_parallel(strip, click, slot, car, points=None, kappa=KAPPA_PARK):
    """The car is alongside the slot it chose from afar: find the gaps again in the close-range scan (from afar the parked cars hide parts
    of each other, so a gap can look longer than it is, or a car can hide behind another), choose again by the same rule among the gaps
    within REFINE_REACH_M of the planned slot and re-plan from here (working frame = the vehicle now); if none has a manoeuvre, the
    nearest viable gap anywhere.
    `slot` = the planned slot's corners (4, 2). The lane itself (the kerb) is not visible to a 2D LiDAR: its position comes from the
    car's pose, so a lateral drift of the pose is not corrected here (tried: aligning to the neighbours' road-side faces was off by up
    to 7 cm between a far and a near view - worse than the drift). Returns a ParkingPlan."""
    s_corners, _ = strip.to_strip(np.asarray(slot, float).reshape(-1, 2))
    a, b = float(s_corners.min()), float(s_corners.max())
    s_click, _ = strip.project(click)
    near = [g for g in find_gaps(strip, points) if g[1] > a - REFINE_REACH_M and g[0] < b + REFINE_REACH_M]   # the close-range scan
    for sp in choose_slots(near, s_click, car, strip.depth):                                           # same rule as from afar
        plan = _plan_slot(strip, sp, car, (0.0, 0.0, 0.0), points, kappa, click)
        if plan is not None:
            return plan
    return plan_parallel(strip, click, car, points=points, kappa=kappa)     # (the path may then start at a pull-up pose: ParkingPlan.pull_up)


# ------------------------------------------------------------------------------------------------ perpendicular (ArUco)
def rear_camera():
    """The rear camera as tests/tools model it (adas.markers.Camera, looking backwards). Pitch 8 deg: with 32 deg a floor-level
    wall marker (z 9 cm, camera z 12 cm) is above the image at every distance (sim/parking_validation.py, V5); at 8 deg a 10 cm
    marker is seen from ~0.3 to ~2.4 m. Aim the real camera the same way (MEASURE)."""
    from .markers import Camera
    return Camera(x=-0.05, z=0.12, pitch_deg=8.0, yaw_deg=180.0, hfov_deg=70.0)


def detect_rear_marker(frame_bgr, marker_id, size_m, cam=None):
    """The rear-wall marker in a camera frame: MarkerObs in the vehicle frame (position, facing) or None."""
    from .markers import detect_image
    obs = [o for o in detect_image(frame_bgr, cam or rear_camera(), {marker_id: size_m}) if o.id == marker_id]
    return max(obs, key=lambda o: o.size_px) if obs else None


def fuse_markers(obs_list):
    """One pose from several observations of the same marker (weight = apparent size): averages out pixel noise."""
    w = np.array([max(o.size_px, 1e-3) for o in obs_list], float)
    w /= w.sum()
    x = float(sum(wi * o.x for wi, o in zip(w, obs_list)))
    y = float(sum(wi * o.y for wi, o in zip(w, obs_list)))
    facing = math.atan2(sum(wi * math.sin(o.facing) for wi, o in zip(w, obs_list)),
                        sum(wi * math.cos(o.facing) for wi, o in zip(w, obs_list)))
    return obs_list[0].__class__(id=obs_list[0].id, corners=obs_list[0].corners, x=x, y=y, z=obs_list[0].z, facing=facing)


def _open_side(poses, m, n, car, gap):
    """Rear corners of the body stay on the open side of the wall plane (through m, normal n)."""
    rear, hw = car.rear_x, car.width / 2
    for dy in (-hw, hw):
        cx = poses[:, 0] + rear * np.cos(poses[:, 2]) - dy * np.sin(poses[:, 2])
        cy = poses[:, 1] + rear * np.sin(poses[:, 2]) + dy * np.cos(poses[:, 2])
        if ((cx - m[0]) * n[0] + (cy - m[1]) * n[1]).min() < gap - 0.01:
            return False
    return True


def wall_from_lidar(points, m, n, along=0.6, band=0.15, min_points=12, max_turn_deg=25.0):
    """The rear wall as the LiDAR sees it around the marker: a line fitted (PCA) to the returns within `along` of the marker along the
    wall and `band` of its plane. Returns (point on the wall level with the marker, unit normal toward the bay's mouth) or None when the
    wall is not seen well enough (too few returns, not a line, or turned more than max_turn_deg from the marker's facing). A 2D LiDAR
    measures the wall's direction and distance far better than a marker's pose from four pixel corners (PnP at 2 m: degrees of facing
    error, and a marker can be mounted askew)."""
    if points is None or not len(points):
        return None
    P = np.asarray(points, float).reshape(-1, 2)
    t = np.array([-n[1], n[0]])
    d, a = (P - m) @ n, (P - m) @ t
    sel = (np.abs(d) < band) & (np.abs(a) < along)
    if int(sel.sum()) < min_points:
        return None
    Q = P[sel]
    c = Q.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((Q - c).T))
    for _ in range(3):                                       # refit on the inliers: neighbours' rear faces a few cm in front of the
        r = (Q - c) @ v[:, 0]                                # wall fall out (iteratively reweighted / trimmed least squares)
        keep = np.abs(r) < 0.025
        if int(keep.sum()) < min_points:
            return None
        Q = Q[keep]
        c = Q.mean(axis=0)
        w, v = np.linalg.eigh(np.cov((Q - c).T))
    if w[0] > 0.012 ** 2 or w[1] < 25 * w[0]:               # still thick or not elongated: not a clean wall line
        return None
    nn = v[:, 0] * (1.0 if v[:, 0] @ n >= 0 else -1.0)      # normal on the marker's side
    if math.degrees(math.acos(float(np.clip(nn @ n, -1.0, 1.0)))) > max_turn_deg:
        return None
    mm = m - ((m - c) @ nn) * nn                             # the marker's position projected onto the fitted wall
    return mm, nn


def plan_perpendicular(marker, car, pose=(0.0, 0.0, 0.0), space_id=None, stop_gap=STOP_GAP_M, kappa=KAPPA_PARK, align_m=ALIGN_M,
                       points=None):
    """Reverse into a bay whose rear wall carries `marker` (MarkerObs, vehicle frame at observation time; `pose` = the vehicle
    pose in the working frame then). The car ends on the marker's centre line, nose out, rear bumper `stop_gap` from the wall. With
    `points` (LiDAR, working frame) the wall's line and distance come from the LiDAR (wall_from_lidar), the bay's place from the marker;
    on the last reverse leg the stop follows the wall as measured (AutoNav._wall_stop)."""
    c, s = math.cos(pose[2]), math.sin(pose[2])
    m = np.array([pose[0] + c * marker.x - s * marker.y, pose[1] + s * marker.x + c * marker.y])
    f = _wrap(marker.facing + pose[2])
    n = np.array([math.cos(f), math.sin(f)])
    wall = wall_from_lidar(points, m, n)
    if wall is not None:                                  # camera: WHICH bay (the marker's place on the wall); LiDAR: the wall's line
        m, n = wall
        f = math.atan2(n[1], n[0])
    goal = np.array([*(m + n * (stop_gap + abs(car.rear_x))), f])
    pre = np.array([*(goal[:2] + n * align_m), f])
    stage = np.array([*(goal[:2] + n * (align_m + 0.45)), f])
    ha = HybridAStar(car, kappa_max=kappa, margin=CLEAR_M)
    pts = np.asarray(points, float).reshape(-1, 2) if points is not None and len(points) else np.empty((0, 2))
    start = np.array(pose, float)
    tail = _with_dir(np.column_stack([np.array([pre[:2] - n * d for d in np.arange(DS, align_m + 1e-9, DS)]),
                                      np.full(int(round(align_m / DS)), f)]), -1)
    tail[-1, :3] = goal
    ok = lambda p: _open_side(p, m, n, car, stop_gap) and (len(pts) == 0 or _clear(ha, pts, p, CLEAR_M))
    path = None
    R = 1.0 / kappa
    P = dubins.sample_reverse(tuple(start), tuple(pre), R)                       # one reverse move
    if P is not None and np.abs(np.diff(np.unwrap(np.concatenate([[start[2]], P[:, 2]])))).sum() < math.radians(270) \
            and ok(P):
        path = np.vstack([_with_dir(P, -1), tail])
    else:                                                                        # forward to a staging pose, then reverse
        F = dubins.sample(tuple(start), tuple(stage), R)
        if F is not None and np.abs(np.diff(np.unwrap(np.concatenate([[start[2]], F[:, 2]])))).sum() < math.radians(270) \
                and ok(F):
            back = _with_dir(np.column_stack([np.array([stage[:2] - n * d for d in np.arange(DS, 0.45 + 1e-9, DS)]),
                                              np.full(int(round(0.45 / DS)), f)]), -1)
            path = np.vstack([_with_dir(F, 1), back, tail])
    sid = marker.id if space_id is None else space_id
    g = (float(goal[0]), float(goal[1]), math.degrees(goal[2]))
    # the rear wall (through the marker, normal toward the bay): the LiDAR measures it on the last reverse leg and the stop follows it
    req = {"wall": (m.tolist(), n.tolist()), "stop_gap": float(stop_gap)}
    return ParkingPlan(sid, g, path, None, req) if path is not None else ParkingPlan(sid, g, None)
