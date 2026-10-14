"""Parking validation framework: scenario suite + Monte Carlo for both Parking Simulator worlds (TODO V1-V3).

    python -m sim.parking_validation suite            # every scenario once on the nominal twin (seed 0)
    python -m sim.parking_validation mc [runs] [workers]   # every scenario `runs` times with randomised sensors / steering
    python -m sim.parking_validation quick             # a handful of cases (smoke test)
    -> reports/parking_validation/results.csv, summary.csv, summary.json (then `python -m sim.parking_dashboard`)

What is executed. The planners (adas/park_select.py: `plan_perpendicular` from the rear-wall ArUco marker, `plan_parallel` from a
click on the parking strip) return a `ParkingPlan`; nothing on the car consumes it yet, so the plan is handed to the relay's OWN
click-to-go follower (`AutoNav`: pure pursuit, speed profile, straightening strokes) exactly as `AutoNav.start` would after its
search, and the whole relay stack (RelayAssists, PathGate brake, RelaySpeed, ThrottleSmoother) drives the twin
(`sim.relay_scenarios.run`) with the operator holding the throttle. Planning code is not modified; only its output is injected.

Sensing. Perpendicular: the sim camera model (`MarkerSensor`, rear camera of `adas.park_select.rear_camera`) sees the marker
for 1.2 s while the car is still, the observations are fused (`fuse_markers`) and planned from. Parallel: one continuous parking lane
(strip = road edge + lane depth, what the HMI publishes) with cars parked at arbitrary spacing - no spaces; the car finds the gaps
from its own LiDAR scan.

Ground truth for the error metrics is computed from the world geometry, never from the planner's goal:
  perpendicular  rear axle on the bay's centre line, rear bumper 6 cm off the wall, heading = the bay axis (nose out)
  parallel       car body in the slot the same rule gives on the TRUE gaps (nearest viable gap to the click, slot slid toward the
                 click, KERB_GAP_M off the kerb), heading = the strip's; it must end inside that gap
Success = a plan was made and driven to a stop, 0 contacts, pose error <= 5 cm and heading error <= 5 deg (user target, 30 Sep;
target), the car inside the space for parallel; scenarios that must be refused (space too small, marker not visible) succeed when
the system refuses. `collisions` is 0 / 1: the twin stops the drive at the first contact.

Monte Carlo. Each run draws a randomised car (`sim.twin_intent_data.randomise`: steering gain, servo offset, motor lag, command
delay, LiDAR noise / dropout / yaw error), a start pose jitter, camera pixel noise, extra LiDAR range noise and (for the
drift group) localisation drift injected into the follower's pose estimate.
"""
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "..", "reports", "parking_validation")

POS_TOL_M, HEAD_TOL_DEG = 0.05, 5.0          # user, 30 Sep: parking to under 5 cm, the heading right
KERB_MIN_CM, KERB_MAX_CM = 1.0, 8.0            # a parallel-parked body: clear of the kerb line (never clipping it) and close to it
T_OBSERVE = (0.3, 1.5)                 # s: the still car watches the marker
T_MAX = 70.0
STOP_GAP = 0.06                        # m rear bumper to the wall (adas.park_select.STOP_GAP_M; ground truth uses the spec value)
WALL_X = -3.0                          # sim.parking_worlds.perpendicular
BAY_W = 0.46
Y_EDGE, Y_ROAD0, Y_FACE, CAR_W = 0.55, -0.85, 1.45, 0.20     # street geometry (as sim.parking_worlds.parallel_street)
STRIP_X0 = 0.9                         # the parking lane starts 0.9 m ahead of the car's start


def legacy_rear_camera():
    """The rear camera as configured before V5 (pitch 32 deg): the floor-level wall marker of park_perp (z 9 cm, camera z 12 cm) is
    above the image at every distance, so nothing can be parked from it (perp_camera_pitch32 keeps that finding as a regression test)."""
    from adas.markers import Camera
    return Camera(x=-0.05, z=0.12, pitch_deg=32.0, yaw_deg=180.0, hfov_deg=70.0)


def _wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


# ------------------------------------------------------------------------------------------------ scenario definitions
def perpendicular_cases():
    """(scenario, group, params) for the perpendicular world. Defaults: marker 2.0 m from the car's rear axle, 16 cm marker."""
    base = dict(camera="rear", marker_deg=0.0, dist_m=2.0, pixel_noise=0.6, detect_prob=0.96, frame_drop=0.0, occluder=None,
                lidar_sigma=0.0, drift_cm_s=0.0, drift_deg_s=0.0, start_y=0.0, start_deg=0.0)
    out = [("perp_perfect", dict(base, pixel_noise=0.0, detect_prob=1.0))]
    out.append(("perp_camera_pitch32", dict(base, camera="pitch32", pixel_noise=0.0, detect_prob=1.0, expect="refuse")))   # the old camera aim
    out.append(("perp_nominal", dict(base)))
    for d in (5, 10, 20):
        out.append((f"perp_rot{d}", dict(base, marker_deg=float(d))))
    for f in (0.25, 0.5, 0.75):
        out.append((f"perp_drop{int(f * 100)}", dict(base, frame_drop=f)))
    out.append(("perp_occluder_side", dict(base, occluder="side")))        # a box beside the path: the marker stays visible
    out.append(("perp_occluder_full", dict(base, occluder="full", expect="refuse")))   # a box on the line of sight: must not drive
    for d in (1.2, 1.6, 2.4, 2.8, 3.0, 4.4):   # the 16 cm marker is over the camera model's 18 px detection limit up to ~4.1 m;
        out.append((f"perp_dist{d:.1f}", dict(base, dist_m=d, expect="refuse" if d >= 4.3 else "park")))   # beyond: refuse
    for s in (0.01, 0.02, 0.04):
        out.append((f"perp_lidar{int(s * 100)}cm", dict(base, lidar_sigma=s)))
    out.append(("perp_drift_mild", dict(base, drift_cm_s=2.0, drift_deg_s=0.5)))
    out.append(("perp_drift_strong", dict(base, drift_cm_s=5.0, drift_deg_s=1.5)))
    out.append(("perp_offset_left", dict(base, start_y=0.25, start_deg=8.0)))
    out.append(("perp_offset_right", dict(base, start_y=-0.25, start_deg=-8.0)))
    for _, p in out:
        p.setdefault("expect", "park")
    return [(n, "perpendicular", p) for n, p in out]


def street_spec(i):
    """Street i (0..): a continuous parking lane with cars parked at arbitrary spacing - no spaces (user, 30 Sep: "no dedicated
    rectangles"). Returns [(kind, length_m)] along the strip, kind "car" or "gap", with car lengths 0.30-0.46 m and gaps from a
    mix of too short (0.40-0.88), tight (0.95-1.10), normal (1.10-1.45) and long (1.5-2.2); at least one viable gap."""
    rng = np.random.default_rng(1000 + i)
    segs = [("gap", float(rng.uniform(0.2, 0.6)))]
    for _ in range(int(rng.integers(4, 7))):
        segs.append(("car", float(rng.uniform(0.30, 0.46))))
        r = rng.random()
        g = (rng.uniform(0.40, 0.88) if r < 0.35 else rng.uniform(0.95, 1.10) if r < 0.55 else rng.uniform(1.10, 1.45) if r < 0.85
             else rng.uniform(1.5, 2.2))
        segs.append(("gap", float(g)))
    if not any(k == "gap" and ln >= 1.0 for k, ln in segs[1:-1]):
        segs[2] = ("gap", float(rng.uniform(1.0, 1.25)))
    segs.append(("car", float(rng.uniform(0.30, 0.46))))
    return segs


def parallel_cases(n_streets=24):
    out = []
    policies = ("gap", "car", "short_gap", "beyond")
    for i in range(n_streets):
        for lane, y in (("a", -0.15), ("b", -0.35)):       # a = the shipped start (0.70 m off the road edge), b = 0.90 m
            out.append((f"par_street{i:02d}{lane}", dict(street=i, click=policies[i % 4], segs=street_spec(i), start_y=y)))
    car = lambda ln=0.40: ("car", ln)
    gap = lambda ln: ("gap", ln)
    tgt = lambda ln: ("target", ln)                                            # the gap the named case clicks into
    named = {
        "par_tight": [gap(0.3), car(), tgt(0.97), car(), gap(0.5)],               # just above the minimum (0.93 m)
        "par_standard": [gap(0.3), car(), tgt(1.2), car(), gap(0.5)],
        "par_wide": [gap(0.3), car(), tgt(1.9), car(), gap(0.5)],
        "par_long_click_end": [gap(0.3), car(), gap(2.6), car(), gap(0.5)],       # long gap: park where clicked, not in its middle
        "par_too_short": [gap(0.3), car(), tgt(0.55), car(), gap(0.5)],             # below the minimum (car + 0.30 m = 0.63 m)
        "par_all_full": [gap(0.3), car(), tgt(0.5), car(), gap(0.55), car(), gap(0.4)],
        "par_skip_short_gap": [gap(0.3), car(), gap(1.2), car(), tgt(0.5), car(), gap(1.15), car(), gap(0.4)],
        "par_edge_first": [tgt(1.15), car(), car(0.36), gap(0.5)],               # the viable gap is the strip's first stretch
        "par_edge_last": [gap(0.3), car(), gap(0.5), car(), tgt(1.15)],          # ... the last one (the road runs on)
        "par_dead_end": [gap(0.3), car(), gap(0.5), car(), tgt(1.15)],           # a wall right after the gap: forward entry needed
        "par_click_car": [gap(0.3), car(), gap(1.2), car(0.44), gap(1.2), car(), gap(0.4)],
        "par_shallow_lane": [gap(0.3), car(), tgt(1.2), car(), gap(0.5)],
    }
    clicks = {"par_click_car": "car", "par_long_click_end": "gap_end"}
    for name, segs in named.items():
        p = dict(street=None, click=clicks.get(name, "target"), segs=segs, start_y=-0.15, dead_end=name == "par_dead_end",
                 expect="refuse" if name in ("par_too_short", "par_all_full", "par_shallow_lane", "par_dead_end") else "park")
        # (par_dead_end: leaving forward is blocked and a forward entry swings the long front into the kerb with this steering: the car
        # must refuse, not try - a known limitation)
        if name == "par_shallow_lane":
            p["depth"] = 0.26                                                   # narrower than the car + 8 cm: must refuse
        out.append((name, p))
    for name, segs in (("par_wide_far_lane", named["par_wide"]), ("par_tight_far_lane", named["par_tight"])):
        out.append((name, dict(street=None, click="target", segs=segs, start_y=-0.35, expect="park")))
    for _, p in out:
        p.setdefault("expect", "park")
    return [(n, "parallel", p) for n, p in out]


def all_cases():
    return perpendicular_cases() + parallel_cases()


# ------------------------------------------------------------------------------------------------ worlds
def build_perpendicular(p, rng_jitter=None):
    from sim.hw_worlds import _finish
    from sim.parking_worlds import perpendicular
    from sim.world import Box, Marker
    world, _ = perpendicular(23)
    for o in world.objects:
        if isinstance(o, Marker):
            o.yaw = math.radians(p["marker_deg"])
    if p["occluder"]:
        # a box (a bag, a person's legs): "full" sits on the rear camera's line of sight to the marker (hides it), "side" beside
        # the path (visible marker, an obstacle the plan has to keep clear of)
        y = 0.0 if p["occluder"] == "full" else 0.30
        world.add(Box(WALL_X + 0.9, y, 0.12, 0.14, 0.0, height=0.25, style="box", colour="#b45309"))
    _finish(world)
    return world


def build_parallel(p, seed=0):
    """A street along +x (road y in [-0.85, 0.55], the parking lane on the left up to the kerb, sidewalk + facade behind) with the
    cars of `p['segs']` parked a few cm off the kerb, slightly skewed. Returns world, start pose, strip (road edge, lane depth) and
    the true cars as [(s0, s1)] along the strip (their footprints' extent); `world.layout["target"]` = a named case's target gap."""
    from adas.park_select import LANE_DEPTH_M, ParkingStrip
    from sim.hw_worlds import _finish
    from sim.world import Box, FloorLine, Wall, World
    rng = np.random.default_rng(seed + 55)
    depth = p.get("depth", LANE_DEPTH_M)
    xs, y_edge = STRIP_X0, Y_EDGE                  # the lane starts ahead of the car (x = 0) and ends after the last segment
    y_kerb = y_edge + depth
    x = xs
    w = World()
    cars, target = [], None
    for kind, ln in p["segs"]:
        if kind == "car":
            off, yaw = float(rng.uniform(0.02, 0.08)), math.radians(float(rng.uniform(-3, 3)))
            cx, cy = x + ln / 2, y_kerb - off - CAR_W / 2
            w.add(Box(cx, cy, ln, CAR_W, yaw, height=0.12, style="car", colour="#8fa3bf"))
            c, sn = abs(math.cos(yaw)), abs(math.sin(yaw))
            half = 0.5 * (ln * c + CAR_W * sn)                                 # the footprint's extent along the strip
            cars.append((cx - half - xs, cx + half - xs))
        elif kind == "target":
            target = (x - xs, x + ln - xs)
        x += ln
    xe = x
    x0w, x1w = -1.2, (xe + 0.02 if p.get("dead_end") else xe + 1.5)           # the road runs on past the lane unless a dead end
    w.add(Wall(x0w, Y_ROAD0, x1w, Y_ROAD0))
    w.add(Wall(x0w, Y_FACE, x1w, Y_FACE))
    w.add(Wall(x0w, Y_ROAD0, x0w, Y_FACE))
    w.add(Wall(x1w, Y_ROAD0, x1w, Y_FACE))
    w.add(FloorLine([[xs, y_edge], [xe, y_edge]], width=0.02))
    strip = ParkingStrip([[xs, y_edge], [xe, y_edge]], 1, depth)
    w.layout = {"x0": xs, "target": target}
    _finish(w)
    return w, (0.0, float(p.get("start_y", -0.15)), 0.0), strip, cars


# ------------------------------------------------------------------------------------------------ helpers
def _to_start_frame(xy_th, start):
    """World (x, y, th) -> the start vehicle frame (the relay's working frame)."""
    x0, y0, t0 = start
    c, s = math.cos(-t0), math.sin(-t0)
    dx, dy = xy_th[0] - x0, xy_th[1] - y0
    return (c * dx - s * dy, s * dx + c * dy) + ((_wrap(xy_th[2] - t0),) if len(xy_th) > 2 else ())


def _from_start_frame(xy, start):
    x0, y0, t0 = start
    c, s = math.cos(t0), math.sin(t0)
    return (x0 + c * xy[0] - s * xy[1], y0 + s * xy[0] + c * xy[1])


def _gear_changes(dirs):
    d = [x for x in dirs if x != 0]
    return int(sum(1 for a, b in zip(d, d[1:]) if a != b))


def _path_metrics(trace):
    """Distance driven, gear changes actually made (velocity sign flips beyond 2 cm/s) from the true trace."""
    tr = np.array([(r[1], r[2], r[3], r[4]) for r in trace]) if trace else np.zeros((0, 4))
    if len(tr) < 2:
        return 0.0, 0
    length = float(np.hypot(np.diff(tr[:, 0]), np.diff(tr[:, 1])).sum())
    sgn = np.sign(np.where(np.abs(tr[:, 3]) > 0.02, tr[:, 3], 0.0))
    return length, _gear_changes(sgn.tolist())


def _body_corners(pose, p):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    return [(x + c * lx - s * ly, y + s * lx + c * ly) for lx in (p.rear_x, p.front_x) for ly in (-p.width / 2, p.width / 2)]


# ------------------------------------------------------------------------------------------------ one run
def run_case(job):
    """job = (name, group, params, seed, mc) -> one flat result row (plus a decimated trace under 'trace')."""
    name, group, params, seed, mc = job
    import sim.relay_scenarios as rs
    rng = np.random.default_rng(seed * 7919 + 13)
    rs.RANDOMISE["level"], rs.RANDOMISE["seed"] = (1.0, seed) if mc else (None, 0)
    t_wall = time.perf_counter()
    if group == "perpendicular":
        row = _run_perpendicular(name, params, seed, mc, rng)
    else:
        row = _run_parallel(name, params, seed, mc, rng)
    row.update(name=name, group=group, seed=seed, mc=int(bool(mc)), wall_s=round(time.perf_counter() - t_wall, 1))
    return row


def _make_fault(sigma, dropout=0.0, seed=0):
    if sigma <= 0 and dropout <= 0:
        return None
    r = np.random.default_rng(seed + 991)

    def fault(t, pts):
        out = []
        for a, d in pts:
            if dropout and r.random() < dropout:
                continue
            out.append((a, round(float(d + r.normal(0.0, sigma)), 3)))
        return out
    return fault


def _drift_hook(assist, rate_m_s, rate_deg_s, dt=0.05):
    """Localisation drift: the follower's pose estimate is pushed off a little every tick (the scan matcher pulls it back at
    every scan; what remains is what the follower steers by)."""
    if rate_m_s <= 0 and rate_deg_s <= 0:
        return
    x, y, th = assist.nav_pose
    assist.nav_pose = (x + rate_m_s * dt * 0.7, y + rate_m_s * dt * 0.7, th + math.radians(rate_deg_s) * dt)


def _execute(world, start, seed, params, plan_fn, ideal_fn, mc, rng, sensing=None, fault=None):
    """Common closed loop. plan_fn(assist, pts, st) is called once when planning is due and returns
    ('plan', path, goal_xyh, info) with everything in the start frame, or ('refuse', reason, None, info)."""
    from sim.relay_scenarios import run
    st = {"pose": start, "phase": "wait", "t_plan": None, "planned": None, "info": {}, "refused": None, "sensed": []}
    st["done"] = False

    def operator(t, x, y, th, v):
        st["pose"] = (x, y, th)
        holding = st["phase"] == "drive"
        return 0.0, (150.0 if holding else 0.0)

    def hook(t, assist, pts, seq):
        if st["phase"] == "wait":
            if sensing is not None:
                sensing(t, st)
            if plan_fn is not None and t >= T_OBSERVE[1] and seq >= 2:
                info = plan_fn(assist, pts, st)          # gives the relay its parking command (the car's own planning path)
                st["info"] = info
                if info.get("refused"):
                    st["refused"], st["phase"] = info["refused"], "refused"
                else:
                    st["t_plan"], st["phase"] = t, "drive"
        elif st["phase"] == "drive":
            nav = assist.nav
            if st["planned"] is None and nav.state == "driving" and nav.path is not None:
                st["planned"] = nav.path.copy()
            if nav.state == "idle":
                st["phase"] = "finished" if st["planned"] is not None else "refused"
                st["refused"] = nav.msg
            else:
                _drift_hook(assist, params.get("drift_cm_s", 0.0) / 100.0, params.get("drift_deg_s", 0.0))

    r = run(world, operator, T_MAX, start=start, seed=seed, hook=hook, fault=fault,
            stop_when=lambda t, x, y, th, v: st["phase"] == "refused" or (st["phase"] == "finished" and abs(v) < 0.02))
    return r, st


def _row_from(r, st, ideal_err, expect, extra):
    nav = r["assist"].nav
    length, gears = _path_metrics(r["trace"])
    planned_gears = 0 if st["planned"] is None else _gear_changes(st["planned"][:, 3].astype(int).tolist())
    parked = st["phase"] in ("finished",) and st["planned"] is not None
    refused = st["phase"] == "refused"
    row = {"expect": expect, "planned": int(st["planned"] is not None), "refused": int(refused), "arrived": int(nav.msg.startswith("arrived")),
           "nav_msg": nav.msg if st["planned"] is not None else (st["refused"] or ""), "collisions": int(bool(r["collided"])),
           "min_clear_cm": round(r["min_clear"] * 100, 1), "time_s": None, "sim_end_s": round(r["t_end"], 1),
           "path_m": round(length, 3), "gear_changes": gears, "planned_gear_changes": planned_gears,
           "corrections": int(max(0, gears - planned_gears) + nav.replans + nav.fine_strokes), "replans": nav.replans,
           "strokes": nav.fine_strokes}
    row.update(extra)
    if parked:
        row["time_s"] = round(r["t_end"] - st["t_plan"], 1)
    row.update(ideal_err)
    if expect == "refuse":
        row["success"] = int(refused or (st["planned"] is None))
    else:
        err_cm = row.get("score_err_cm", row.get("pos_err_cm"))
        ok = (parked and not r["collided"] and err_cm is not None and err_cm <= POS_TOL_M * 100
              and row.get("head_err_deg") is not None and row["head_err_deg"] <= HEAD_TOL_DEG and row.get("inside", 1) == 1)
        row["success"] = int(bool(ok))
    row["failure"] = "" if row["success"] else _why(row, r, st)
    tr = r["trace"][::5]
    row["trace"] = [(round(a[1], 3), round(a[2], 3), round(a[3], 3)) for a in tr]
    return row


def _why(row, r, st):
    if r["collided"]:
        return "collision"
    if st["planned"] is None and row["expect"] != "refuse":
        return "no plan: " + str(st["refused"])
    if row["expect"] == "refuse":
        return "drove although it should refuse"
    if st["phase"] != "finished":
        return "timeout / stuck: " + str(r["assist"].nav.msg)
    if row.get("inside", 1) == 0:
        k = row.get("kerb_gap_cm")
        if k is not None and k < KERB_MIN_CM:
            return f"on the kerb ({k} cm)"
        if k is not None and k > KERB_MAX_CM:
            return f"too far from the kerb ({k} cm)"
        return "ended outside the space"
    err_cm = row.get("score_err_cm", row.get("pos_err_cm"))
    if err_cm is not None and err_cm > POS_TOL_M * 100:
        return "position error"
    return "heading error"


# ------------------------------------------------------------------------------------------------ perpendicular
def _run_perpendicular(name, p, seed, mc, rng):
    from adas.park_select import fuse_markers, plan_perpendicular, rear_camera
    from sim.marker_sensor import MarkerSensor
    dist = p["dist_m"]
    sy, sd = p["start_y"], p["start_deg"]
    pixel, drop = p["pixel_noise"], p["frame_drop"]
    if mc:                                                      # randomised start, camera noise, LiDAR range noise
        sy += float(rng.uniform(-0.12, 0.12))
        sd += float(rng.uniform(-6, 6))
        dist += float(rng.uniform(-0.15, 0.15))
        pixel = float(rng.uniform(0.3, 1.2)) if pixel > 0 else 0.0
    world = build_perpendicular(p)
    start = (WALL_X + dist + 0.05, sy, math.radians(sd))          # rear axle; camera 5 cm behind it
    cam = rear_camera() if p["camera"] == "rear" else legacy_rear_camera()
    sensor = MarkerSensor(cam, pixel_noise=pixel, detect_prob=p["detect_prob"], seed=seed)
    frames_seen = {"n": 0, "frames": 0}

    def sensing(t, st):
        if T_OBSERVE[0] <= t < T_OBSERVE[1]:
            x, y, th = st["pose"]
            obs = sensor.sense(world, SimpleNamespace(x=x, y=y, theta=th), t)
            if obs is not None:
                frames_seen["frames"] += 1
                if drop and rng.random() < drop:
                    obs = []
                st["sensed"].extend(obs)
                frames_seen["n"] += len(obs)

    def plan_fn(assist, pts, st):
        info = {"marker_frames": frames_seen["frames"], "marker_obs": frames_seen["n"]}
        assist.marker_source = lambda: st["sensed"]
        msg = assist.park_marker(pts)                     # RelayAssists -> plan service -> AutoNav.start_plan
        if msg == "marker not visible":
            info["refused"] = "marker not detected"
        else:
            info["marker_facing_deg"] = round(math.degrees(fuse_markers(st["sensed"]).facing), 1)
        return info

    fault = _make_fault(p["lidar_sigma"] + (float(rng.uniform(0, 0.015)) if mc else 0.0), seed=seed)
    r, st = _execute(world, start, seed, p, plan_fn, None, mc, rng, sensing=sensing, fault=fault)
    a = r["assist"].p
    goal = (WALL_X + STOP_GAP + a.rear_overhang, 0.0, 0.0)
    x, y, th, *_ = (r["x"], r["y"], r["th"])
    ex, ey, eh = x - goal[0], y - goal[1], math.degrees(_wrap(th - goal[2]))
    err = {"pos_err_cm": round(math.hypot(ex, ey) * 100, 1), "lat_err_cm": round(abs(ey) * 100, 1),
           "long_err_cm": round(abs(ex) * 100, 1), "head_err_deg": round(abs(eh), 1), "inside": 1}
    extra = {"start_x": round(start[0], 2), "start_y": round(start[1], 2), "start_deg": sd, "target": "P2",
             "selected": "P2" if st["planned"] is not None else "", **{k: v for k, v in st["info"].items()}}
    return _row_from(r, st, err if st["planned"] is not None else {"pos_err_cm": None, "head_err_deg": None, "lat_err_cm": None,
                                                                   "long_err_cm": None, "inside": 0}, p["expect"], extra)


# ------------------------------------------------------------------------------------------------ parallel
def _true_gaps(strip, cars):
    """The free stretches between the true car footprints and the strip's ends."""
    gaps, prev = [], 0.0
    for a, b in sorted(cars):
        if a > prev:
            gaps.append((prev, a))
        prev = max(prev, b)
    if prev < strip.length:
        gaps.append((prev, strip.length))
    return gaps


def _expected_slot(strip, cars, click, car):
    """Independent of the car's perception: the slot the same rule (adas/park_select.choose_slots) gives on the TRUE gaps."""
    from adas.park_select import choose_slots
    s_click, _ = strip.project(click)
    sl = choose_slots(_true_gaps(strip, cars), s_click, car, strip.depth)
    return sl[0] if sl else None


SELECT_TIE_M = 0.10


def _tie_ok(strip, cars, click, car, chosen_s):
    """The chosen slot centre lies in a viable true gap whose slot is within SELECT_TIE_M as near to the click as the best one."""
    from adas.park_select import choose_slots
    s_click, _ = strip.project(click)
    slots = choose_slots(_true_gaps(strip, cars), s_click, car, strip.depth)
    if not slots:
        return False
    best = abs(slots[0].mid - s_click)
    return any(sp.gap[0] <= chosen_s <= sp.gap[1] and abs(sp.mid - s_click) <= best + SELECT_TIE_M for sp in slots)


def _click_for(policy, strip, cars, car, rng, target=None):
    """(click point in the world frame, description). gap = inside a viable gap, gap_end = near the end of the longest gap, car = on a
    parked car, short_gap = inside a gap too short for the car, gap_any = in any gap, beyond = past the end of the row."""
    from adas.park_select import min_space_length
    gaps = _true_gaps(strip, cars)
    need = min_space_length(car)
    viable = [g for g in gaps if g[1] - g[0] >= need]
    short = [g for g in gaps[1:-1] if g[1] - g[0] < need]
    depth_mid = strip.depth / 2
    at = lambda s: (strip.point_at(s)[0], strip.point_at(s)[1] + depth_mid)
    if policy == "target" and target is not None:
        return at(0.5 * (target[0] + target[1])), f"in the {100 * (target[1] - target[0]):.0f} cm gap"
    if policy == "gap" and viable:
        g = viable[int(rng.integers(len(viable)))]
        return at(g[0] + (g[1] - g[0]) * float(rng.uniform(0.3, 0.7))), f"in a {100 * (g[1] - g[0]):.0f} cm gap"
    if policy == "gap_end" and viable:
        g = max(viable, key=lambda g: g[1] - g[0])
        return at(g[1] - 0.25), f"near the end of a {100 * (g[1] - g[0]):.0f} cm gap"
    if policy == "car" and cars:
        a, b = cars[int(rng.integers(len(cars)))]
        return at(0.5 * (a + b)), "on a parked car"
    if policy == "short_gap" and short:
        g = short[int(rng.integers(len(short)))]
        return at(0.5 * (g[0] + g[1])), f"in a {100 * (g[1] - g[0]):.0f} cm gap (too short)"
    if policy == "gap_any" and gaps:
        g = max(gaps[1:-1] or gaps, key=lambda g: g[1] - g[0])
        return at(0.5 * (g[0] + g[1])), f"in a {100 * (g[1] - g[0]):.0f} cm gap"
    end = strip.point_at(strip.length)
    return (end[0] + 0.5, end[1] + depth_mid), "beyond the row"


def _run_parallel(name, p, seed, mc, rng):
    from adas.park_select import KERB_GAP_M
    world, start, strip, cars = build_parallel(p, seed)
    from sim.relay_scenarios import _tuning
    from pi.relay_assists import RelayAssists
    car = RelayAssists(_tuning()).p
    # the click is fixed by the case (seeded), not by the MC noise
    click_w, click_desc = _click_for(p["click"], strip, cars, car, np.random.default_rng(31 + (p["street"] or 0) * 7 + len(cars)),
                                     world.layout.get("target"))
    if mc:
        start = (start[0] + float(rng.uniform(-0.15, 0.15)), start[1] + float(rng.uniform(-0.06, 0.06)),
                 math.radians(float(rng.uniform(-6, 6))))
    expected = _expected_slot(strip, cars, click_w, car)

    def plan_fn(assist, pts, st):
        info = {"click": click_desc}
        # what the HMI publishes (docs/interfaces.md 16b): the strip and the click in the relay's world frame (= the start pose here)
        strip_w = {"points": [list(_to_start_frame((q[0], q[1]), start)) for q in strip.pts], "side": 1, "depth": strip.depth}
        t0 = time.perf_counter()
        info["command"] = assist.park_target(strip_w, list(_to_start_frame(click_w, start)), pts)
        info["plan_ms"] = round((time.perf_counter() - t0) * 1000)
        return info

    fault = _make_fault(float(rng.uniform(0, 0.015)) if mc else 0.0, seed=seed)
    r, st = _execute(world, start, seed, p, plan_fn, None, mc, rng, fault=fault)
    a = r["assist"].p
    nav = r["assist"].nav
    slot = getattr(nav, "parking_slot", None) if st["planned"] is not None else None
    err = {"pos_err_cm": None, "head_err_deg": None, "lat_err_cm": None, "long_err_cm": None, "inside": 0}
    chosen_s = None
    if slot is not None:
        cw = _from_start_frame(np.asarray(slot, float).mean(axis=0), start)
        chosen_s = strip.project(cw)[0]
    gaps = _true_gaps(strip, cars)
    if expected is not None and chosen_s is not None and not expected.gap[0] <= chosen_s <= expected.gap[1]:
        from adas.park_select import choose_slots
        alt = [sp for sp in choose_slots(gaps, strip.project(click_w)[0], car, strip.depth) if sp.gap[0] <= chosen_s <= sp.gap[1]]
        expected = alt[0] if alt else expected                          # a near-tie: the car is scored in the gap it chose
    if expected is not None and st["planned"] is not None:
        x, y, th = r["x"], r["y"], r["th"]
        bc = 0.5 * (a.front_x + a.rear_x)                              # body centre ahead of the rear axle
        bx, by = x + bc * math.cos(th), y + bc * math.sin(th)
        cx = strip.point_at(expected.mid)[0]
        cy = Y_EDGE + max(strip.depth / 2, strip.depth - KERB_GAP_M - a.width / 2)
        err = {"pos_err_cm": round(math.hypot(bx - cx, by - cy) * 100, 1), "lat_err_cm": round(abs(by - cy) * 100, 1),
               "long_err_cm": round(abs(bx - cx) * 100, 1), "head_err_deg": round(abs(math.degrees(_wrap(th))), 1)}
        g0, g1 = expected.gap
        x0s, x1s = strip.point_at(g0)[0], strip.point_at(g1)[0]
        cs = _body_corners((x, y, th), a)
        # inside = the whole body in the gap and in the lane, clear of the kerb by KERB_MIN_CM (user, 30 Sep: 'the back wheels keep
        # slightly clipping the kerb' - a car on the kerb line is not parked, whatever the position error says)
        kerb_gap = Y_EDGE + strip.depth - max(q[1] for q in cs)
        err["kerb_gap_cm"] = round(kerb_gap * 100, 1)
        err["inside"] = int(all(x0s - 0.01 <= q[0] <= x1s + 0.01 and Y_EDGE <= q[1] for q in cs)
                            and KERB_MIN_CM <= kerb_gap * 100 <= KERB_MAX_CM)
        # scored on what the user asks for: centred along the gap (the along error), straight, inside the lane close to the kerb - the
        # lateral target itself follows the neighbours (in line with them, ADR-042), so it is not a fixed number to be scored against
        err["score_err_cm"] = err["long_err_cm"]
    # the right gap: the one the rule gives on the true gaps, or - when the car's perception made a near-tie go the other way - a gap whose
    # slot is within SELECT_TIE_M as near to the click (e.g. 0.84 m vs 0.87 m: a driver would call both "the nearest")
    sel_ok = int((expected is None and chosen_s is None) or (expected is not None and chosen_s is not None and (
        expected.gap[0] <= chosen_s <= expected.gap[1] or _tie_ok(strip, cars, click_w, car, chosen_s))))
    extra = {"start_x": round(start[0], 2), "start_y": round(start[1], 2), "start_deg": round(math.degrees(start[2])),
             "target": "" if expected is None else f"{expected.mid:.2f}", "selected": "" if chosen_s is None else f"{chosen_s:.2f}",
             "spaces": len(cars), "free_spaces": sum(1 for g in gaps if g[1] - g[0] >= 0.93),
             "space_len_cm": None if expected is None else round((expected.gap[1] - expected.gap[0]) * 100),
             "space_ratio": None if expected is None else round((expected.gap[1] - expected.gap[0]) / (a.front_x - a.rear_x), 2),
             "click": st["info"].get("click", ""), "plan_ms": st["info"].get("plan_ms"), "selection_ok": sel_ok}
    row = _row_from(r, st, err, p["expect"], extra)
    if p["expect"] == "park" and row["success"] and not sel_ok:
        row["success"], row["failure"] = 0, "parked in the wrong gap"
    return row


# ------------------------------------------------------------------------------------------------ drivers
COLUMNS = ["name", "group", "seed", "mc", "expect", "success", "failure", "planned", "refused", "arrived", "pos_err_cm", "lat_err_cm",
           "long_err_cm", "head_err_deg", "inside", "kerb_gap_cm", "corrections", "gear_changes", "planned_gear_changes", "replans", "strokes",
           "collisions", "min_clear_cm", "time_s", "path_m", "sim_end_s", "start_x", "start_y", "start_deg", "target", "selected",
           "selection_ok", "click", "spaces", "free_spaces", "space_len_cm", "space_ratio", "plan_ms", "marker_frames", "marker_obs",
           "marker_facing_deg", "nav_msg", "wall_s"]


def write_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in COLUMNS})


def summarise(rows):
    """Per scenario: n, success rate, medians of the metrics over the runs that parked, collisions."""
    by = {}
    for r in rows:
        by.setdefault((r["group"], r["name"], r["mc"]), []).append(r)
    out = []
    for (g, n, mc), rs_ in sorted(by.items()):
        parked = [r for r in rs_ if r["time_s"] is not None]

        def med(k, src=parked):
            v = [r[k] for r in src if r.get(k) is not None]
            return round(float(np.median(v)), 2) if v else None

        def p95(k, src=parked):
            v = [r[k] for r in src if r.get(k) is not None]
            return round(float(np.percentile(v, 95)), 2) if v else None
        fails = {}
        for r in rs_:
            if not r["success"]:
                fails[r["failure"].split(":")[0]] = fails.get(r["failure"].split(":")[0], 0) + 1
        out.append({"group": g, "name": n, "mc": mc, "runs": len(rs_), "success": sum(r["success"] for r in rs_),
                    "success_pct": round(100 * sum(r["success"] for r in rs_) / len(rs_), 1), "parked": len(parked),
                    "collisions": sum(r["collisions"] for r in rs_), "expect": rs_[0]["expect"],
                    "pos_err_cm_med": med("pos_err_cm"), "pos_err_cm_p95": p95("pos_err_cm"),
                    "head_err_deg_med": med("head_err_deg"), "head_err_deg_p95": p95("head_err_deg"),
                    "corrections_med": med("corrections"), "time_s_med": med("time_s"), "path_m_med": med("path_m"),
                    "selection_ok": sum(r.get("selection_ok") or 0 for r in rs_) if g == "parallel" else None,
                    "failures": fails})
    return out


def run_all(jobs, workers=8, label="suite"):
    rows = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(run_case, jobs, chunksize=1), 1):
            rows.append(r)
            if i % 20 == 0 or i == len(jobs):
                print(f"  {label}: {i}/{len(jobs)} runs, {time.perf_counter() - t0:.0f} s", flush=True)
    return rows


def save(rows, tag):
    os.makedirs(OUT, exist_ok=True)
    write_csv(rows, os.path.join(OUT, f"results_{tag}.csv"))
    summ = summarise(rows)
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summ, "n_runs": len(rows)}, f, indent=1)
    with open(os.path.join(OUT, f"traces_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump([{"name": r["name"], "seed": r["seed"], "success": r["success"], "trace": r["trace"]} for r in rows], f)
    return summ


def print_summary(summ):
    for s in summ:
        print(f"{s['name']:24s} {s['success']:3d}/{s['runs']:<3d} ok  coll {s['collisions']}  pos {s['pos_err_cm_med']} cm  head {s['head_err_deg_med']} deg"
              f"  corr {s['corrections_med']}  t {s['time_s_med']} s  path {s['path_m_med']} m  {s['failures'] or ''}")


def main(argv):
    mode = argv[1] if len(argv) > 1 else "suite"
    if mode == "quick":
        cases = [c for c in all_cases() if c[0] in ("perp_perfect", "par_standard", "par_too_short")]
        rows = run_all([(n, g, p, 0, False) for n, g, p in cases], 3, "quick")
        print_summary(save(rows, "quick"))
    elif mode == "suite":
        rows = run_all([(n, g, p, 0, False) for n, g, p in all_cases()], 8, "suite")
        print_summary(save(rows, "suite"))
    elif mode == "mc":
        runs = int(argv[2]) if len(argv) > 2 else 10
        workers = int(argv[3]) if len(argv) > 3 else 8
        jobs = [(n, g, p, 100 + s, True) for n, g, p in all_cases() for s in range(runs)]
        print(f"{len(jobs)} Monte Carlo runs ({len(all_cases())} scenarios x {runs}), {workers} workers")
        rows = run_all(jobs, workers, "mc")
        print_summary(save(rows, "mc"))


if __name__ == "__main__":
    main(sys.argv)
