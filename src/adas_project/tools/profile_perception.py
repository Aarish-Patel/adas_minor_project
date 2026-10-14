"""Cost per LiDAR scan of the perception chain, laptop and estimated Pi 5 (TODO P2-1).

    python tools/profile_perception.py [laps]        -> table + models/perception_profile.json

The pieces are timed one by one on a twin drive round the 'room' world (the closed route of sim/slam_eval.py, 0.5 m/s, one
720-beam scan per 5 cm), with the scans' range noise, in the order the relay runs them:
  convert   relay scan -> vehicle frame                          (control loop, per scan)
  memory    blind-zone memory: advance, prune contradicted, add  (control loop, per scan)
  ekf       range flow (RF2O) + EKF correction                   (control loop, per scan) + predict per packet (20 Hz)
  icp       keyframed scan matching (pi/scanmatch.Odometry)      (control loop, per scan)
  slam      submit -> pose graph add_scan (+ loop closure)       (worker thread, only the newest scan waits)
  tracking  clusters, association, velocities                    (LiDAR thread, per scan)
Budgets (docs/modules/perception.md): everything within one scan period (100 ms at 10 Hz) and the control-loop pieces within the
20 ms share of the 50 ms tick. The Pi factor is PI_COMPUTE_FACTOR (sim/hw_sim.py, 3.5x for single-thread Python).
"""
import json
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

SCAN_PERIOD_MS = 100.0
LOOP_BUDGET_MS = 20.0
V_M_S = 0.5
WARMUP = 5


def relay_scans(laps=1, seed=0):
    """(t, relay-format points, true pose) for a twin drive along the closed route with a noisy 720-beam LiDAR."""
    from adas.config import load_tuning
    from pi.relay_assists import car_params
    from sim.hw_sim import SimLidar, VirtualCar
    from sim.hw_worlds import WORLDS
    from sim.slam_eval import route
    world, start = WORLDS["room"]()
    tun = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
    p = car_params(tun.mount)
    car = VirtualCar(world, p, start, threaded=False)
    lid = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
    pts_w, th_w = route()
    cw = (-np.degrees(lid.ccw)) % 360
    cw = np.where(cw > 180, cw - 360, cw)
    out, t = [], 0.0
    for _ in range(laps):
        for (x, y), th in zip(pts_w, th_w):
            ox, oy = x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th)
            best, _ = lid._raycast(ox, oy, th)
            r = best + lid.rng.normal(0, 0.008, len(best))
            ok = np.isfinite(best) & (r >= 0.2) & (r < 12)
            out.append((t, [(round(float(a), 1), round(float(d), 3)) for a, d in zip(cw[ok], r[ok])], (x, y, th)))
            t += 0.05 / V_M_S
    return out, p, tun


def main(laps=1):
    from adas.memory import ObstacleMemory
    from adas.submaps import PoseGraphMap
    from adas.tracking import Tracker, moving_object_contact
    from pi.relay_assists import RelayAssists, RelaySpeed
    from pi.scanmatch import Odometry, polar_to_xy
    from sim.hw_sim import PI_COMPUTE_FACTOR
    scans, p, tun = relay_scans(laps)
    speed = RelaySpeed(tun.speed_model, p.lidar_x)
    mem = ObstacleMemory(p, blind_radius=0.27, keep_radius=1.2, max_age=60.0, max_points=700, max_travel=1.5)
    odo, tracker, graph = None, Tracker(), PoseGraphMap()
    searched, plain = [], []
    T = {k: [] for k in ("convert", "memory", "ekf (scan)", "ekf (predict, per packet)", "icp", "slam (worker)", "tracking")}
    for k, (t, pts, true) in enumerate(scans):
        def tic():
            return time.perf_counter()
        t0 = tic()
        xy_v = RelayAssists.points_vehicle_frame(pts, p.lidar_x)
        T["convert"].append((tic() - t0) * 1e3)

        t0 = tic()
        mem.advance(0.1, V_M_S, 0.0)
        mem.prune_contradicted(xy_v)
        mem.add_scan(xy_v)
        mem.blind_points()
        T["memory"].append((tic() - t0) * 1e3)

        t0 = tic()
        speed.command(t, 0.05, 150.0, 87.0)
        T["ekf (predict, per packet)"].append((tic() - t0) * 1e3)

        t0 = tic()
        a = -np.radians(np.array([q[0] for q in pts]))
        d = np.array([q[1] for q in pts])
        xy_l = np.column_stack([d * np.cos(a), d * np.sin(a)])
        e = speed.ekf
        meas = speed.rf.update(xy_l, t, guess=(e.v, e.w * e.lx, e.w))
        if meas is not None:
            e.correct(t, meas)
        T["ekf (scan)"].append((tic() - t0) * 1e3)

        t0 = tic()
        xy = polar_to_xy(pts)
        if odo is None:
            odo = Odometry()
        odo.update(xy, t, speed.ekf.v, 0.0)
        if k % 20 == 19:                                           # a keyframe every ~1 m
            odo = None
        T["icp"].append((tic() - t0) * 1e3)

        t0 = tic()
        n_opt = graph.corrections
        graph.add_scan(np.array(true, float), xy_v)
        dt_slam = (tic() - t0) * 1e3
        T["slam (worker)"].append(dt_slam)
        (searched if dt_slam > 5.0 else plain).append(dt_slam)

        t0 = tic()
        tracks = tracker.update(xy_v, t, V_M_S, 0.0)
        moving_object_contact(tracks, 0.0, 1, V_M_S, p, horizon=1.5)
        T["tracking"].append((tic() - t0) * 1e3)

    for k in T:
        T[k] = T[k][WARMUP:]                                   # first scans pay imports and cache misses, not the steady state
    print(f"{len(scans)} scans, {np.mean([len(s[1]) for s in scans]):.0f} points each, {V_M_S} m/s, laptop vs Pi x{PI_COMPUTE_FACTOR}")
    print(f"{'piece':28s} {'median':>7s} {'p95':>7s} {'max':>7s}   {'Pi median':>9s} {'Pi p95':>7s} {'Pi max':>7s}   (ms)")
    res = {}
    for k, v in T.items():
        a = np.array(v)
        res[k] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
        print(f"{k:28s} {np.median(a):7.2f} {np.percentile(a, 95):7.2f} {a.max():7.1f}   "
              f"{np.median(a) * PI_COMPUTE_FACTOR:9.1f} {np.percentile(a, 95) * PI_COMPUTE_FACTOR:7.1f} {a.max() * PI_COMPUTE_FACTOR:7.0f}")
    loop = ("convert", "memory", "ekf (scan)", "icp")
    per_scan_loop = np.array([sum(T[k][i] for k in loop) for i in range(len(scans) - WARMUP)])
    per_scan_all = np.array([sum(T[k][i] for k in T if k != "ekf (predict, per packet)") for i in range(len(scans) - WARMUP)])
    res["control-loop share, scan tick"] = {"median": float(np.median(per_scan_loop)), "p95": float(np.percentile(per_scan_loop, 95)),
                                            "max": float(per_scan_loop.max())}
    res["all pieces, per scan"] = {"median": float(np.median(per_scan_all)), "p95": float(np.percentile(per_scan_all, 95)),
                                   "max": float(per_scan_all.max())}
    for label in ("control-loop share, scan tick", "all pieces, per scan"):
        r = res[label]
        print(f"{label:28s} {r['median']:7.2f} {r['p95']:7.2f} {r['max']:7.1f}   {r['median'] * PI_COMPUTE_FACTOR:9.1f} "
              f"{r['p95'] * PI_COMPUTE_FACTOR:7.1f} {r['max'] * PI_COMPUTE_FACTOR:7.0f}")
    print(f"slam: {len(plain)} scans without a loop search (median {np.median(plain):.2f} ms), {len(searched)} with one "
          f"(median {np.median(searched):.0f} ms, max {max(searched):.0f} ms; Pi x{PI_COMPUTE_FACTOR}: {np.median(searched) * PI_COMPUTE_FACTOR:.0f} / "
          f"{max(searched) * PI_COMPUTE_FACTOR:.0f} ms)")
    print(f"budgets: control-loop pieces (scan tick) <= {LOOP_BUDGET_MS:.0f} ms; all pieces within one scan period ({SCAN_PERIOD_MS:.0f} ms)")
    res["slam_no_search_median"] = float(np.median(plain))
    res["slam_search_median"], res["slam_search_max"] = float(np.median(searched)), float(max(searched))
    res["pi_factor"], res["scans"] = PI_COMPUTE_FACTOR, len(scans)
    json.dump(res, open(os.path.join(ROOT, "models", "perception_profile.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
