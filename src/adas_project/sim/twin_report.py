"""Digital-twin evidence from a real drive log: is the simulator the car?

    python -m sim.twin_report [LOG]          -> reports/twin_lidar.png, reports/twin_path.png, models/twin_report.json

1. LiDAR: the room map is built from the real scans at their scan-matched poses (even scans only); at the poses of
   the held-out (odd) scans the simulated LiDAR (sim/hw_sim.py SimLidar: beam geometry, 0.20 m blind range) is cast
   and compared beam by beam with what the real sensor returned.
2. Motion: the logged throttle and steering are replayed through the twin's car model (the one sim/hw_sim.py drives)
   in 3 s windows re-anchored on the real path; simulated vs real path, with the error.
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pi.drive_log import load  # noqa: E402
from sim.log_fit import command_series, hold, kinematics, reconstruct  # noqa: E402
from sim.hw_sim import load_car_model  # noqa: E402

ROOT = os.path.join(HERE, "..")
REPORTS = os.path.join(ROOT, "reports")


def lidar_to_car_frame(ang, dist, yaw, dmin=0.2):
    """Raw sensor angles/ranges -> points in the LiDAR frame (x forward, y RIGHT, as pi/scanmatch)."""
    a = np.radians((np.asarray(ang) - yaw) % 360.0)
    d = np.asarray(dist)
    ok = d >= dmin
    return a[ok], d[ok]


CELL = 0.03


def build_map(log, rows, yaw, use, min_scans=3):
    """Static occupancy map: 3 cm cells the LiDAR hit in at least `min_scans` different scans (transient things,
    e.g. a person walking past, drop out). Each kept cell stores the mean position of its hits: the wall surface."""
    cells = {}                                       # key -> [distinct scans, sum x, sum y, points, last scan]
    for i in use:
        t, ang, dist, _ = log["scans"][i]
        x, y, th = rows[i, 1], rows[i, 2], rows[i, 3]
        a, d = lidar_to_car_frame(ang, dist, yaw)
        px, py = d * np.cos(a), d * np.sin(a)
        wx = x + np.cos(th) * px - np.sin(th) * py
        wy = y + np.sin(th) * px + np.cos(th) * py
        for X, Y in zip(wx, wy):
            c = cells.setdefault((int(np.floor(X / CELL)), int(np.floor(Y / CELL))), [0, 0.0, 0.0, 0, -1])
            c[1] += X
            c[2] += Y
            c[3] += 1
            if c[4] != i:
                c[0] += 1
                c[4] = i
    return {k: (c[1] / c[3], c[2] / c[3]) for k, c in cells.items() if c[0] >= min_scans}


def raycast(occ, x, y, th, beams, max_range=4.0, step=0.01):
    """First statically occupied cell along each beam; range = that cell's mean surface point projected on the beam."""
    r = np.arange(0.2, max_range, step)
    ang = th + beams
    kx = np.floor((x + np.cos(ang)[:, None] * r[None, :]) / CELL).astype(int)
    ky = np.floor((y + np.sin(ang)[:, None] * r[None, :]) / CELL).astype(int)
    out = np.full(len(beams), np.inf)
    for bi in range(len(beams)):
        for j in range(len(r)):
            m = occ.get((kx[bi, j], ky[bi, j]))
            if m is not None:
                out[bi] = (m[0] - x) * math.cos(ang[bi]) + (m[1] - y) * math.sin(ang[bi])
                break
    return out


def lidar_check(log, rows, yaw, n_test=12):
    trusted = np.flatnonzero(rows[:, 4] > 0)
    build = trusted[trusted % 2 == 0]
    test = trusted[trusted % 2 == 1]
    test = test[np.linspace(0, len(test) - 1, min(n_test, len(test))).astype(int)]
    segs = build_map(log, rows, yaw, build)
    errs, examples = [], []
    for i in test:
        t, ang, dist, _ = log["scans"][i]
        a, d = lidar_to_car_frame(ang, dist, yaw)
        sim = raycast(segs, rows[i, 1], rows[i, 2], rows[i, 3], a)
        ok = np.isfinite(sim) & (d < 4.0)
        e = sim[ok] - d[ok]
        errs.append(e)
        if len(examples) < 3:
            examples.append((a, d, sim))
    e = np.concatenate(errs)
    ae = np.abs(e)
    return {"beams_compared": int(len(e)), "median_abs_error_cm": float(np.median(ae) * 100),
            "p90_abs_error_cm": float(np.percentile(ae, 90) * 100), "within_5cm_pct": float((ae < 0.05).mean() * 100),
            "bias_cm": float(np.median(e) * 100), "map_cells": int(len(segs)), "test_scans": int(len(test))}, examples, e


def motion_check(log, rows, horizon=3.0):
    m = load_car_model()
    servo, pwm = command_series(log)
    tv, v, w, good = kinematics(rows)
    t = rows[:, 0]
    real, simp, errs = [], [], []
    i = 0
    while i < len(t):
        j = np.searchsorted(t, t[i] + horizon)
        if j >= len(t):
            break
        if rows[i, 4] == 0:
            i += 1
            continue
        x, y, th = rows[i, 1:4]
        vv = float(np.interp(t[i], tv, v))
        tt, path = t[i], []
        while tt < t[j]:
            u = float(hold(pwm, tt - m["delay_s"]))
            s = float(hold(servo, tt - m["delay_s"] - 0.05))
            if u == 0:
                dec = min(abs(vv) / max(m["tau_motor"], 1e-3), m["coast_decel"]) * 0.01
                vv = 0.0 if abs(vv) <= dec else vv - math.copysign(dec, vv)
            else:
                vss = math.copysign(m["v_max"] * min(1.0, max(0.0, (abs(u) - m["deadband"]) / (255 - m["deadband"]))), u)
                vv += max(-4.0, min(3.0, (vss - vv) / max(m["tau_motor"], 0.02))) * 0.01
            th += m["k_curv_per_deg"] * (s - m["servo_centre"]) * vv * 0.01
            x += vv * math.cos(th) * 0.01
            y += vv * math.sin(th) * 0.01
            path.append((x, y))
            tt += 0.01
        simp.append(np.array(path))
        real.append(rows[i:j + 1, 1:3])
        errs.append(math.hypot(x - rows[j, 1], y - rows[j, 2]))
        i = j
    e = np.array(errs) * 100
    return {"windows": len(e), "horizon_s": horizon, "median_end_error_cm": float(np.median(e)),
            "p90_end_error_cm": float(np.percentile(e, 90)), "distance_driven_m": float(np.abs(v[good]).sum() * np.median(np.diff(tv)))}, real, simp


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else sorted(glob.glob(os.path.join(ROOT, "logs", "*log_drive*.jsonl.gz")),
                                                          key=os.path.getsize)[-1]
    log = load(path)
    yaw = log["meta"]["tuning"]["mount"]["yaw_offset_deg"]
    rows = reconstruct(log)
    lid, examples, e = lidar_check(log, rows, yaw)
    mot, real, simp = motion_check(log, rows)
    print(f"log {os.path.basename(path)}")
    print(f"LiDAR  : {lid['beams_compared']} beams in {lid['test_scans']} held-out scans: median |sim - real| "
          f"{lid['median_abs_error_cm']:.1f} cm, 90th {lid['p90_abs_error_cm']:.1f} cm, {lid['within_5cm_pct']:.0f}% within 5 cm, bias {lid['bias_cm']:+.1f} cm")
    print(f"Motion : {mot['windows']} x {mot['horizon_s']:.0f} s windows: twin ends {mot['median_end_error_cm']:.1f} cm from the real car "
          f"(median), 90th {mot['p90_end_error_cm']:.1f} cm")

    from sim.report import style
    plt = style()
    os.makedirs(REPORTS, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, (a, d, sim) in zip(axes[:3], examples):
        ax.scatter(d * np.sin(a), d * np.cos(a), s=2, c="#60a5fa", label="real LiDAR")
        ok = np.isfinite(sim)
        ax.scatter(sim[ok] * np.sin(a[ok]), sim[ok] * np.cos(a[ok]), s=2, c="#f97316", label="simulated LiDAR", alpha=0.6)
        ax.plot([0], [0], "^", color="white")
        ax.set_aspect("equal"); ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_title("held-out scan (car at centre)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    axes[3].hist(np.clip(e * 100, -20, 20), bins=60, color="#2dd4bf")
    axes[3].set_title(f"sim - real per beam: median |e| {lid['median_abs_error_cm']:.1f} cm"); axes[3].set_xlabel("cm")
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "twin_lidar.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for k, (r, s) in enumerate(zip(real, simp)):
        ax.plot(r[:, 0], -r[:, 1], color="#60a5fa", lw=2, label="real car (scan-matched)" if k == 0 else None)
        ax.plot(s[:, 0], -s[:, 1], color="#f97316", lw=1.2, ls="--", label="digital twin (same commands)" if k == 0 else None)
    ax.set_aspect("equal"); ax.legend(frameon=False)
    ax.set_title(f"3 s replays: twin ends {mot['median_end_error_cm']:.1f} cm from the car (median)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m, left)")
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "twin_path.png"), dpi=140); plt.close(fig)
    json.dump({"log": os.path.basename(path), "lidar": lid, "motion": mot},
              open(os.path.join(ROOT, "models", "twin_report.json"), "w"), indent=1)
    print("wrote reports/twin_lidar.png, reports/twin_path.png, models/twin_report.json")


if __name__ == "__main__":
    main()
