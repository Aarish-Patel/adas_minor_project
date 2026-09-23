"""Servo-center search with the final grips: 5 candidate centers x 5 repeats = 25 straight runs.

Straightness = change in the FRONT WALL'S ANGLE over the run (line fit to the front points).
A wall's orientation is unaffected by the car translating, so - unlike tracking the nearest
point's bearing, which broke down all night in this cluttered room - the change is purely
the car's own rotation. Candidates are interleaved (not blocked) so drift in the floor/battery
can't masquerade as a servo effect. The ideal center is where a line fit of
delta_heading vs commanded servo angle crosses zero."""
import json
import math
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
import pi.speed_run as SR  # noqa: E402
from pi.lidar_steering_diag import Rig, TICK_S
from pi.test_gui import TestGui  # noqa: E402

D.RAMP_STEP_PWM = 25
CANDIDATES = [85.0, 87.5, 90.0, 92.5, 95.0]
REPEATS = 5
PWM = 130
RUN_S = 1.1
REPORT = "/home/pi/rc_car/pi/center_scan_report.json"


def profile(points, res=1.0):
    """Median range per 1deg bin over the full 360 (NaN where empty)."""
    n = int(360 / res)
    bins = [[] for _ in range(n)]
    for a, d in points:
        bins[int(round((a % 360) / res)) % n].append(d)
    return np.array([np.median(b) if b else np.nan for b in bins])


def rotation_between(p0, p1, max_shift=15):
    """Bearing shift (deg, signed) that best aligns scan p1 onto scan p0, using only ranges
    >0.7m (nearby ranges change with the car's translation; far ones are ~ rotation-only).
    Sub-degree via a parabola through the best 3 shifts. Returns (shift, median_abs_resid)."""
    r0, r1 = profile(p0), profile(p1)
    best = []
    for s in range(-max_shift, max_shift + 1):
        r1s = np.roll(r1, -s)
        ok = ~np.isnan(r0) & ~np.isnan(r1s) & (r0 > 0.7) & (r1s > 0.7)
        if ok.sum() < 60:
            best.append(np.inf); continue
        best.append(float(np.median(np.abs(r0[ok] - r1s[ok]))))
    best = np.array(best)
    k = int(np.argmin(best))
    if not np.isfinite(best[k]):
        return None
    shift = float(k - max_shift)
    if 0 < k < len(best) - 1 and np.isfinite(best[k - 1]) and np.isfinite(best[k + 1]):
        den = best[k - 1] - 2 * best[k] + best[k + 1]
        if den > 1e-9:
            shift += 0.5 * (best[k - 1] - best[k + 1]) / den
    return shift, float(best[k])


def stable_scans(rig, n=3, timeout=2.0):
    out, last_t, end = [], None, time.time() + timeout
    while time.time() < end and len(out) < n:
        last_t, s = SR.unique_scan_sample(rig, last_t)
        if s:
            pts = rig.points()
            if len(pts) > 150:
                out.append(pts)
        time.sleep(0.03)
    return out


def heading_change(scans0, scans1):
    vals = []
    for p0 in scans0:
        for p1 in scans1:
            r = rotation_between(p0, p1)
            if r and r[1] < 0.20:
                vals.append(r[0])
    return (float(np.median(vals)), len(vals)) if len(vals) >= 3 else (None, len(vals))


def one_run(rig, center):
    SR.back_up_to_rear_limit(rig)
    rig.steer(center)
    time.sleep(0.5)
    s0 = stable_scans(rig)
    f0 = rig.front()
    if len(s0) < 3 or f0 is None or f0 < 0.95:
        return {"center": center, "skipped": f"scans={len(s0)} front={f0}"}
    t0 = time.time()
    while time.time() - t0 < RUN_S:
        f = rig.front()
        if f is not None and f < 0.55:
            break
        rig.motor_forward(PWM)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(0.7)
    s1 = stable_scans(rig)
    d, n = heading_change(s0, s1)
    if d is None:
        return {"center": center, "skipped": f"poor scan match ({n} good pairs)"}
    return {"center": center, "d_heading_deg": d, "pairs": n,
            "travel_m": f0 - (rig.front() or f0)}


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Servo center scan (final grips)",
            f"{len(CANDIDATES)} candidate centers x {REPEATS} repeats = {len(CANDIDATES) * REPEATS} straight runs; "
            "straightest = smallest heading change over the run", 0.0)
    total = len(CANDIDATES) * REPEATS
    runs = []
    try:
        for rep in range(REPEATS):
            for c in CANDIDATES:
                gui.set(message=f"run {len(runs) + 1}/{total}: servo {c} deg (repeat {rep + 1}/{REPEATS})",
                        progress=len(runs) / total)
                r = one_run(rig, c)
                if "d_heading_deg" in r:
                    gui.log(f"#{len(runs) + 1} servo {c}: heading {r['d_heading_deg']:+.2f} deg, travel {r['travel_m']:.2f} m")
                else:
                    gui.log(f"#{len(runs) + 1} servo {c}: SKIPPED ({r['skipped']})")
                runs.append(r)
    finally:
        rig.stop(); rig.steer(90); time.sleep(0.2); rig.close()
        good = [r for r in runs if "d_heading_deg" in r]
        summary = {}
        for c in CANDIDATES:
            v = [r["d_heading_deg"] for r in good if r["center"] == c]
            if v:
                summary[c] = {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))}
        fit = None
        if len(good) >= 8:
            x = np.array([r["center"] for r in good]); y = np.array([r["d_heading_deg"] for r in good])
            m, b = np.polyfit(x, y, 1)
            fit = {"slope_deg_per_servo_deg": float(m), "zero_crossing_center": float(-b / m) if m else None}
        json.dump({"runs": runs, "summary": summary, "fit": fit}, open(REPORT, "w"), indent=2)
        print("SUMMARY", json.dumps(summary), "FIT", json.dumps(fit), flush=True)
        try:
            for c, v in summary.items():
                gui.log(f"servo {c}: mean {v['mean']:+.2f} deg  (sd {v['std']:.2f}, n={v['n']})")
            if fit and fit["zero_crossing_center"] is not None:
                gui.log(f"IDEAL CENTER ~ {fit['zero_crossing_center']:.1f} deg (zero heading change)")
            gui.set("Servo center scan - DONE", "results below", 1.0)
            time.sleep(60)   # keep the page up so the results can be read
        except Exception:
            pass


if __name__ == "__main__":
    main()
