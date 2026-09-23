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
CANDIDATES = [86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0, 93.0, 94.0]
REPEATS = 3
PWM = 140
RUN_S = 1.5
REPORT = "/home/pi/rc_car/pi/center_scan_report.json"


def profile(points, res=1.0):
    """Median range per 1deg bin over the full 360 (NaN where empty)."""
    n = int(360 / res)
    bins = [[] for _ in range(n)]
    for a, d in points:
        bins[int(round((a % 360) / res)) % n].append(d)
    return np.array([np.median(b) if b else np.nan for b in bins])


# Forward/backward motion barely changes the range to things at +-90deg (they slide along the
# line of sight's perpendicular), but a rotation shifts them - so match on the SIDES only. Front
# and rear ranges change a lot with translation and add nothing about rotation.
_ang = np.arange(360)
SIDE_MASK = ((_ang >= 62) & (_ang <= 118)) | ((_ang >= 242) & (_ang <= 298))


def rotation_between(p0, p1, max_shift=15):
    """Bearing shift (deg, signed) that best aligns scan p1 onto scan p0, using only ranges
    >0.7m (nearby ranges change with the car's translation; far ones are ~ rotation-only).
    Sub-degree via a parabola through the best 3 shifts. Returns (shift, median_abs_resid)."""
    r0, r1 = profile(p0), profile(p1)
    best = []
    for s in range(-max_shift, max_shift + 1):
        r1s = np.roll(r1, -s)
        ok = ~np.isnan(r0) & ~np.isnan(r1s) & (r0 > 0.5) & (r1s > 0.5) & SIDE_MASK
        if ok.sum() < 40:
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


def _xy(points, max_n=220):
    arr = np.array([(a, d) for a, d in points if 0.3 <= d <= 4.0], dtype=float)
    if len(arr) > max_n:
        arr = arr[np.linspace(0, len(arr) - 1, max_n).astype(int)]
    r = np.radians(arr[:, 0])
    return np.stack([arr[:, 1] * np.cos(r), arr[:, 1] * np.sin(r)], axis=1)


def icp(p0, p1, iters=30, gate=0.25):
    """2D point-to-point ICP: rotation (deg) and translation that map scan p1 onto p0, solved
    TOGETHER, so the car's own travel between the scans doesn't corrupt the rotation the way
    range-profile matching did. Returns (rot_deg, mean_resid_m, n_inliers) or None."""
    A, B = _xy(p0), _xy(p1)
    if len(A) < 60 or len(B) < 60:
        return None
    R = np.eye(2); t = np.zeros(2)
    for _ in range(iters):
        Bt = B @ R.T + t
        d2 = ((Bt[:, None, :] - A[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        dist = np.sqrt(d2[np.arange(len(B)), j])
        keep = dist < gate
        if keep.sum() < 40:
            return None
        P, Q = Bt[keep], A[j[keep]]
        pc, qc = P.mean(0), Q.mean(0)
        H = (P - pc).T @ (Q - qc)
        U, _, Vt = np.linalg.svd(H)
        Ri = Vt.T @ U.T
        if np.linalg.det(Ri) < 0:
            Vt[-1] *= -1; Ri = Vt.T @ U.T
        ti = qc - Ri @ pc
        R = Ri @ R; t = Ri @ t + ti
    Bt = B @ R.T + t
    d2 = ((Bt[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    dist = np.sqrt(d2.min(1))
    ok = dist < gate
    return float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))), float(dist[ok].mean()), int(ok.sum())


def heading_change(scans0, scans1):
    vals = []
    for p0 in scans0:
        for p1 in scans1:
            r = icp(p0, p1)
            if r and r[1] < 0.06 and r[2] >= 80:
                vals.append(r[0])
    return (float(np.median(vals)), len(vals)) if len(vals) >= 3 else (None, len(vals))


def go_to_open_center(rig, gui):
    """Drive straight to where there is room on BOTH sides of us (rear >= 0.9m and front >= 1.0m)
    so every test run starts from open space instead of hugging a wall."""
    gui.set(activity="FINDING OPEN SPACE - driving to the open center", servo=90.0)
    rig.steer(90)
    t0 = time.time()
    while time.time() - t0 < 6.0:
        f, r = rig.front(), rig.rear()
        if not rig.is_fresh() or (f is not None and f < 1.0):
            break
        if r is not None and r >= 0.9:
            break
        rig.motor_forward(120)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(0.6)
    return rig.front(), rig.rear()


def return_to_start(rig, rear0):
    """Reverse straight (closed loop on the rear reading) until we are back at the start spot."""
    rig.steer(90)
    t0 = time.time()
    while time.time() - t0 < 4.0:
        r = rig.rear()
        if not rig.is_fresh() or r is None or r <= rear0 + 0.02 or r < 0.35:
            break
        rig.motor_reverse(110)
        time.sleep(TICK_S)
    rig.stop()
    time.sleep(0.5)


def one_run(rig, center, gui=None):
    rig.steer(center + 10)      # approach the target angle from the same side (backlash)
    time.sleep(0.35)
    rig.steer(center)
    time.sleep(0.5)
    if gui:
        gui.set(activity="TESTING WHEEL CENTER - driving straight", servo=float(center))
    s0 = stable_scans(rig)
    f0, rear0 = rig.front(), rig.rear()
    if len(s0) < 3 or f0 is None or f0 < 0.9:
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
    out = ({"center": center, "skipped": f"poor scan match ({n} good pairs)"} if d is None else
           {"center": center, "d_heading_deg": d, "pairs": n, "travel_m": f0 - (rig.front() or f0)})
    if gui:
        gui.set(activity="FINDING OPEN SPACE - returning to the start spot", servo=90.0)
    return_to_start(rig, rear0 if rear0 is not None else 0.9)
    return out


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Servo center scan (final grips)",
            f"{len(CANDIDATES)} candidate centers x {REPEATS} repeats = {len(CANDIDATES) * REPEATS} straight runs; "
            "straightest = smallest heading change over the run", 0.0)
    total = len(CANDIDATES) * REPEATS
    runs = []
    try:
        fc, rc = go_to_open_center(rig, gui)
        gui.log(f"open center reached: front {fc if fc is None else round(fc, 2)} m, rear {rc if rc is None else round(rc, 2)} m")
        for rep in range(REPEATS):
            for c in CANDIDATES:
                gui.set(message=f"run {len(runs) + 1}/{total}: servo {c} deg (repeat {rep + 1}/{REPEATS})",
                        progress=len(runs) / total)
                r = one_run(rig, c, gui)
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
            res = y - (m * x + b)
            se = float(np.sqrt(res.var(ddof=2) / max(np.sum((x - x.mean()) ** 2), 1e-9))) if len(x) > 2 else None
            sig = abs(m) / se if se else 0.0
            fit = {"slope_deg_per_servo_deg": float(m), "slope_t": float(sig),
                   "zero_crossing_center": float(-b / m) if m else None,
                   "resid_sd_deg": float(res.std())}
        json.dump({"runs": runs, "summary": summary, "fit": fit}, open(REPORT, "w"), indent=2)
        print("SUMMARY", json.dumps(summary), "FIT", json.dumps(fit), flush=True)
        try:
            for c, v in summary.items():
                gui.log(f"servo {c}: mean {v['mean']:+.2f} deg  (sd {v['std']:.2f}, n={v['n']})")
            if fit and fit["zero_crossing_center"] is not None:
                ok = fit["slope_t"] >= 2.5 and 84 <= fit["zero_crossing_center"] <= 96
                gui.log(f"fit: slope {fit['slope_deg_per_servo_deg']:+.3f} deg/deg (t={fit['slope_t']:.1f}), "
                        f"zero at {fit['zero_crossing_center']:.1f}, resid sd {fit['resid_sd_deg']:.2f}")
                gui.log("IDEAL CENTER = %.1f deg" % fit["zero_crossing_center"] if ok else
                        "TREND NOT SIGNIFICANT - servo center cannot be resolved better than the noise; keep 90")
            gui.set("Servo center scan - DONE", "results below", 1.0)
            time.sleep(60)   # keep the page up so the results can be read
        except Exception:
            pass


if __name__ == "__main__":
    main()
