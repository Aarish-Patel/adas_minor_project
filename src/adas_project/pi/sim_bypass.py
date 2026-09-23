"""Laptop simulator for the bypass controller: a kinematic car, a ray-cast LiDAR (with noise) in a
room with an obstacle, the REAL scan-matching odometry and the REAL controller. Reports the
trajectory, clearance to the obstacle, and how well the car rejoined the original line.

    python -m pi.sim_bypass            (from the RC_Car folder)"""
import math
import sys

import numpy as np

sys.path.insert(0, ".")
from pi.bypass_core import Bypass, SERVO_STRAIGHT, K_CURV_PER_DEG, FRONT_EXTENT, REAR_EXTENT, BODY_HALF_W  # noqa: E402
from pi.scanmatch import Odometry, polar_to_xy, rot  # noqa: E402


def box(x0, y0, x1, y1):
    return [((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))]


def raycast(segs, pose, n=720, noise=0.01, rng=None):
    x, y, th = pose
    out = []
    for a in np.linspace(-180, 180, n, endpoint=False):
        ang = th + math.radians(a)
        dx, dy = math.cos(ang), math.sin(ang)
        best = None
        for (p, q) in segs:
            ex, ey = q[0] - p[0], q[1] - p[1]
            den = dx * ey - dy * ex
            if abs(den) < 1e-9:
                continue
            t = ((p[0] - x) * ey - (p[1] - y) * ex) / den
            u = ((p[0] - x) * dy - (p[1] - y) * dx) / den
            if t > 0.05 and 0 <= u <= 1 and (best is None or t < best):
                best = t
        if best is not None and best < 5.0:
            out.append((a, best + rng.normal(0, noise)))
    return out


def run(obstacle, extra_segs, label, seed=0, verbose=False, k_true=1.0, servo_bias=0.0):
    rng = np.random.default_rng(seed)
    room = box(-1.0, -1.6, 5.0, 1.6)
    segs = room + box(*obstacle) + extra_segs
    true = np.array([0.0, 0.0, 0.0])       # LiDAR pose in the world (== start frame)
    v, dt_scan = 0.26, 0.125
    ctl, odo = Bypass(pwm=115), Odometry()
    servo_cmd = SERVO_STRAIGHT
    servo_act = SERVO_STRAIGHT
    t = 0.0
    traj, min_clear, states = [], 9.0, []
    obs_rect = obstacle
    for step in range(400):
        scan = raycast(segs, true, rng=rng)
        xy = polar_to_xy(scan)
        kappa_cmd = K_CURV_PER_DEG * (servo_cmd - SERVO_STRAIGHT)
        est = odo.update(xy, t, v, kappa_cmd)
        r = ctl.step(est, xy)
        states.append(r["state"])
        if r["pwm"] == 0:
            break
        servo_cmd = r["servo_deg"]
        # servo lag + kinematics for one scan period
        for _ in range(5):
            servo_act += (servo_cmd - servo_act) * min(1.0, 0.025 / 0.15)
            kappa = k_true * K_CURV_PER_DEG * (servo_act - SERVO_STRAIGHT + servo_bias)
            h = dt_scan / 5
            true[2] += kappa * v * h
            true[0] += v * math.cos(true[2]) * h
            true[1] += v * math.sin(true[2]) * h
        t += dt_scan
        traj.append(true.copy())
        # clearance: nearest body-corner sample to obstacle rectangle
        corners = np.array([(FRONT_EXTENT, BODY_HALF_W), (FRONT_EXTENT, -BODY_HALF_W), (-REAR_EXTENT, BODY_HALF_W),
                            (-REAR_EXTENT, -BODY_HALF_W), (FRONT_EXTENT, 0), (0, BODY_HALF_W), (0, -BODY_HALF_W), (-REAR_EXTENT, 0)])
        cw = corners @ rot(true[2]).T + true[:2]
        x0, y0, x1, y1 = obs_rect
        dxx = np.maximum(np.maximum(x0 - cw[:, 0], 0), cw[:, 0] - x1)
        dyy = np.maximum(np.maximum(y0 - cw[:, 1], 0), cw[:, 1] - y1)
        inside = ((cw[:, 0] > x0) & (cw[:, 0] < x1) & (cw[:, 1] > y0) & (cw[:, 1] < y1)).any()
        min_clear = min(min_clear, -1.0 if inside else float(np.hypot(dxx, dyy).min()))
    traj = np.array(traj)
    fin = traj[-1]
    print(f"[{label}] end state={ctl.state}  msg: {ctl.msg}")
    print(f"    steps={len(traj)}  max lateral={np.abs(traj[:,1]).max():.2f} m  final y={fin[1]:+.3f} m  final heading={math.degrees(fin[2]):+.1f} deg  "
          f"min body-to-obstacle={min_clear:.3f} m  odom fallbacks={odo.fallbacks}/{odo.n} ref-fixes={odo.ref_fixes}  side={ctl.side:+d}")
    return ctl.state, min_clear, fin


if __name__ == "__main__":
    print("=== obstacle dead ahead, both sides open ===")
    run((1.4, -0.10, 1.6, 0.10), [], "open room")
    print("=== obstacle slightly off-center (+y), left more open? ===")
    run((1.4, 0.00, 1.6, 0.22), [], "offset obstacle")
    print("=== wall close on the +y side -> should go around the -y side ===")
    run((1.4, -0.10, 1.6, 0.10), box(0.8, 0.45, 2.6, 0.5), "+y side blocked")
    print("=== wall close on the -y side -> should go around the +y side ===")
    run((1.4, -0.10, 1.6, 0.10), box(0.8, -0.5, 2.6, -0.45), "-y side blocked")
    print("=== boxed in both sides -> must refuse ===")
    run((1.4, -0.10, 1.6, 0.10), box(0.8, 0.35, 2.6, 0.4) + box(0.8, -0.4, 2.6, -0.35), "both blocked")
    print("=== wider obstacle (0.4 m) ===")
    run((1.4, -0.20, 1.6, 0.20), [], "wide obstacle")
    print("=== model mismatch: steering 25% weaker, +1.5 deg servo bias ===")
    run((1.4, -0.10, 1.6, 0.10), [], "weak steering", k_true=0.75, servo_bias=1.5)
    print("=== model mismatch: steering 30% stronger, -1.5 deg bias ===")
    run((1.4, 0.00, 1.6, 0.22), [], "strong steering", k_true=1.3, servo_bias=-1.5)
