"""How well does the car know its own speed and yaw rate? (TODO C1/C5)

    python -m sim.odometry_eval            -> table + reports/odometry.png, models/odometry_eval.json

Estimators compared, all causal (what the relay could run live):
  pwm model   the relay's current speed: the fitted throttle->speed model with a lag (adas/aeb.SpeedEstimator)
  icp diff    scan-to-scan ICP poses (pi/scanmatch.py), differentiated - how speed was measured before
  rf2o        range-flow velocity (adas/rf2o.py, Jaimez et al. ICRA 2016), one estimate per scan
  ekf         the fitted car model fused with rf2o (adas/speed_ekf.py)
1. digital twin (sim/hw_sim.py): true speed and yaw rate known exactly;
2. the real logging drive: reference = the scan-matched path smoothed over +-0.25 s (non-causal, so it is better
   than any live estimate can be).
"""
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


def make_ekf(m, lidar_x=0.12, robust=False):
    from adas.speed_ekf import SpeedEKF
    return SpeedEKF(m["v_max"], m["deadband"], tau=max(m["tau_motor"], 0.05), delay=m["delay_s"],
                    coast_decel=m["coast_decel"], k_curv_per_deg=m["k_curv_per_deg"],
                    servo_centre=m["servo_centre"], lidar_x=lidar_x, robust=robust)


def stats(est, ref, ok):
    e = (np.asarray(est) - np.asarray(ref))[ok]
    e = e[np.isfinite(e)]
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "median_abs": float(np.median(np.abs(e))), "n": int(len(e))}


# ------------------------------------------------------------------ 1. digital twin
def twin_run(seconds=24.0, seed=3, battery=1.0, robust=False, corrupt=0.0, mag=(0.10, 0.35)):
    """battery < 1: the twin car is that much slower than the model the estimators believe (a sagging battery or a
    carpet) - the case a throttle-only speed estimate cannot see."""
    from adas.aeb import SpeedEstimator, SpeedModel
    from adas.rf2o import RangeFlow
    from pi.scanmatch import Odometry
    from sim.hw_sim import SimLidar, VirtualCar, load_car_model
    from sim.hw_worlds import room_walls
    from sim.world import Box, World
    from pi.relay_assists import car_params
    from adas.config import load_tuning
    tun = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
    p = car_params(tun.mount)
    m = load_car_model()
    world = World()                                # a 10 x 8 m hall with furniture along the walls only
    room_walls(world, -5.0, 5.0, -4.0, 4.0)
    for bx, by in ((-4.4, 3.4), (0.5, 3.6), (4.4, -3.4), (-1.0, -3.6), (4.5, 1.2), (-4.5, -1.5)):
        world.add(Box(bx, by, 0.4, 0.3))
    from sim.hw_worlds import _finish
    world = _finish(world)
    car = VirtualCar(world, p, (-2.5, -1.5, 0.0), threaded=False)
    car.m = dict(car.m, v_max=car.m["v_max"] * battery)
    car.last_cmd_t = 0.0
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=1360, seed=seed)
    pwm_est = SpeedEstimator(SpeedModel(v_max=m["v_max"], deadband=m["deadband"]))
    rf, ekf, icp = RangeFlow(), make_ekf(m, p.lidar_x, robust), Odometry()
    crng = np.random.default_rng(seed + 5)
    centre = m["servo_centre"]
    # a drive with speed steps, coasting and both turn directions, staying inside the room
    # (until t, throttle, servo offset: below centre = left)
    plan = [(0.5, 0, 0), (2.0, 110, 0), (3.5, 200, 0.0), (5.0, 0, 0), (7.0, 140, -22), (9.0, 200, -22),
            (10.0, 0, -22), (12.5, 130, 22), (14.0, 0, 22), (17.0, 160, -18), (18.0, 0, 0), (20.5, 120, 18),
            (22.0, 170, 18), (seconds, 0, 0)]
    dt, t, next_scan = 0.02, 0.0, 0.1
    rows, prev_icp, prev_icp_t = [], None, None
    th_prev = car.pose()[2]
    while t < seconds:
        u, off = next((u, o) for te, u, o in plan if t < te)
        servo = centre + off
        car.command(f"A {servo:.1f} {servo:.1f}", now=t)
        car.command(f"M {-int(u)}", now=t)
        ekf.command(t, u, servo)
        pwm_est.update(dt, u)
        ekf.predict(t)
        t += dt
        car.step_to(t)
        x, y, th, v, *_r, crashed = car.pose()
        w_true = (th - th_prev) / dt
        th_prev = th
        meas = None
        if t >= next_scan:
            next_scan += 0.1
            best, _ = lidar._raycast(x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th), th)
            r = best + lidar.rng.normal(0, 0.008, len(best))
            okb = np.isfinite(best) & (r >= 0.2) & (r < 12) & (lidar.rng.random(len(best)) > 0.04)
            xy = np.column_stack([r[okb] * np.cos(lidar.ccw[okb]), r[okb] * np.sin(lidar.ccw[okb])])
            meas = rf.update(xy, t, guess=(ekf.v, ekf.w * p.lidar_x, ekf.w))
            if meas is not None and corrupt and crng.random() < corrupt:
                # a bad scan match that still reports a small covariance (the case a covariance gate cannot see)
                meas = (meas[0] + crng.choice([-1, 1]) * crng.uniform(*mag), meas[1], meas[2] + crng.normal(0, 0.3), meas[3], meas[4])
            if meas is not None:
                ekf.correct(t, meas)
            pose = icp.update(np.column_stack([xy[:, 0], -xy[:, 1]]), t, ekf.v, 0.0)
            v_icp = None
            if prev_icp is not None:
                d = pose - prev_icp
                v_icp = (d[0] * math.cos(pose[2]) + d[1] * math.sin(pose[2])) / (t - prev_icp_t)
            prev_icp, prev_icp_t = pose.copy(), t
            rows.append((t, v, w_true, pwm_est.v, np.nan if v_icp is None else v_icp,
                         np.nan if meas is None else meas[0], np.nan if meas is None else meas[2], ekf.v, ekf.w))
        if crashed:
            break
    a = np.array(rows)
    ok = (a[:, 0] > 1.0) & (np.abs(a[:, 1]) > 0.05)   # moving only
    res = {"pwm model": {"v": stats(a[:, 3], a[:, 1], ok)},
           "icp diff": {"v": stats(a[:, 4], a[:, 1], ok)},
           "rf2o": {"v": stats(a[:, 5], a[:, 1], ok), "w": stats(a[:, 6], a[:, 2], ok)},
           "ekf": {"v": stats(a[:, 7], a[:, 1], ok), "w": stats(a[:, 8], a[:, 2], ok)}}
    return res, a, car.crash_count


# ------------------------------------------------------------------ 2. the real drive log
def log_run(path=None):
    from adas.aeb import SpeedEstimator, SpeedModel
    from adas.rf2o import RangeFlow
    from pi.drive_log import load
    from sim.hw_sim import load_car_model
    from sim.log_fit import command_series, hold, kinematics, reconstruct
    path = path or sorted(glob.glob(os.path.join(ROOT, "logs", "*log_drive*.jsonl.gz")), key=os.path.getsize)[-1]
    log = load(path)
    yaw = log["meta"]["tuning"]["mount"]["yaw_offset_deg"]
    m = load_car_model()
    servo, pwm = command_series(log)
    rows = reconstruct(log)                        # scan-matched path, y RIGHT (pi/scanmatch frame)
    tv, v_ref, w_ref, good = kinematics(rows)
    k = np.ones(5) / 5                             # +-0.25 s centred smoothing on top of kinematics' 3 scans
    v_ref = np.convolve(v_ref, k, mode="same")
    w_ref = -np.convolve(w_ref, k, mode="same")    # -> + = left
    good = good & (np.convolve(good.astype(float), k, mode="same") > 0.99)
    rf, ekf = RangeFlow(), make_ekf(m)
    pwm_est = SpeedEstimator(SpeedModel(v_max=m["v_max"], deadband=m["deadband"]))
    icp_v = np.full(len(tv), np.nan)
    t, ph = rows[:, 0], rows[:, 1:4]
    for i in range(1, len(t)):                     # the live version of icp diff: no smoothing
        dth = (ph[i, 2] + ph[i - 1, 2]) / 2
        icp_v[i - 1] = ((ph[i, 0] - ph[i - 1, 0]) * math.cos(dth) + (ph[i, 1] - ph[i - 1, 1]) * math.sin(dth)) / max(t[i] - t[i - 1], 1e-3)
    ev = {"t": [], "rf_v": [], "rf_w": [], "ekf_v": [], "ekf_w": [], "pwm_v": []}
    tc, last = None, None
    cmd_t = np.union1d(servo[:, 0], pwm[:, 0])
    ci = 0
    for (ts, ang, dist, _q) in log["scans"]:
        # feed commands and the pwm model up to this scan
        while ci < len(cmd_t) and cmd_t[ci] <= ts:
            tt = cmd_t[ci]
            u, s = float(hold(pwm, tt)), float(hold(servo, tt))
            ekf.command(tt, u, s)
            if last is not None:
                pwm_est.update(min(0.2, tt - last[0]), last[1])
            last = (tt, u)
            ci += 1
        ekf.predict(ts)
        a = -np.radians((np.asarray(ang) - yaw) % 360.0)          # CCW, y left
        d = np.asarray(dist)
        okd = d >= 0.2
        xy = np.column_stack([d[okd] * np.cos(a[okd]), d[okd] * np.sin(a[okd])])
        meas = rf.update(xy, ts, guess=(ekf.v, ekf.w * 0.12, ekf.w))
        if meas is not None:
            ekf.correct(ts, meas)
        ev["t"].append(ts)
        ev["rf_v"].append(np.nan if meas is None else meas[0])
        ev["rf_w"].append(np.nan if meas is None else meas[2])
        ev["ekf_v"].append(ekf.v)
        ev["ekf_w"].append(ekf.w)
        ev["pwm_v"].append(pwm_est.v)
    E = {k2: np.asarray(v2) for k2, v2 in ev.items()}
    # everything onto the reference's time stamps (scan mid-points); rf2o measures over the interval ending at
    # the scan, so compare at the interval mid-point
    at = lambda series, shift=0.0: np.interp(tv, E["t"] - shift, series)
    good = good & (np.abs(v_ref) > 0.05)           # moving only: parked, every estimator trivially reads 0
    res = {"pwm model": {"v": stats(at(E["pwm_v"]), v_ref, good)},
           "icp diff": {"v": stats(icp_v, v_ref, good)},
           "rf2o": {"v": stats(at(E["rf_v"], 0.05), v_ref, good), "w": stats(at(E["rf_w"], 0.05), w_ref, good)},
           "ekf": {"v": stats(at(E["ekf_v"]), v_ref, good), "w": stats(at(E["ekf_w"]), w_ref, good)}}
    series = {"t": tv, "ref_v": v_ref, "ref_w": w_ref, "good": good, "pwm_v": at(E["pwm_v"]), "icp_v": icp_v,
              "rf_v": at(E["rf_v"], 0.05), "ekf_v": at(E["ekf_v"]), "rf_w": at(E["rf_w"], 0.05), "ekf_w": at(E["ekf_w"])}
    return res, series, os.path.basename(path)


def main():
    tw, a, crashes = twin_run()
    moving = a[:, 1] > 0.05
    print(f"twin drive: {crashes} crashes, moving on {moving.mean() * 100:.0f} % of scans, "
          f"top speed {a[:, 1].max():.2f} m/s, yaw rate up to {math.degrees(np.abs(a[:, 2]).max()):.0f} deg/s")
    tw80, _a80, _c80 = twin_run(battery=0.8)
    lg, s, name = log_run()
    for title, res in (("digital twin (true speed known)", tw), ("digital twin, car 20 % slower than its model (battery)", tw80),
                       (f"real drive {name} (vs smoothed scan-matched path)", lg)):
        print(title)
        for est, r in res.items():
            line = f"  {est:10s} speed error: RMSE {r['v']['rmse'] * 100:5.1f} cm/s, median {r['v']['median_abs'] * 100:4.1f} cm/s"
            if "w" in r:
                line += f" | yaw rate: RMSE {math.degrees(r['w']['rmse']):5.1f} deg/s, median {math.degrees(r['w']['median_abs']):4.1f} deg/s"
            print(line)
    from sim.report import style
    plt = style()
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    g = s["good"]
    tt = s["t"] - s["t"][0]
    axes[0].plot(tt, np.where(g, s["ref_v"], np.nan), color="white", lw=2.5, label="reference (smoothed scan matching)")
    axes[0].plot(tt, s["pwm_v"], color="#94a3b8", lw=1, label="throttle model (current relay)")
    axes[0].plot(tt, s["icp_v"], color="#f87171", lw=0.6, alpha=0.6, label="ICP differentiated")
    axes[0].plot(tt, s["rf_v"], color="#facc15", lw=0.8, alpha=0.8, label="RF2O range flow")
    axes[0].plot(tt, s["ekf_v"], color="#2dd4bf", lw=1.6, label="EKF: car model + RF2O")
    axes[0].set_ylabel("speed (m/s)")
    axes[0].set_ylim(-0.6, 1.0)
    axes[0].legend(frameon=False, fontsize=8, ncol=3)
    axes[1].plot(tt, np.degrees(np.where(g, s["ref_w"], np.nan)), color="white", lw=2.5)
    axes[1].plot(tt, np.degrees(s["rf_w"]), color="#facc15", lw=0.8, alpha=0.8)
    axes[1].plot(tt, np.degrees(s["ekf_w"]), color="#2dd4bf", lw=1.6)
    axes[1].set_ylabel("yaw rate (deg/s)")
    axes[1].set_xlabel("time in the real drive (s)")
    fig.suptitle(f"Speed and yaw rate on the real drive: EKF speed RMSE {lg['ekf']['v']['rmse'] * 100:.1f} cm/s "
                 f"vs throttle model {lg['pwm model']['v']['rmse'] * 100:.1f} cm/s")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "reports", "odometry.png"), dpi=130)
    json.dump({"twin": tw, "twin_battery_80": tw80, "log": lg, "log_file": name},
              open(os.path.join(ROOT, "models", "odometry_eval.json"), "w"), indent=1)
    print("wrote reports/odometry.png, models/odometry_eval.json")


if __name__ == "__main__":
    main()
