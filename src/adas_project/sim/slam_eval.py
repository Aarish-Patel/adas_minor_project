"""Loop closure on the twin: drive a loop in a room with a badly drifting odometry and see the pose graph pull it straight.

    python -m sim.slam_eval          # writes reports/slam.png and models/slam.json

The car is placed along a closed route (out, round the furniture, back to the start); the scans are the twin's LiDAR at the true pose;
the front-end pose is the true motion with a scale error (3 %), a yaw bias (0.012 rad per metre, ~0.7 deg/m) and noise, i.e. far worse
than the real scan matcher (1.7 %), so the effect of the loop closure is easy to see. Reported: position error of the final pose and
of the whole trajectory, before and after (adas/submaps.py).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def route(step=0.05):
    """A closed route in the 'room' world: out along the wall, up, back, ending on the start point and heading."""
    way = [(0.0, 0.0), (2.9, 0.0), (2.9, 1.15), (0.2, 1.15), (0.2, -0.6), (0.0, -0.6), (0.0, 0.0)]
    pts = []
    for (x0, y0), (x1, y1) in zip(way[:-1], way[1:]):
        L = math.hypot(x1 - x0, y1 - y0)
        for s in np.arange(0.0, L, step):
            pts.append((x0 + (x1 - x0) * s / L, y0 + (y1 - y0) * s / L))
    pts.append(way[-1])
    pts = np.array(pts)
    th = np.zeros(len(pts))
    d = np.diff(pts, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    th[:-1] = ang
    th[-1] = ang[-1]
    # smooth the heading over ~0.25 m so corners are turns, not jumps
    k = 5
    unwrapped = np.unwrap(th)
    sm = np.convolve(np.pad(unwrapped, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    return pts, sm


def run(seed=0, scale_err=1.03, yaw_bias=0.012, noise=(0.003, 0.002), closure=True, verbose=True):
    from adas.submaps import PoseGraphMap, between, compose
    from adas.config import load_tuning
    from pi.relay_assists import car_params
    from sim.hw_sim import SimLidar, VirtualCar
    from sim.hw_worlds import WORLDS
    world, start = WORLDS["room"]()
    tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
    p = car_params(tun.mount)
    car = VirtualCar(world, p, start, threaded=False)
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
    pts_w, th_w = route()
    rng = np.random.default_rng(seed)
    est = np.zeros(3)
    g = PoseGraphMap()
    truth, est_trace, corr_trace, node_true = [], [], [], []
    prev = np.array([pts_w[0, 0], pts_w[0, 1], th_w[0]])
    for k in range(len(pts_w)):
        t_pose = np.array([pts_w[k, 0], pts_w[k, 1], th_w[k]])
        d = between(prev, t_pose)
        ds = math.hypot(d[0], d[1])
        step = np.array([d[0] * scale_err + rng.normal(0, noise[0]) * (ds > 0), d[1] + rng.normal(0, noise[0]) * (ds > 0),
                         d[2] + yaw_bias * ds + rng.normal(0, noise[1]) * (ds > 0)])
        est = compose(est, step)
        prev = t_pose
        if k % 2:
            continue
        car.x, car.y, car.th = float(t_pose[0]), float(t_pose[1]), float(t_pose[2])
        ox, oy = t_pose[0] + p.lidar_x * math.cos(t_pose[2]), t_pose[1] + p.lidar_x * math.sin(t_pose[2])
        best, _ = lidar._raycast(ox, oy, t_pose[2])
        r = best + lidar.rng.normal(0, 0.008, len(best))
        ok = np.isfinite(best) & (r >= 0.2) & (r < 12)
        xy = np.column_stack([p.lidar_x + r[ok] * np.cos(lidar.ccw[ok]), r[ok] * np.sin(lidar.ccw[ok])])
        n_before = len(g.est)
        pose = g.add_scan(est, xy) if closure else est
        if closure and len(g.est) > n_before:
            node_true.append(t_pose.copy())
        truth.append(t_pose)
        est_trace.append(est.copy())
        corr_trace.append(np.array(pose))
    truth, est_trace, corr_trace = np.array(truth), np.array(est_trace), np.array(corr_trace)
    e_est = np.hypot(*(est_trace[:, :2] - truth[:, :2]).T)
    e_cor = np.hypot(*(corr_trace[:, :2] - truth[:, :2]).T)
    res = {"final_err_odometry_cm": float(100 * e_est[-1]), "final_err_corrected_cm": float(100 * e_cor[-1]),
           "mean_err_odometry_cm": float(100 * e_est.mean()), "mean_err_corrected_cm": float(100 * e_cor.mean()),
           "final_heading_err_odometry_deg": float(abs(math.degrees(est_trace[-1, 2] - truth[-1, 2]))),
           "final_heading_err_corrected_deg": float(abs(math.degrees((corr_trace[-1, 2] - truth[-1, 2] + math.pi) % (2 * math.pi) - math.pi))),
           "loops": len(g.loops), "nodes": len(g.nodes), "submaps": len(g.submaps), "corrections": g.corrections}
    if verbose:
        print(f"route {len(truth)} scans, {len(g.nodes)} nodes, {len(g.submaps)} submaps, {len(g.loops)} loop closures, {g.corrections} optimisations")
        print(f"final position error: odometry {res['final_err_odometry_cm']:.1f} cm -> corrected {res['final_err_corrected_cm']:.1f} cm; "
              f"heading {res['final_heading_err_odometry_deg']:.1f} -> {res['final_heading_err_corrected_deg']:.1f} deg")
        print(f"mean position error along the route: odometry {res['mean_err_odometry_cm']:.1f} cm -> corrected {res['mean_err_corrected_cm']:.1f} cm")
    return res, truth, est_trace, corr_trace, g, world


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    print("real-like drift (1.7 % scale, 0.3 deg/m yaw):")
    res_real = run(scale_err=1.017, yaw_bias=0.005)[0]
    print("harsh drift (3 % scale, 0.7 deg/m yaw):")
    res, truth, est, cor, g, world = run()
    res = {"real_like": res_real, "harsh": res}
    r0 = res["harsh"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for x1, y1, x2, y2 in world.segments():
        ax.plot([x1, x2], [y1, y2], color="0.6", lw=1)
    ax.plot(truth[:, 0], truth[:, 1], color="k", lw=2, label="true route")
    ax.plot(est[:, 0], est[:, 1], color="tab:red", lw=1.5, label="odometry (drifting)")
    nodes = np.array(g.nodes)
    ax.plot(nodes[:, 0], nodes[:, 1], color="tab:blue", lw=1.5, label="pose graph after loop closure")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"loop closure (harsh drift): end error {r0['final_err_odometry_cm']:.0f} cm -> {r0['final_err_corrected_cm']:.0f} cm")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "..", "reports"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "reports", "slam.png"), dpi=130)
    json.dump(res, open(os.path.join(HERE, "..", "models", "slam.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
