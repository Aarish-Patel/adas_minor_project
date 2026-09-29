"""Train the learned intent model (adas/intent_net.py) on simulated human driving, test on held-out scenarios.

    python -m sim.train_intent_net [train_runs]   -> models/intent_net.json (+ pi/intent_net.json for the car)

Data: the Monte Carlo's human drivers (lapsing and late styles) driving WITHOUT ADAS in random rooms; at every tick
the features the car can observe and the stick position 0.5 s later. Real drive logs can be added the same way
once there are enough of them (the relay records the stick and every scan).
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


def collect(args):
    seed, style = args
    from adas.config import load_tuning
    from adas.intent_net import DriverProfile, features, free_now
    from pi.relay_assists import K_CURV_PER_SERVO_DEG as K, RelayAssists, apply_car_model
    from sim.hw_sim import SimLidar, VirtualCar
    from sim.relay_mc import DT, SCAN_DT, T_MAX, HumanDriver, scenario

    tun = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
    apply_car_model(tun)
    world, goal, _ = scenario(seed)
    ra = RelayAssists(tun)
    p = ra.p
    car = VirtualCar(world, p, (0.0, 0.0, 0.0), threaded=False)
    car.last_cmd_t = 0.0
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)   # the same sensing the relay and MC see
    driver = HumanDriver(world, goal, np.random.default_rng(seed + 1000), p, K, ra.centre, style)
    from sim.relay_mc import NEAR_MISS_M
    t, next_scan, hist, pwm_hist, rows, clear = 0.0, 0.0, [], [], [], []
    prof = DriverProfile()
    pts = np.empty((0, 2))
    while t < T_MAX:
        x, y, th, v, _, _, crashed = car.pose()
        if crashed or math.hypot(x - goal[0], y - goal[1]) < 0.35:
            break
        clear.append(world.clearance(x, y, th, p))
        if t >= next_scan:
            next_scan += SCAN_DT
            ox, oy = x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th)
            best, _ = lidar._raycast(ox, oy, th)
            r = best + lidar.rng.normal(0, 0.008, len(best))             # range noise and dropouts, as in the MC
            ok = np.isfinite(best) & (r >= 0.2) & (r < 12) & (lidar.rng.random(len(best)) > 0.04)
            cw = (-np.degrees(lidar.ccw)) % 360
            cw = np.where(cw > 180, cw - 360, cw)
            pts = RelayAssists.points_vehicle_frame([(a, d) for a, d, o in zip(cw, r, ok) if o], p.lidar_x)
        s, u = driver.command(t, (x, y, th), v)
        hist.append(s)
        hist = hist[-80:]
        pwm_hist = (pwm_hist + [u])[-40:]
        f = features(hist, u, v, pts, p, ra.centre, K, profile=prof, pwm_hist=pwm_hist)
        if f is not None:
            prof.update(hist, free_now(f))
        rows.append((f, s, driver.lapsed(t)))
        car.command(f"A {s:.1f} {s:.1f}", now=t)
        car.command(f"M {-int(u)}", now=t)
        t += DT
        car.step_to(t)
    # label every THREAT moment (the current arc touches something within 1.2 m) by what really happened: did this
    # driver, with no ADAS, crash - or come within NEAR_MISS_M (a near miss) - within the next 2 s? (the Monte
    # Carlo's ground truth for a needed intervention)
    crash_t = t if car.crash_count else None
    clear = np.asarray(clear)
    n2 = int(round(2.0 / DT))
    X, Y, L = [], [], []
    for i, (f, s, lap) in enumerate(rows):
        if f is None or free_now(f) > 1.2:
            continue
        ti = i * DT
        crash = crash_t is not None and crash_t - ti <= 2.0
        near = len(clear[i:i + n2]) and clear[i:i + n2].min() < NEAR_MISS_M
        X.append(f)
        Y.append(1 if (crash or near) else 0)
        L.append(lap)
    return X, Y, L


def main(train_runs=60, test_runs=20):
    from sim.relay_mc import DEFAULT_STYLES
    train_jobs = [(100 + i, st) for i in range(train_runs) for st in DEFAULT_STYLES]
    test_jobs = [(500 + i, st) for i in range(test_runs) for st in DEFAULT_STYLES]
    with ProcessPoolExecutor() as ex:
        tr = list(ex.map(collect, train_jobs, chunksize=2))
        te = list(ex.map(collect, test_jobs, chunksize=2))
    Xtr = np.array([x for X, _, _ in tr for x in X]); Ytr = np.array([y for _, Y, _ in tr for y in Y])
    Xte = np.array([x for X, _, _ in te for x in X]); Yte = np.array([y for _, Y, _ in te for y in Y])
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score
    # gradient-boosted trees: on these tabular features they beat the MLP (average precision 0.48 vs 0.38 on held-out
    # rooms), and 100 trees of depth 3 cost ~0.05 ms per tick in numpy on the car (adas/intent_net.py)
    gbm = HistGradientBoostingClassifier(max_iter=100, max_depth=3, learning_rate=0.15, random_state=0)
    gbm.fit(Xtr, Ytr)
    prob = gbm.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(Yte, prob)) if len(set(Yte)) > 1 else float("nan")
    hold = prob < 0.5              # the relay's trust threshold
    report = {"model": "gradient-boosted trees (100 x depth 3)", "features": int(Xtr.shape[1]),
              "train_threat_moments": int(len(Xtr)), "test_threat_moments": int(len(Xte)),
              "train_crash_fraction": float(Ytr.mean()), "test_auc": auc,
              "test_average_precision": float(average_precision_score(Yte, prob)),
              "held_back_fraction": float(hold.mean()),
              "crash_moments_wrongly_held_back": int(((Yte == 1) & hold).sum()), "crash_moments": int((Yte == 1).sum())}
    out = export_trees(gbm)
    out["report"] = report
    for path in (os.path.join(ROOT, "models", "intent_net.json"), os.path.join(ROOT, "pi", "intent_net.json")):
        json.dump(out, open(path, "w"))
    # the car's numpy evaluator must give exactly sklearn's answer
    from adas.intent_net import IntentNet
    net = IntentNet(os.path.join(ROOT, "pi", "intent_net.json"))
    mine = np.array([net.crash_probability(x) for x in Xte[:500]])
    report["numpy_vs_sklearn_max_diff"] = float(np.abs(mine - prob[:500]).max())
    print(json.dumps(report, indent=1))


def export_trees(gbm):
    """HistGradientBoostingClassifier -> padded node arrays [tree, node] for adas/intent_net.IntentNet."""
    trees = [pred[0].nodes for pred in gbm._predictors]
    n = max(len(t) for t in trees)
    T = len(trees)
    feat, thr = np.zeros((T, n), int), np.zeros((T, n))
    left, right = np.zeros((T, n), int), np.zeros((T, n), int)
    leaf, value = np.ones((T, n), bool), np.zeros((T, n))
    depth = 0
    for i, nodes in enumerate(trees):
        k = len(nodes)
        feat[i, :k], thr[i, :k] = nodes["feature_idx"], nodes["num_threshold"]
        left[i, :k], right[i, :k] = nodes["left"], nodes["right"]
        leaf[i, :k], value[i, :k] = nodes["is_leaf"].astype(bool), nodes["value"]
        depth = max(depth, int(nodes["depth"].max()))
    return {"kind": "trees", "feature": feat.tolist(), "threshold": thr.tolist(), "left": left.tolist(),
            "right": right.tolist(), "leaf": leaf.tolist(), "value": value.tolist(),
            "baseline": float(np.ravel(gbm._baseline_prediction)[0]), "max_depth": depth}


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
