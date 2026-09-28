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
    from adas.intent_net import DriverProfile, features, HORIZON_S
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
    t, next_scan, hist, rows = 0.0, 0.0, [], []
    prof = DriverProfile()
    pts = np.empty((0, 2))
    while t < T_MAX:
        x, y, th, v, _, _, crashed = car.pose()
        if crashed or math.hypot(x - goal[0], y - goal[1]) < 0.35:
            break
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
        f = features(hist, u, v, pts, p, ra.centre, K, profile=prof)
        if f is not None:
            prof.update(hist, f[len(f) - 5] * 1.5)
        rows.append((f, s, driver.lapsed(t)))
        car.command(f"A {s:.1f} {s:.1f}", now=t)
        car.command(f"M {-int(u)}", now=t)
        t += DT
        car.step_to(t)
    # label every THREAT moment (the current arc touches something within 1.2 m) by what really happened:
    # did this driver, with no ADAS, crash within the next 2 s?
    crash_t = t if car.crash_count else None
    X, Y, L = [], [], []
    for i, (f, s, lap) in enumerate(rows):
        if f is None or f[len(f) - 5] * 1.5 > 1.2:              # feature: free distance on the current arc
            continue
        ti = i * DT
        X.append(f)
        Y.append(1 if (crash_t is not None and crash_t - ti <= 2.0) else 0)
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
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    net = MLPClassifier(hidden_layer_sizes=(32, 32), activation="tanh", alpha=1e-3, max_iter=500, early_stopping=True,
                        random_state=0)
    net.fit((Xtr - mu) / sd, Ytr)
    prob = net.predict_proba((Xte - mu) / sd)[:, 1]
    auc = float(roc_auc_score(Yte, prob)) if len(set(Yte)) > 1 else float("nan")
    # at the threshold the ADAS uses (hold back only below 0.2): how many real crashes would be trusted to the driver?
    hold = prob < 0.2
    report = {"train_threat_moments": int(len(Xtr)), "test_threat_moments": int(len(Xte)),
              "train_crash_fraction": float(Ytr.mean()), "test_auc": auc,
              "held_back_fraction": float(hold.mean()),
              "crash_moments_wrongly_held_back": int(((Yte == 1) & hold).sum()), "crash_moments": int((Yte == 1).sum())}
    out = {"kind": "classifier", "W": [w.tolist() for w in net.coefs_], "b": [b.tolist() for b in net.intercepts_],
           "mu": mu.tolist(), "sd": sd.tolist(), "activation": "tanh", "report": report}
    for path in (os.path.join(ROOT, "models", "intent_net.json"), os.path.join(ROOT, "pi", "intent_net.json")):
        json.dump(out, open(path, "w"))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
