"""Generate driving data, train the driver-intent model and evaluate it.

    python -m sim.intent_data [episodes]

Episodes are split BY EPISODE (never random rows), so the model is tested on
arenas and drives it has not seen. The learned model is compared with the
"physics" baseline that just assumes the driver keeps doing what they do now.
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from adas.intent import CLASSES, FEATURES, IntentLogger, IntentModel, make_labels, rows_to_matrix

from .drivers import HumanLikeDriver
from .simulator import Simulator
from .world import Box, Cone, MovingCircle, World

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def random_world(rng, pedestrians=True):
    w = World()
    w.add_room(-0.6, 5.6, -1.7, 1.7)
    for _ in range(int(rng.integers(3, 8))):
        w.add(Cone(rng.uniform(0.8, 5.0), rng.uniform(-1.3, 1.3), 0.03))
    for _ in range(int(rng.integers(1, 4))):
        w.add(Box(rng.uniform(1.2, 5.0), rng.uniform(-1.2, 1.2), rng.uniform(0.15, 0.4),
                  rng.uniform(0.1, 0.25), rng.uniform(0, math.pi)))
    if pedestrians:
        for _ in range(int(rng.integers(0, 3))):
            side = rng.choice([-1, 1])
            w.add(MovingCircle(rng.uniform(1.5, 4.5), side * rng.uniform(1.0, 1.5), 0.0,
                               -side * rng.uniform(0.1, 0.4), 0.05))
    return w


def episode(seed, duration=30.0, mode="advisory", lapses=True):
    rng = np.random.default_rng(seed)
    world = random_world(rng)
    logger = IntentLogger(rate_hz=20.0)
    sim = Simulator(world, (0.0, 0.0, 0.0), adas_on=mode, logger=logger, seed=seed)
    sim.run(HumanLikeDriver(seed=seed, lapses=lapses), duration, stop_when_stopped=False)
    return sim, logger.rows


def _episode_rows(seed):
    sim, rows = episode(seed)
    return seed, rows, make_labels(rows), bool(sim.car.collided)


def build_dataset(seeds, workers=None):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_episode_rows, seeds, chunksize=1))


def baseline_predict(rows, thr=0.25):
    """Physics-only guess: the driver keeps doing what they are doing right now."""
    out = []
    for r in rows:
        if r["pwm_rate"] * 500 < -60.0 and r["pwm"] > 0.2:
            out.append("brake")
        elif r["steer"] > thr:
            out.append("left")
        elif r["steer"] < -thr:
            out.append("right")
        else:
            out.append("straight")
    return out


def f1_report(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    rep = classification_report(y_true, y_pred, labels=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=CLASSES)
    return acc, rep, cm


def main(n_episodes=120):
    seeds = list(range(5000, 5000 + n_episodes))
    print(f"generating {n_episodes} driving episodes ...")
    data = build_dataset(seeds)

    split = int(len(data) * 0.7)
    train, test = data[:split], data[split:]
    tr_rows = [r for _, rows, _, _ in train for r in rows]
    tr_lab = [l for _, _, labs, _ in train for l in labs]
    te_rows = [r for _, rows, _, _ in test for r in rows]
    te_lab = [l for _, _, labs, _ in test for l in labs]

    model = IntentModel().fit(tr_rows, tr_lab)
    keep = [i for i, l in enumerate(te_lab) if l is not None]
    y_true = [te_lab[i] for i in keep]
    y_model = [CLASSES[int(k)] for k in model.predict_batch([te_rows[i] for i in keep])]
    y_base = baseline_predict([te_rows[i] for i in keep])

    acc_m, rep_m, cm_m = f1_report(y_true, y_model)
    acc_b, rep_b, cm_b = f1_report(y_true, y_base)
    dist = {c: y_true.count(c) for c in CLASSES}

    print(f"train episodes {len(train)}, test episodes {len(test)}  (test samples {len(y_true)})")
    print("class balance (test):", dist)
    print(f"{'':10s}{'accuracy':>10s}{'macro F1':>10s}")
    print(f"{'baseline':10s}{acc_b:10.3f}{rep_b['macro avg']['f1-score']:10.3f}")
    print(f"{'learned':10s}{acc_m:10.3f}{rep_m['macro avg']['f1-score']:10.3f}")
    print("per-class F1 (learned):", {c: round(rep_m[c]['f1-score'], 3) for c in CLASSES})

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "intent.joblib"))
    metrics = {
        "episodes": n_episodes, "test_samples": len(y_true), "class_balance": dist,
        "baseline": {"accuracy": acc_b, "macro_f1": rep_b["macro avg"]["f1-score"],
                     "per_class_f1": {c: rep_b[c]["f1-score"] for c in CLASSES}, "confusion": cm_b.tolist()},
        "learned": {"accuracy": acc_m, "macro_f1": rep_m["macro avg"]["f1-score"],
                    "per_class_f1": {c: rep_m[c]["f1-score"] for c in CLASSES}, "confusion": cm_m.tolist()},
    }
    with open(os.path.join(MODEL_DIR, "intent_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("saved models/intent.joblib and models/intent_metrics.json")
    return metrics


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
