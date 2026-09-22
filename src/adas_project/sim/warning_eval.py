"""Does knowing the driver's intent make the warnings better?

    python -m sim.warning_eval [train_episodes] [test_episodes]

Each episode is a virtual driver who sometimes loses attention, driving with the
ADAS in ADVISORY mode (it warns but never intervenes), so we see what would
really have happened. Three warning systems are scored on the same held-out drives:

    physics-only    assumes the current steering is kept
    intent blend    checks the paths the driver is likely to take, weighted by probability
    adaptive        a model trained on earlier drives that predicts the chance of a hazard
                    from the physics risk, the intent probabilities and driver behaviour

Ground truth: a hazard is a real collision or a near miss (body within 3 cm of an
obstacle at speed). A warning is a run of risk above a threshold; sweeping the
threshold gives the trade-off between catching hazards early and false alarms.
A hazard only counts as caught if the warning came at least MIN_LEAD seconds before it.
"""

import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from adas.intent import FEATURES, IntentEstimator, IntentModel
from adas.warning import RiskScorer, ttc_for_steer, risk_from_ttc

from .drivers import HumanLikeDriver
from .intent_data import MODEL_DIR, random_world
from .simulator import Simulator

WINDOW = 2.0          # a warning is "true" if a hazard follows within this many seconds
NEAR_MISS = 0.03      # m
MIN_LEAD = 0.4        # s: a warning later than this before the hazard does not count as caught
LOOKAHEAD = 2.5       # s: the learned model predicts "hazard within this long ..."
LEAD_FLOOR = 0.2      # ... but not later than this (too late to act on)
TICK = 0.05

EXTRA = ["risk_base", "ttc_base", "risk_blend", "ttc_cur", "ttc_str", "ttc_left", "ttc_right",
         "ttc_brake", "p_straight", "p_left", "p_right", "p_brake"]
COLUMNS = FEATURES + EXTRA


def _cap(x, hi=5.0):
    return min(x, hi) if math.isfinite(x) else hi


def run_episode(seed, duration=40.0):
    model = IntentModel.load(os.path.join(MODEL_DIR, "intent.joblib"))
    rng = np.random.default_rng(seed)
    world = random_world(rng)
    sim = Simulator(world, (0.0, 0.0, 0.0), adas_on="advisory", seed=seed)
    estimator = IntentEstimator(model)
    scorer = RiskScorer(sim.p)
    driver = HumanLikeDriver(seed=seed)

    rows, times = [], []
    next_eval = 0.0
    while sim.t < duration and not sim.car.collided:
        steer, pwm = driver(sim.t, sim)
        sim.step(steer, pwm)
        probs = estimator.update(sim.t, steer, pwm, sim.adas)
        if sim.t >= next_eval and estimator.row is not None:
            next_eval = sim.t + TICK
            v = sim.adas.estimator.v
            d = 1 if v >= 0 else -1
            speed = abs(v)
            pts, tracks, p = sim.adas.points, sim.adas.tracks, sim.p
            rb, ttc_b = scorer.baseline(sim.adas, steer, pwm)
            rblend, _ = scorer.intent_aware(sim.adas, steer, pwm, probs)
            ttcs = [_cap(ttc_for_steer(pts, tracks, st, speed, d, p)) for st in
                    (steer, 0.0, scorer.class_steer["left"], scorer.class_steer["right"])]
            ttc_brake = _cap(ttc_for_steer(pts, tracks, steer, speed * scorer.brake_speed_factor, d, p))
            row = [estimator.row[f] for f in FEATURES]
            row += [rb, _cap(ttc_b), rblend, *ttcs, ttc_brake,
                    probs["straight"], probs["left"], probs["right"], probs["brake"]]
            rows.append(row)
            times.append(sim.t)

    events = hazard_times(sim)
    X = np.array(rows, dtype=float) if rows else np.empty((0, len(COLUMNS)))
    t = np.array(times)
    y = np.zeros(len(t), dtype=int)
    for h in events:
        y[(t < h - LEAD_FLOOR) & (t >= h - LOOKAHEAD)] = 1
    return seed, X, t, y, events, float(sim.t)


def hazard_times(sim):
    """Times of collisions and near misses (at speed), merged if within 1 s."""
    log = np.array(sim.log)
    if len(log) == 0:
        return []
    t, v, clr = log[:, 0], np.abs(log[:, 4]), log[:, 9]
    cand = list(t[(clr < NEAR_MISS) & (v > 0.15)])
    if sim.car.collided and sim.car.impact_speed >= 0.05:
        cand.append(sim.t)
    events = []
    for x in sorted(cand):
        if not events or x - events[-1] > 1.0:
            events.append(float(x))
    return events


def score(series_list, threshold):
    """series_list: [(t, risk, events, t_end)] -> detection, mean lead, false alarms/min."""
    hits, leads, n_haz, false_alarms, minutes = 0, [], 0, 0, 0.0
    for t, r, events, t_end in series_list:
        if len(t) == 0:
            continue
        minutes += t_end / 60.0
        above = r >= threshold
        starts = [t[i] for i in range(len(t)) if above[i] and (i == 0 or not above[i - 1])]
        for h in events:
            n_haz += 1
            pre = [s for s in starts if h - 2.5 <= s <= h - MIN_LEAD]
            live = [ti for ti, a in zip(t, above) if a and h - 2.5 <= ti <= h - MIN_LEAD]
            if pre or live:
                hits += 1
                leads.append(h - (min(pre) if pre else min(live)))
        for s in starts:
            if not any(s <= h <= s + WINDOW for h in events):
                false_alarms += 1
    return {"threshold": float(threshold), "detection": hits / n_haz if n_haz else 0.0,
            "lead_time": float(np.mean(leads)) if leads else 0.0,
            "false_alarms_per_min": false_alarms / minutes if minutes else 0.0, "hazards": n_haz}


def monotonic_constraints():
    """More physical risk (or less distance / time to collision) may never LOWER the warning."""
    up = {"risk_base", "risk_blend"}
    down = {"ttc_base", "ttc_cur", "ttc_str", "ttc_left", "ttc_right", "ttc_brake", "D", "ttc"}
    return [1 if c in up else -1 if c in down else 0 for c in COLUMNS]


def train_adaptive(train):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([x for _, x, _, _, _, _ in train if len(x)])
    y = np.concatenate([yy for _, x, _, yy, _, _ in train if len(x)])
    clf = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_depth=5, min_samples_leaf=60,
                                         l2_regularization=1.0, monotonic_cst=monotonic_constraints(),
                                         random_state=0)
    clf.fit(X, y)
    return clf


CACHE = os.path.join(MODEL_DIR, "warning_data.pkl")


def combos(rb, p):
    """Ways of combining the physics risk with the learned hazard probability."""
    return {"physics": rb, "adaptive": p, "gated": np.sqrt(np.clip(rb, 0, 1) * np.clip(p, 0, 1)),
            "gated_hard": np.clip(rb, 0, 1) * np.clip(p, 0, 1)}


def auc_low_fa(curve, budget=8.0):
    """Mean detection over false-alarm budgets 1..budget (higher is better; unreachable budgets score 0)."""
    total = 0.0
    budgets = np.linspace(1.0, budget, 8)
    for b in budgets:
        rows = [r["detection"] for r in curve if r["false_alarms_per_min"] <= b]
        total += max(rows) if rows else 0.0
    return total / len(budgets)


def main(n_train=400, n_test=100, use_cache=False):
    import pickle
    train_seeds = list(range(9000, 9000 + n_train))
    test_seeds = list(range(9500, 9500 + n_test))
    if use_cache and os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            train, test = pickle.load(f)
        print("using cached drives")
    else:
        print(f"collecting {n_train} training drives and {n_test} test drives ...")
        with ProcessPoolExecutor() as pool:
            train = list(pool.map(run_episode, train_seeds, chunksize=1))
            test = list(pool.map(run_episode, test_seeds, chunksize=1))
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump((train, test), f)

    fit, val = train[:-60], train[-60:]
    print(f"fit drives {len(fit)} ({sum(len(e) for *_, e, _ in fit)} hazards), validation {len(val)}, "
          f"test {len(test)} ({sum(len(e) for *_, e, _ in test)} hazards in {sum(t for *_, t in test) / 60.0:.1f} min)")

    clf = train_adaptive(fit)
    col = {name: i for i, name in enumerate(COLUMNS)}

    def systems_for(episodes):
        out = {k: [] for k in combos(np.zeros(1), np.zeros(1))}
        for _, X, t, y, ev, te in episodes:
            if len(X) == 0:
                for k in out:
                    out[k].append((t, np.empty(0), ev, te))
                continue
            for k, r in combos(X[:, col["risk_base"]], clf.predict_proba(X)[:, 1]).items():
                out[k].append((t, r, ev, te))
        out["blend"] = [(t, X[:, col["risk_blend"]] if len(X) else np.empty(0), ev, te) for _, X, t, y, ev, te in episodes]
        return out

    thresholds = [round(x, 3) for x in np.linspace(0.05, 0.98, 32)]
    val_sys = systems_for(val)
    val_score = {k: auc_low_fa([score(s, th) for th in thresholds]) for k, s in val_sys.items()}
    print("validation score (mean detection at 1-8 false alarms/min):",
          {k: round(v, 3) for k, v in val_score.items()})
    best = max((k for k in val_score if k != "physics"), key=lambda k: val_score[k])
    print("best learned combination on validation:", best)

    # final model on everything except test
    clf = train_adaptive(train)
    import joblib
    joblib.dump({"clf": clf, "columns": COLUMNS, "combine": best}, os.path.join(MODEL_DIR, "adaptive_warning.joblib"))
    systems = systems_for(test)
    curves = {name: [score(s, th) for th in thresholds] for name, s in systems.items()}

    for name in ("physics", "adaptive", best):
        print(f"\n{name}")
        print(f"{'thr':>6}{'detect':>8}{'lead s':>8}{'false/min':>11}")
        for r in curves[name][::4]:
            print(f"{r['threshold']:6.2f}{r['detection']:8.2f}{r['lead_time']:8.2f}{r['false_alarms_per_min']:11.2f}")

    def at_fa(name, target):
        best_r = None
        for r in curves[name]:
            if r["false_alarms_per_min"] <= target and (best_r is None or r["detection"] > best_r["detection"]):
                best_r = r
        return best_r

    print("\ndetection rate at a fixed false-alarm budget (test drives):")
    for target in (1.0, 2.0, 4.0, 6.0, 8.0, 12.0):
        line = f"  <= {target:>4.0f} false alarms/min:"
        for name in ("physics", "blend", "adaptive", best):
            r = at_fa(name, target)
            line += f"  {name} {r['detection']:.2f}" if r else f"  {name} n/a"
        print(line)

    out = {"train_episodes": n_train, "test_episodes": n_test, "curves": curves, "best": best,
           "validation_score": val_score,
           "hazards_test": sum(len(e) for *_, e, _ in test),
           "minutes_test": sum(t for *_, t in test) / 60.0}
    with open(os.path.join(MODEL_DIR, "warning_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved models/warning_eval.json")
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    nums = [int(x) for x in args]
    main(*(nums if nums else (400, 100)), use_cache="--cached" in sys.argv)
