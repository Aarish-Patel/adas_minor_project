"""Train the 'would this takeover be needless?' model (TODO W5) on the decision dataset from sim/needless_data.py.

    python -m sim.train_needless            # writes pi/needless_v1.json and models/needless_report.json

Target: needed = the same driver, left alone, would come within 2 cm of something in the next 2 s. Features: the car's own v3 window
features at the moment ADAS would take over. Compared against using the v3 crash risk directly as the score (what the intent-aware
system does today, via v2/v3 thresholds). Trees are kept small (Pi-light); everything is evaluated on drives (seeds) not used for training.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def load(name):
    d = np.load(os.path.join(HERE, "..", "data", "needless", name), allow_pickle=False)
    return {k: d[k] for k in d.files}


def metrics(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score
    ok = ~np.isnan(p)
    return {"auc": float(roc_auc_score(y[ok], p[ok])), "ap": float(average_precision_score(y[ok], p[ok])), "n": int(ok.sum()),
            "positives": int(y[ok].sum())}


def operating_point(y, p, max_missed=0.0):
    """Highest threshold below which at most `max_missed` of the needed takeovers fall (we may only hold back takeovers that are not
    needed): returns (threshold, share of needless takeovers held back)."""
    need_p = np.sort(p[y])
    k = int(np.floor(max_missed * len(need_p)))
    thr = need_p[k] if len(need_p) else 1.0            # takeover allowed iff p >= thr
    held = float(np.mean(p[~y] < thr)) if (~y).any() else 0.0
    return float(thr), held


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sim.train_intent_net import export_trees
    tr, te = load("decisions_100.npz"), load("decisions_400.npz")
    seeds = np.unique(tr["seed"])
    rng = np.random.default_rng(0)
    val_seeds = set(rng.choice(seeds, size=max(1, len(seeds) // 5), replace=False).tolist())
    is_val = np.array([s in val_seeds for s in tr["seed"]])
    Xtr, ytr, Xva, yva = tr["X"][~is_val], tr["y"][~is_val], tr["X"][is_val], tr["y"][is_val]
    print(f"train {len(ytr)} takeovers ({ytr.mean() * 100:.1f} % needed), validation {len(yva)}, held-out test {len(te['y'])} ({te['y'].mean() * 100:.1f} % needed)")
    res = {"n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(te["y"])), "needed_rate_test": float(te["y"].mean())}
    res["baseline_p_v3"] = metrics(te["y"], te["p_v3"])
    print("baseline (v3 crash risk as the score):", res["baseline_p_v3"])
    lr = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    lr.fit((Xtr - mu) / sd, ytr)
    res["logistic"] = metrics(te["y"], lr.predict_proba((te["X"] - mu) / sd)[:, 1])
    print("logistic regression:", res["logistic"])
    best = None
    for depth, lr_, it in ((3, 0.08, 200), (4, 0.06, 300), (5, 0.05, 300)):
        gbm = HistGradientBoostingClassifier(max_depth=depth, learning_rate=lr_, max_iter=it, early_stopping=True,
                                             validation_fraction=0.2, random_state=0, class_weight="balanced")
        gbm.fit(Xtr, ytr)
        p_va = gbm.predict_proba(Xva)[:, 1]
        m_va = metrics(yva, p_va)
        print(f"  trees depth {depth}: {gbm.n_iter_} iterations, validation AP {m_va['ap']:.3f} AUC {m_va['auc']:.3f}")
        if best is None or m_va["ap"] > best[0]:
            best = (m_va["ap"], gbm, depth)
    gbm = best[1]
    p_te = gbm.predict_proba(te["X"])[:, 1]
    res["trees"] = dict(metrics(te["y"], p_te), depth=best[2], iterations=int(gbm.n_iter_))
    print("gradient-boosted trees (held-out):", res["trees"])
    # temperature on the validation set so that the output is a calibrated P(needed)
    from sim.train_intent_torch import fit_temperature
    raw_va = gbm.decision_function(Xva)
    T = fit_temperature(raw_va, yva.astype(np.float32))
    p_cal = 1 / (1 + np.exp(-gbm.decision_function(te["X"]) / T))
    for name, p in (("v3 risk", te["p_v3"]), ("needed-model", p_cal)):
        thr, held = operating_point(te["y"], np.nan_to_num(p, nan=1.0), max_missed=0.0)
        thr10, held10 = operating_point(te["y"], np.nan_to_num(p, nan=1.0), max_missed=0.10)
        res[f"hold_back_{name}"] = {"thr_no_miss": thr, "needless_held_back_no_miss": held, "thr_miss10": thr10, "needless_held_back_miss10": held10}
        print(f"{name:13s}: holds back {100 * held:5.1f} % of the needless takeovers without missing a needed one, "
              f"{100 * held10:5.1f} % missing at most 10 % of the needed ones")
    ex = export_trees(gbm)
    ex.update(kind="trees3", temperature=float(T), note="P(this takeover is needed); trained on Monte Carlo counterfactual labels")
    json.dump(ex, open(os.path.join(HERE, "..", "pi", "needless_v1.json"), "w"))
    json.dump(res, open(os.path.join(HERE, "..", "models", "needless_report.json"), "w"), indent=1)
    print("wrote pi/needless_v1.json")


if __name__ == "__main__":
    main()
