"""Domain-randomisation ablation (RESEARCH.md section 7, step 6): does training on randomised twins make the
intent model robust to a car it has never seen?

    python -m sim.intent_ablation   (after: python -m sim.twin_intent_data ablation)
                                    -> models/intent_ablation.json, reports/intent_ablation.png

Two models with the same recipe (gradient-boosted trees on the v3 features, same size and seed):
  nominal     trained on the nominal twin only (data/intent/train_nominal.npz, randomisation level 0)
  randomised  trained on randomised twins (data/intent/train.npz, level 1)
Both are tested on the nominal twin (test_nominal) and on cars and sensors drawn from 1.6x WIDER ranges than any
training set saw (test_wide) - the stand-in for the real car, whose exact parameters are unknown.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sim.train_intent_torch import flat_all, load, metrics, run_level
    sets = {k: load(k) for k in ("train_nominal", "train", "test_nominal", "test_wide")}
    feats = {}
    for k, d in sets.items():
        feats[k] = flat_all(d, np.arange(len(d["y"])))
    res = {}
    for name, tr in (("nominal", "train_nominal"), ("randomised", "train")):
        X, _ = feats[tr]
        y = sets[tr]["y"]
        n = min(len(y), len(sets["train_nominal"]["y"]))         # same amount of data for both
        idx = np.random.default_rng(0).choice(len(y), n, replace=False)
        gbm = HistGradientBoostingClassifier(max_iter=400, max_depth=6, learning_rate=0.08, early_stopping=True,
                                             validation_fraction=0.1, random_state=0).fit(X[idx], y[idx])
        res[name] = {"train_ticks": int(n)}
        for te in ("test_nominal", "test_wide"):
            Xt, floor = feats[te]
            p = np.maximum(gbm.predict_proba(Xt)[:, 1], floor)
            d = sets[te]
            res[name][te] = {"ticks": metrics(p, d["y"].astype(float)), "runs": run_level(d, np.arange(len(p)), p)}
        print(name, {te: round(res[name][te]["ticks"]["ap"], 3) for te in ("test_nominal", "test_wide")}, flush=True)
    json.dump(res, open(os.path.join(ROOT, "models", "intent_ablation.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, (key, title, sub) in zip(axes, (("ap", "average precision", "ticks"), ("missed", "crash drives never warned", "runs"),
                                            ("safe_runs_with_false_alarm", "safe drives with a false alarm", "runs"))):
        for i, te in enumerate(("test_nominal", "test_wide")):
            for j, (name, col) in enumerate((("nominal", "#9ca3af"), ("randomised", "#2563eb"))):
                v = res[name][te][sub][key] or 0
                ax.bar(i + (j - 0.5) * 0.38, v, 0.36, color=col, label=f"trained on {name} twin" if i == 0 else None)
                ax.text(i + (j - 0.5) * 0.38, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["nominal test cars", "wider-range test cars\n(unseen, like the real car)"])
        ax.set_title(title)
    axes[0].legend(fontsize=8)
    fig.suptitle("Domain-randomisation ablation: the same model trained on one twin vs randomised twins")
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "intent_ablation.png"), dpi=110)


if __name__ == "__main__":
    main()
