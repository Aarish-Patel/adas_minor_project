"""Optimise the intent-aware ADAS: sweep which intent model decides takeovers and how much it trusts the driver, on the
Monte Carlo (same drives for every configuration), then confirm the winner on seeds it was not tuned on.

    python -m sim.intent_optimise            # writes models/intent_optimise.json

Score per configuration (36 drives = 12 seeds x 3 driver styles): crashes, goals reached, needless interventions,
needless takeovers, overridden-needlessly seconds. Configuration = (model file, RC_TRUST threshold).
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

STYLES = ("lapsing", "aggressive", "late")


def _job(args):
    seed, style, variant, model, trust = args[:5]
    if len(args) > 5 and args[5]:
        os.environ["RC_STALL"] = args[5]
    else:
        os.environ.pop("RC_STALL", None)
    if model:
        os.environ["RC_INTENT_PATH"] = model
    else:
        os.environ.pop("RC_INTENT_PATH", None)
    from sim import relay_mc
    relay_mc.TRUST_THRESHOLD = trust
    r = relay_mc.run((seed, variant, style))
    return {"seed": seed, "style": style, "variant": variant, "crashed": r["crashed"], "reached": r["reached"],
            "interventions": r["interventions"], "needless": r["false_positives"],
            "takeovers": sum(1 for e in r["events"] if e[2] == "fp" and e[3] in ("evasive", "steer")),
            "overridden_s": r["burden"]["needless_s"], "min_clear": r["min_clear"], "retreat": _retreat(r)}


def _retreat(r):
    """Metres the car moved AWAY from the goal (sum of the increases of the distance to it) - a driver-visible measure of
    'going backwards' / wandering."""
    tr, g = np.array(r["trace"]), np.array(r["goal"])
    d = np.hypot(*(tr - g).T)
    return float(np.clip(np.diff(d), 0, None).sum())


def score(rows):
    return {"runs": len(rows), "crashes": sum(r["crashed"] for r in rows),
            "goals": sum(r["reached"] is not None for r in rows),
            "needless": sum(r["needless"] for r in rows), "takeovers": sum(r["takeovers"] for r in rows),
            "overridden_s": round(sum(r["overridden_s"] for r in rows), 1),
            "retreat_m": round(sum(r["retreat"] for r in rows), 1),
            "median_time_s": float(np.median([r["reached"] for r in rows if r["reached"]] or [np.nan]))}


def sweep(configs, seeds, ex):
    out = {}
    base = ex.map(_job, [(s, st, "adas", None, 0.5) for s in seeds for st in STYLES])
    out["adas"] = score(list(base))
    for name, (model, trust) in configs.items():
        jobs = [(s, st, "adas+intent", model, trust) for s in seeds for st in STYLES]
        out[name] = score(list(ex.map(_job, jobs)))
        print(f"{name:26s} {out[name]}", flush=True)
    return out


def stall_sweep():
    """Tune the stuck-car detector (seconds without progress, metres) with the v3 decider at trust 0.33."""
    v3 = os.path.join(HERE, "..", "pi", "intent_v3.json")
    res = {}
    with ProcessPoolExecutor() as ex:
        for st in ("8,1.0", "6,0.9", "5,0.8", "3.5,0.7"):
            rows = list(ex.map(_job, [(s, sty, "adas+intent", v3, 0.33, st) for s in range(12) for sty in STYLES]))
            res[st] = score(rows)
            print("stall", st, res[st], flush=True)
        base = list(ex.map(_job, [(s, sty, "adas", None, 0.5) for s in range(12) for sty in STYLES]))
        print("adas", score(base), flush=True)
        v2r = list(ex.map(_job, [(s, sty, "adas+intent", None, 0.5) for s in range(12) for sty in STYLES]))
        print("v2 0.5", score(v2r), flush=True)
    return res


def confirm():
    """Untuned seeds 12-43 (96 drives per configuration), paired against plain ADAS and against the current decider."""
    from scipy.stats import wilcoxon
    v3 = os.path.join(HERE, "..", "pi", "intent_v3.json")
    configs = {"v2 trust 0.5 (current)": (None, 0.5), "v3 trust 0.2": (v3, 0.2), "v3 trust 0.33": (v3, 0.33)}
    seeds = range(12, 44)
    rows = {}
    with ProcessPoolExecutor() as ex:
        rows["adas"] = list(ex.map(_job, [(s, st, "adas", None, 0.5) for s in seeds for st in STYLES]))
        for name, (model, trust) in configs.items():
            rows[name] = list(ex.map(_job, [(s, st, "adas+intent", model, trust) for s in seeds for st in STYLES]))
    res = {k: score(v) for k, v in rows.items()}
    for k, v in res.items():
        print(f"{k:24s} {v}", flush=True)

    def paired(a, b, key):
        x, y = np.array([r[key] for r in rows[a]], float), np.array([r[key] for r in rows[b]], float)
        d = x - y
        p = float(wilcoxon(x, y, alternative="greater").pvalue) if np.any(d != 0) else 1.0
        return {"mean_a": float(x.mean()), "mean_b": float(y.mean()), "p_a_greater": p}
    stats = {}
    for a, b in (("adas", "v2 trust 0.5 (current)"), ("adas", "v3 trust 0.33"), ("v2 trust 0.5 (current)", "v3 trust 0.33"),
                 ("v2 trust 0.5 (current)", "v3 trust 0.2")):
        for key in ("needless", "takeovers", "overridden_s", "retreat"):
            stats[f"{a} vs {b}: {key}"] = paired(a, b, key)
            print(f"{a} vs {b}: {key}", stats[f"{a} vs {b}: {key}"], flush=True)
    path = os.path.join(HERE, "..", "models", "intent_optimise.json")
    d = json.load(open(path)) if os.path.exists(path) else {}
    d["confirm"] = {"scores": res, "paired": stats}
    json.dump(d, open(path, "w"), indent=1)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "confirm":
        return confirm()
    if len(sys.argv) > 1 and sys.argv[1] == "stall":
        return stall_sweep()
    v2 = None                                                    # relay_mc default: models/intent_net.json
    v3 = os.path.join(HERE, "..", "pi", "intent_v3.json")
    configs = {f"v2 trust {t}": (v2, t) for t in (0.3, 0.5, 0.7)}
    configs.update({f"v3 trust {t}": (v3, t) for t in (0.05, 0.1, 0.2, 0.33, 0.5)})
    res = {}
    with ProcessPoolExecutor() as ex:
        print("tuning seeds 0-11", flush=True)
        res["tune"] = sweep(configs, range(12), ex)
        print("adas", res["tune"]["adas"], flush=True)
    json.dump(res, open(os.path.join(HERE, "..", "models", "intent_optimise.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
