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
    seed, style, variant, model, trust = args
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
            "overridden_s": r["burden"]["needless_s"], "min_clear": r["min_clear"]}


def score(rows):
    return {"runs": len(rows), "crashes": sum(r["crashed"] for r in rows),
            "goals": sum(r["reached"] is not None for r in rows),
            "needless": sum(r["needless"] for r in rows), "takeovers": sum(r["takeovers"] for r in rows),
            "overridden_s": round(sum(r["overridden_s"] for r in rows), 1),
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


def main():
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
