"""Compare the takeover-needed model (TODO W5, pi/needless_v1.json) with v2 as the decider on the Monte Carlo.

    python -m sim.need_eval tune             # seeds 0-23: sweep RC_NEED_TAU (writes models/need_eval_tune.json)
    python -m sim.need_eval confirm <tau..>  # untouched seeds 24-47: ADAS, v2, need model at the given taus (models/need_eval_confirm.json)

Same drives for every configuration (72 = 24 seeds x 3 styles); paired one-sided Wilcoxon of each candidate against v2 on
needless takeovers, needless interventions, overridden-needlessly seconds, retreat and goals. Adopt only on a significant paired
improvement over v2 with 0 crashes and no fewer goals.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sim.intent_optimise import STYLES, _retreat  # noqa: E402

NEED = os.path.join(HERE, "..", "pi", "needless_v1.json")


def _job(a):
    seed, style, name = a
    for k in ("RC_NEED_PATH", "RC_NEED_TAU", "RC_INTENT_PATH", "RC_STALL", "RC_DEFER", "RC_EVADE_TTC", "RC_COMMIT",
              "RC_STUCKWAIT", "RC_RESPECT"):
        os.environ.pop(k, None)
    variant = "adas" if name == "adas" else "adas+intent"
    if name.startswith("need"):
        os.environ["RC_NEED_PATH"], os.environ["RC_NEED_TAU"] = NEED, name[4:]
    from sim import relay_mc
    relay_mc.TRUST_THRESHOLD = 0.5
    r = relay_mc.run((seed, variant, style))
    return {"seed": seed, "style": style, "name": name, "crashed": r["crashed"], "goal": r["reached"] is not None,
            "time": r["reached"], "interventions": r["interventions"], "needless": r["false_positives"],
            "takeovers": sum(1 for e in r["events"] if e[2] == "fp" and e[3] in ("evasive", "steer")),
            "episodes": r["episodes"], "overridden_s": r["burden"]["needless_s"], "retreat": _retreat(r)}


KEYS = ("interventions", "needless", "takeovers", "episodes", "overridden_s", "retreat")


def summarise(rows, names, ref="v2"):
    from scipy.stats import wilcoxon
    by = {(r["seed"], r["style"], r["name"]): r for r in rows}
    keys = sorted({(r["seed"], r["style"]) for r in rows})
    out = {}
    for n in names:
        sel = [by[k + (n,)] for k in keys]
        o = {"drives": len(sel), "crashes": sum(r["crashed"] for r in sel), "goals": sum(r["goal"] for r in sel),
             "median_time_s": float(np.median([r["time"] for r in sel if r["time"]] or [np.nan]))}
        for key in KEYS:
            o[key] = round(float(sum(r[key] for r in sel)), 1)
        if n != ref and ref in names:
            for key in KEYS:                       # is the candidate LOWER than v2?  (one-sided paired Wilcoxon)
                x = np.array([by[k + (ref,)][key] for k in keys], float)
                y = np.array([by[k + (n,)][key] for k in keys], float)
                o["p_" + key] = float(wilcoxon(x, y, alternative="greater").pvalue) if np.any(x != y) else 1.0
            o["goals_gained_vs_v2"] = o["goals"] - out[ref]["goals"] if ref in out else None
        out[n] = o
    return out


def run(names, seeds, out_path):
    jobs = [(s, st, n) for s in seeds for st in STYLES for n in names]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex, open(out_path + "l", "w") as log:
        for r in ex.map(_job, jobs, chunksize=1):
            rows.append(r)
            log.write(json.dumps(r) + "\n")
            log.flush()
    res = summarise(rows, names)
    print(json.dumps(res, indent=1))
    json.dump({"summary": res, "rows": rows}, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "tune":
        taus = sys.argv[2:] or ["0.1", "0.2", "0.35", "0.5"]
        run(["adas", "v2"] + ["need" + t for t in taus], range(24), os.path.join(HERE, "..", "models", "need_eval_tune.json"))
    else:
        run(["adas", "v2"] + ["need" + t for t in sys.argv[2:]], range(24, 48), os.path.join(HERE, "..", "models", "need_eval_confirm.json"))
