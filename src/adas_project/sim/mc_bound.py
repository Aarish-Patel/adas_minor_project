"""How many interventions can an intent-aware ADAS avoid at best?

Runs driver-only, brake-only, ADAS and ADAS + intent on the same drives and splits the drives by whether the DRIVER ALONE
reaches the goal without a crash. In drives the driver handles alone every intervention is avoidable (needless for safety and for
progress); in the other drives at least one is unavoidable. That bound is what a '90 % fewer interventions' target has to be
compared with.

    python -m sim.mc_bound [seeds]           # writes models/mc_bound.json
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

STYLES = ("lapsing", "aggressive", "late")
VARIANTS = ("off", "brake-only", "adas", "adas+intent", "adas+oracle")


def _job(a):
    seed, style, variant = a
    from sim import relay_mc
    r = relay_mc.run((seed, variant, style))
    kinds = {}
    for e in r["events"]:
        kinds[str(e[3]) + (":needed" if e[2] == "ok" else ":needless")] = kinds.get(str(e[3]) + (":needed" if e[2] == "ok" else ":needless"), 0) + 1
    tr = np.array(r["trace"])
    d = np.hypot(*(tr - np.array(r["goal"])).T)
    return {"seed": seed, "style": style, "variant": variant, "crashed": r["crashed"], "reached": r["reached"],
            "interventions": r["interventions"], "needless": r["false_positives"], "episodes": r["episodes"],
            "progress": r["progress"], "kinds": kinds,
            "retreat": float(np.clip(np.diff(d), 0, None).sum()), "burden": r["burden"], "min_clear": r["min_clear"]}


def summarise(rows):
    by = {(r["seed"], r["style"], r["variant"]): r for r in rows}
    keys = sorted({(r["seed"], r["style"]) for r in rows})
    alone_ok = [k for k in keys if by[k + ("off",)]["reached"] is not None and not by[k + ("off",)]["crashed"]]
    alone_bad = [k for k in keys if k not in alone_ok]
    out = {"drives": len(keys), "driver_alone_ok": len(alone_ok), "driver_alone_fails": len(alone_bad), "systems": {}}
    for v in VARIANTS:
        sel = [by[k + (v,)] for k in keys]
        out["systems"][v] = {
            "crashes": sum(r["crashed"] for r in sel), "goals": sum(r["reached"] is not None for r in sel),
            "interventions": sum(r["interventions"] for r in sel), "needless": sum(r["needless"] for r in sel),
            "episodes": sum(r["episodes"] for r in sel), "progress_assists": sum(r["progress"] for r in sel),
            "safety_interventions": sum(r["interventions"] - r["progress"] for r in sel),
            "interventions_in_driver_ok_drives": sum(by[k + (v,)]["interventions"] for k in alone_ok),
            "interventions_in_driver_fail_drives": sum(by[k + (v,)]["interventions"] for k in alone_bad),
            "goals_in_driver_fail_drives": sum(by[k + (v,)]["reached"] is not None for k in alone_bad),
            "retreat_m": round(sum(r["retreat"] for r in sel), 1),
            "median_time_s": float(np.median([r["reached"] for r in sel if r["reached"]] or [np.nan]))}
    return out


def paired(rows, a, b, key):
    """One-sided paired Wilcoxon: is system a's per-drive `key` greater than system b's?"""
    from scipy.stats import wilcoxon
    by = {(r["seed"], r["style"], r["variant"]): r for r in rows}
    keys = sorted({(r["seed"], r["style"]) for r in rows})
    x = np.array([by[k + (a,)][key] for k in keys], float)
    y = np.array([by[k + (b,)][key] for k in keys], float)
    p = float(wilcoxon(x, y, alternative="greater").pvalue) if np.any(x != y) else 1.0
    return {"a": a, "b": b, "key": key, "sum_a": float(x.sum()), "sum_b": float(y.sum()),
            "reduction_pct": float(100 * (1 - y.sum() / max(x.sum(), 1e-9))), "p": p}


def main(n_seeds=24, first=0):
    jobs = [(s, st, v) for s in range(first, first + n_seeds) for st in STYLES for v in VARIANTS]
    with ProcessPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(_job, jobs, chunksize=1))
    res = summarise(rows)
    res["paired"] = [paired(rows, "adas", "adas+intent", k) for k in ("interventions", "episodes", "needless", "retreat")] +                     [paired(rows, "adas", "adas+oracle", k) for k in ("interventions", "episodes")]
    print(json.dumps(res, indent=1))
    json.dump({"summary": res, "rows": rows}, open(os.path.join(HERE, "..", "models", "mc_bound.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 24, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
