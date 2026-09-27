"""Paired statistics for the relay Monte Carlo (models/relay_mc.json): intent-aware ADAS vs the same ADAS without the
intent model, on the SAME rooms and drivers (paired by seed and driver style).

    python -m sim.mc_stats          -> table + one-sided Wilcoxon signed-rank tests, models/mc_stats.json
"""
import collections
import json
import os

from scipy.stats import wilcoxon

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# the same classes as sim/relay_mc.py's summary
KINDS = {"takeover": lambda k: k in ("evasive", "steer"),
         "brake": lambda k: str(k).startswith("gate:") and "limited" not in str(k),
         "limit": lambda k: k == "gate:limited"}


def needless(run, kind):
    return sum(1 for e in run["events"] if e[2] == "fp" and KINDS[kind](e[3]))


def main(a="adas", b="adas+intent"):
    runs = json.load(open(os.path.join(ROOT, "models", "relay_mc.json")))["runs"]
    by = collections.defaultdict(dict)
    for r in runs:
        by[(r["seed"], r["style"])][r["variant"]] = r
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    out = {"pairs": len(pairs)}
    print(f"{len(pairs)} paired drives: {a} vs {b}")
    for kind in ("takeover", "brake", "limit", "all"):
        f = (lambda r: sum(needless(r, k) for k in KINDS)) if kind == "all" else (lambda r, k=kind: needless(r, k))
        xa, xb = [f(p[0]) for p in pairs], [f(p[1]) for p in pairs]
        diff = [x - y for x, y in zip(xa, xb)]
        better, worse = sum(d > 0 for d in diff), sum(d < 0 for d in diff)
        p = wilcoxon(xa, xb, alternative="greater", zero_method="wilcox").pvalue if any(diff) else 1.0
        out[kind] = {a: sum(xa), b: sum(xb), "better": better, "worse": worse, "p_one_sided": p}
        print(f"  needless {kind:8s}: {sum(xa):3d} -> {sum(xb):3d}   {better} drives better, {worse} worse, "
              f"one-sided Wilcoxon p = {p:.4f}")
    # what the driver felt: seconds overridden needlessly, of which the wheel was taken, and throttle-seconds removed
    if all("burden" in p[0] and "burden" in p[1] for p in pairs):
        for key, label in (("needless_steer_s", "wheel taken needlessly (s)"),
                           ("needless_throttle_s", "throttle removed needlessly (throttle-s)"),
                           ("needless_s", "overridden needlessly (s)")):
            xa, xb = [p[0]["burden"][key] for p in pairs], [p[1]["burden"][key] for p in pairs]
            diff = [x - y for x, y in zip(xa, xb)]
            better, worse = sum(d > 1e-9 for d in diff), sum(d < -1e-9 for d in diff)
            pv = wilcoxon(xa, xb, alternative="greater").pvalue if any(abs(d) > 1e-9 for d in diff) else 1.0
            out[key] = {a: round(sum(xa), 1), b: round(sum(xb), 1), "better": better, "worse": worse, "p_one_sided": pv}
            print(f"  {label:42s}: {sum(xa):6.1f} -> {sum(xb):6.1f}   {better} better, {worse} worse, p = {pv:.4f}")
    # where the remaining needless interventions happened: free distance on the driver's own path / physical
    # stopping distance at that moment (< 1.3 = inside the brake's safety factor; the driver then swerved late)
    for variant in (a, b):
        ratios = [e[7] for r in runs if r["variant"] == variant for e in r["events"]
                  if e[2] == "fp" and len(e) > 7]
        inside = sum(1 for q in ratios if q is not None and q < 1.3)
        clear = sum(1 for q in ratios if q is None)
        print(f"  {variant}: needless interventions with the driver's path inside 1.3x its stopping distance: "
              f"{inside}/{len(ratios)} (path fully clear: {clear}); ratios {sorted(q for q in ratios if q is not None)}")
        out[f"envelope_{variant}"] = {"inside_1.3": inside, "total": len(ratios), "path_clear": clear}
    json.dump(out, open(os.path.join(ROOT, "models", "mc_stats.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])
