"""Repeatability protocol (TODO N5), after the Euro NCAP AEB test practice of repeating every test: each relay scenario
(sim/relay_scenarios.py) is run N times, every time on a different randomised twin - the car (top speed, dead-band,
motor lag, braking, command delay, steering gain, servo centre) and the LiDAR (noise, dropout, yaw error) drawn around
the identified twin (sim/twin_intent_data.randomise) - and the pass rate is reported per scenario.

    python -m sim.repeat_scenarios [runs=20] [level=1.0]  -> models/repeatability.json, reports/repeatability.png

Why: the real car vibrates and gives inconsistent results (user, 28 Sep). A system that passes once on the nominal twin
proves little; one that passes 95 % of the time across the whole randomisation range is the claim worth making.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)


def one(args):
    idx, seed, level = args
    from sim import relay_scenarios as R
    R.RANDOMISE["level"], R.RANDOMISE["seed"], R.RANDOMISE["log"] = level, seed, []
    name, fn = R.SCENARIOS[idx]
    try:
        ok, detail = fn()
    except Exception as e:                      # a scenario that crashes on some draw counts as a failure
        ok, detail = False, f"error: {e}"
    log = R.RANDOMISE["log"]
    assisted = [r for r in log if r[0]] or log  # the runs with the system under test switched on
    safe = all(not c and m > 0.0 for _, c, m in assisted)
    clear = min((m for _, _, m in assisted), default=9.0)
    return idx, seed, bool(ok), detail, bool(safe), float(clear)


def main(runs=20, level=1.0):
    from sim import relay_scenarios as R
    jobs = [(i, 1000 + k, level) for i in range(len(R.SCENARIOS)) for k in range(runs)]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(one, jobs, chunksize=1))
    out = []
    for i, (name, _) in enumerate(R.SCENARIOS):
        rows = [r for r in res if r[0] == i]
        passed = sum(r[2] for r in rows)
        safe = sum(r[4] for r in rows)
        fails = [r[3] for r in rows if not r[2]][:2]
        out.append({"name": name, "runs": len(rows), "passed": passed, "rate": passed / len(rows), "safe": safe,
                    "safe_rate": safe / len(rows), "min_clearance_cm": round(100 * min(r[5] for r in rows), 1),
                    "example_failures": fails})
        print(f"safe {100 * safe / len(rows):4.0f} %   spec {100 * passed / len(rows):4.0f} %   min clearance "
              f"{100 * min(r[5] for r in rows):5.1f} cm   {name}", flush=True)
    tot = sum(o["passed"] for o in out) / sum(o["runs"] for o in out)
    tot_safe = sum(o["safe"] for o in out) / sum(o["runs"] for o in out)
    print(f"overall: safe {100 * tot_safe:.1f} %, meets the nominal spec {100 * tot:.1f} % of {sum(o['runs'] for o in out)} runs")
    json.dump({"level": level, "runs_per_scenario": runs, "overall": tot, "overall_safe": tot_safe, "scenarios": out},
              open(os.path.join(ROOT, "models", "repeatability.json"), "w"), indent=1)
    figure(out, level, runs)
    return out


def figure(out, level, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from gui import theme
    C = theme.C
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(out) + 1.6), facecolor=C["bg"])
    ax.set_facecolor(C["bg1"])
    y = np.arange(len(out))
    safe = [100 * o["safe_rate"] for o in out]
    spec = [100 * o["rate"] for o in out]
    ax.barh(y, [100] * len(out), color=C["surface"], height=0.62)
    ax.barh(y, safe, color=[C["ok"] if r >= 95 else C["warn"] if r >= 80 else C["bad"] for r in safe], height=0.62)
    ax.scatter(spec, y, marker="|", s=260, color=C["accent"], zorder=3, label="meets the nominal-car spec")
    for yi, r in zip(y, safe):
        ax.text(min(r + 1.5, 97), yi, f"{r:.0f} % safe", va="center", color=C["text"], fontsize=9)
    ax.legend(loc="lower right", facecolor=C["bg1"], edgecolor=C["hair"], labelcolor=C["text2"], fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([o["name"] for o in out], color=C["text2"], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.tick_params(colors=C["dim"])
    for sp in ax.spines.values():
        sp.set_color(C["hair"])
    ax.set_title(f"Safety over {runs} randomised twins per scenario (level {level:.1f}): bars = no contact with the assists on, "
                 f"ticks = meets the nominal spec", color=C["text"], fontsize=11, loc="left")
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    fig.savefig(os.path.join(ROOT, "reports", "repeatability.png"), dpi=130, facecolor=C["bg"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20, float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
