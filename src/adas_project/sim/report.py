"""Collect every evaluation into the viewer's Results tab and into report figures.

    python -m sim.report              use the saved results (models/*.json), regenerate figures
    python -m sim.report --run        also (re)run the scenario suite, Monte Carlo, parking and sweeps

Writes  web/data/results.json  (Results tab)  and  reports/*.png + reports/index.html.
"""

import json
import math
import os
import sys

import numpy as np

from .intent_data import MODEL_DIR

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "web", "data")
REPORTS = os.path.join(ROOT, "reports")


def load(name):
    path = os.path.join(MODEL_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ------------------------------------------------------------------ data gathering
def scenario_rows():
    from .run_scenarios import fmt, real_crash, run_one, verdict
    from .scenarios import default_suite
    rows, passed, total = [], 0, 0
    for scn in default_suite():
        off, on = run_one(scn, False), run_one(scn, True)
        fast, slow = run_one(scn, True, 1.25), run_one(scn, True, 0.75)
        ok = all(verdict(scn, s) for s in (on, fast, slow))
        total += 1
        passed += ok
        rows.append({"name": scn.name, "off": "CRASH" if real_crash(off) else f"ok, gap {max(off.min_clearance, 0) * 100:.0f} cm",
                     "off_crash": bool(real_crash(off)),
                     "on": ("PASS" if ok else "FAIL") + f", gap {max(on.min_clearance, 0) * 100:.0f} cm",
                     "pass": bool(ok)})
    return {"rows": rows, "passed": passed, "total": total}


def monte_carlo_summary(runs=200):
    from concurrent.futures import ProcessPoolExecutor

    from .monte_carlo import _both
    bins = [(90, 130), (130, 170), (170, 210), (210, 256)]
    with ProcessPoolExecutor() as pool:
        res = list(pool.map(_both, [(1000 + i, None) for i in range(runs)], chunksize=2))
    out = []
    for lo, hi in bins:
        sel = [r for r in res if lo <= r[1] < hi]
        out.append({"label": f"{lo}-{hi - 1}", "n": len(sel), "off": int(sum(r[4] for r in sel)), "on": int(sum(r[5] for r in sel))})
    return {"runs": runs, "bins": out, "total_off": int(sum(r[4] for r in res)), "total_on": int(sum(r[5] for r in res))}


def stopping_curve():
    """Final gap to a wall vs throttle, ADAS on."""
    from .scenarios import head_on_wall
    from .simulator import Simulator
    pts = []
    for pwm in range(100, 256, 15):
        scn = head_on_wall(pwm)
        world, start = scn.build()
        sim = Simulator(world, start, adas_on=True)
        sim.run(scn.driver, 6.0)
        v_peak = max(abs(r[4]) for r in sim.log)
        pts.append({"pwm": pwm, "speed": round(v_peak, 2), "gap_cm": round(max(sim.min_clearance, 0) * 100, 1),
                    "crash": bool(sim.car.collided)})
    return pts


def gather(run):
    results = {}
    if run:
        from . import parking_eval, sweeps
        print("scenario suite ...")
        results["scenarios"] = scenario_rows()
        print("random stress test ...")
        results["monte_carlo"] = monte_carlo_summary(200)
        print("parking ...")
        parking_eval.main(60)
        print("sweeps ...")
        sweeps.main(40)
        print("lane keeping ...")
        from . import lane_eval
        lane_eval.main()
        print("fault injection ...")
        from . import faults
        faults.main()
    else:
        cached = load("results_cache.json")
        if cached:
            results.update(cached)
    results["stopping"] = stopping_curve()
    park = load("parking_eval.json")
    if park:
        results["parking"] = {k: v for k, v in park.items() if k != "runs_detail"}
        results["parking"]["detail"] = park["runs_detail"]
    sw = load("sweeps.json")
    if sw:
        results["sweeps"] = sw
    lane = load("lane_eval.json")
    if lane:
        results["lane"] = lane
    faults = load("faults.json")
    if faults:
        results["faults"] = faults
    byp = load("bypass_eval.json")
    if byp:
        results["bypass"] = byp["summary"]
    intent = load("intent_metrics.json")
    if intent:
        results["intent"] = intent
    warn = load("warning_eval.json")
    if warn:
        curves = warn["curves"]
        results["warning"] = {"curves": curves, "hazards": warn.get("hazards_test"), "summary": warning_summary(curves)}
    if run:
        with open(os.path.join(MODEL_DIR, "results_cache.json"), "w") as f:
            json.dump({k: results[k] for k in ("scenarios", "monte_carlo") if k in results}, f)
    return results


def warning_summary(curves):
    def det_at(name, budget):
        rows = [r for r in curves[name] if r["false_alarms_per_min"] <= budget]
        return max((r["detection"] for r in rows), default=None)
    parts = []
    for budget in (4, 8):
        vals = {n: det_at(n, budget) for n in curves}
        txt = ", ".join(f"{n} {'n/a' if v is None else f'{v * 100:.0f}%'}" for n, v in vals.items())
        parts.append(f"hazards caught at <= {budget} false alarms/min: {txt}.")
    return " ".join(parts)


# ------------------------------------------------------------------ figures
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "#0b1120", "axes.facecolor": "#111a2e", "savefig.facecolor": "#0b1120",
                         "axes.edgecolor": "#33415f", "axes.labelcolor": "#c7d2fe", "text.color": "#e5ecff",
                         "xtick.color": "#94a3b8", "ytick.color": "#94a3b8", "grid.color": "#1e2b47",
                         "axes.grid": True, "font.size": 10, "axes.titleweight": "bold",
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


TEAL, RED, YELLOW, BLUE, GREY = "#2dd4bf", "#ef4444", "#facc15", "#60a5fa", "#94a3b8"


def figures(r):
    plt = style()
    os.makedirs(REPORTS, exist_ok=True)
    made = []

    if "monte_carlo" in r:
        m = r["monte_carlo"]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        x = np.arange(len(m["bins"]))
        ax.bar(x - 0.2, [b["off"] for b in m["bins"]], 0.4, color=RED, label="ADAS off")
        ax.bar(x + 0.2, [b["on"] for b in m["bins"]], 0.4, color=TEAL, label="ADAS on")
        ax.set_xticks(x, [f"PWM {b['label']}\n(n={b['n']})" for b in m["bins"]])
        ax.set_ylabel("crashes"); ax.set_title(f"Random stress test: {m['runs']} drives, {m['total_off']} vs {m['total_on']} crashes")
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "crashes_by_throttle.png"), dpi=140); plt.close(fig)
        made.append(("Crashes by throttle level", "crashes_by_throttle.png"))

    if "warning" in r:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        names = {"physics": ("physics-only", GREY), "blend": ("intent blend", YELLOW), "adaptive": ("adaptive (learned)", TEAL)}
        for k, (label, col) in names.items():
            rows = sorted(r["warning"]["curves"][k], key=lambda q: q["false_alarms_per_min"])
            ax.plot([q["false_alarms_per_min"] for q in rows], [q["detection"] * 100 for q in rows], "-o", color=col,
                    lw=2, ms=3.5, label=label)
        ax.set_xlabel("false alarms per minute"); ax.set_ylabel("hazards warned about (%)")
        ax.set_title("Warning quality on unseen drives (up and left is better)"); ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "warning_curves.png"), dpi=140); plt.close(fig)
        made.append(("Warning quality: physics vs intent", "warning_curves.png"))

    if "sweeps" in r:
        sw = r["sweeps"]
        fig, axes = plt.subplots(1, len(sw), figsize=(3.0 * len(sw), 3.2), sharey=True)
        for ax, (k, d) in zip(np.atleast_1d(axes), sw.items()):
            xs = [row["value"] for row in d["rows"]]
            ax.plot(xs, [row["off"] for row in d["rows"]], "-o", color=RED, ms=3, label="off")
            ax.plot(xs, [row["on"] for row in d["rows"]], "-o", color=TEAL, ms=3, label="on")
            ax.set_title(d["label"], fontsize=8.5)
        np.atleast_1d(axes)[0].set_ylabel("crash rate (%)"); np.atleast_1d(axes)[0].legend(frameon=False)
        fig.suptitle("Robustness: what happens when the real car differs from what the ADAS assumes", y=1.03)
        fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "robustness_sweeps.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
        made.append(("Robustness sweeps", "robustness_sweeps.png"))

    if "parking" in r and r["parking"].get("detail"):
        d = [x for x in r["parking"]["detail"] if x["state"] == "done"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        axes[0].scatter([x["lateral_cm"] for x in d], [x["yaw_deg"] for x in d], c=[TEAL if x["ok"] else RED for x in d], s=22)
        axes[0].axhline(0, color="#33415f"); axes[0].axvline(0, color="#33415f")
        axes[0].set_xlabel("lateral offset (cm)"); axes[0].set_ylabel("yaw error (deg)"); axes[0].set_title("Final parking pose")
        axes[1].hist([x["time"] for x in d], bins=10, color=BLUE); axes[1].set_xlabel("time to park (s)"); axes[1].set_title("Time")
        p = r["parking"]
        fig.suptitle(f"Auto-parking from random starts: {p['success']}/{p['runs']} parked correctly, {p['crashes']} crashes", y=1.02)
        fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "parking.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
        made.append(("Auto-parking accuracy", "parking.png"))

    if "stopping" in r:
        s = r["stopping"]
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        ax.plot([q["speed"] for q in s], [q["gap_cm"] for q in s], "-o", color=TEAL, lw=2)
        ax.set_xlabel("speed when the wall appeared (m/s)"); ax.set_ylabel("final gap to the wall (cm)")
        ax.set_title("Automatic emergency braking: stops short at every speed")
        fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "stopping.png"), dpi=140); plt.close(fig)
        made.append(("Emergency braking", "stopping.png"))

    if "intent" in r:
        i = r["intent"]
        classes = ["straight", "left", "right", "brake"]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), gridspec_kw={"width_ratios": [1, 1, 1.15]})
        for ax, key, title in ((axes[0], "baseline", "Keep-doing-the-same baseline"), (axes[1], "learned", "Learned intent model")):
            cm = np.array(i[key]["confusion"], dtype=float)
            cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(4), classes, rotation=30); ax.set_yticks(range(4), classes)
            ax.set_title(f"{title}\nmacro F1 {i[key]['macro_f1']:.2f}", fontsize=9); ax.grid(False)
            for a in range(4):
                for b in range(4):
                    ax.text(b, a, f"{cm[a, b] * 100:.0f}", ha="center", va="center", color="white" if cm[a, b] < 0.6 else "black", fontsize=8)
        x = np.arange(4)
        axes[2].bar(x - 0.2, [i["baseline"]["per_class_f1"][c] for c in classes], 0.4, color=GREY, label="baseline")
        axes[2].bar(x + 0.2, [i["learned"]["per_class_f1"][c] for c in classes], 0.4, color=TEAL, label="learned")
        axes[2].set_xticks(x, classes); axes[2].set_ylabel("F1"); axes[2].legend(frameon=False); axes[2].set_title("Per-class F1", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "intent_model.png"), dpi=140); plt.close(fig)
        made.append(("Driver intent model", "intent_model.png"))
    return made


def html(r, made):
    rows = ""
    if "scenarios" in r:
        rows = "".join(f"<tr><td>{x['name']}</td><td>{x['off']}</td><td>{x['on']}</td></tr>" for x in r["scenarios"]["rows"])
    imgs = "".join(f"<h3>{t}</h3><img src='{f}' width='100%'>" for t, f in made)
    fl = ""
    if "faults" in r:
        ft = "".join(f"<tr><td>{'PASS' if x['pass'] else 'FAIL'}</td><td>{x['name']}</td><td>{x['detail']}</td></tr>" for x in r["faults"])
        fl = f"<h2>Fault injection: {sum(x['pass'] for x in r['faults'])}/{len(r['faults'])} passed</h2><table>{ft}</table>"
    scen = (f"<h2>Scenario suite: {r['scenarios']['passed']}/{r['scenarios']['total']} passed (also with the real car 25% faster or slower "
            f"than calibrated)</h2><table><tr><th>Scenario</th><th>ADAS off</th><th>ADAS on</th></tr>{rows}</table>") if rows else ""
    doc = f"""<!doctype html><meta charset=utf-8><title>RC-ADAS results</title>
<style>body{{background:#0b1120;color:#e5ecff;font-family:Segoe UI,system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #1e2b47;padding:6px 8px;text-align:left}}th{{color:#94a3b8}}
h1{{font-size:26px}}h2,h3{{margin-top:28px}}</style>
<h1>RC-ADAS simulation results</h1>
<p>All numbers come from the simulator with the same ADAS code that runs on the car. Real-car results will differ; use the
tuning panel and calibration procedures to close the gap.</p>{scen}{fl}{imgs}"""
    with open(os.path.join(REPORTS, "index.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def main():
    run = "--run" in sys.argv
    r = gather(run)
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "results.json"), "w") as f:
        json.dump(r, f)
    made = figures(r)
    html(r, made)
    print("wrote web/data/results.json,", len(made), "figures and reports/index.html")


if __name__ == "__main__":
    main()
