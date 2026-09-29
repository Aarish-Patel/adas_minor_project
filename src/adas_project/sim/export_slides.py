"""Collect everything the results slides need into reports/slides/ (numbered figures + summary.md).

    python -m sim.export_slides            (run sim.demo_tests, sim.bypass_eval and sim.report first for fresh numbers)
"""
import json
import os
import shutil

import numpy as np

from .bypass_driver import BypassDriver
from .demo_tests import make_sim
from .intent_data import MODEL_DIR
from .world import Box, Wall, World

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")
OUT = os.path.join(REPORTS, "slides")

ORDER = [("demo_braking.png", "Emergency braking: gap left in front of the wall vs speed"),
         ("demo_rollout.png", "Roll-out after cutting the throttle: simulator vs the real car"),
         ("bypass_paths.png", "Obstacle bypass: paths around a block, real-car profile"),
         ("demo_follow.png", "Follow-the-leader: gap to the leader"),
         ("stopping.png", "Braking distance (simulator, default car)"),
         ("warning_curves.png", "Warnings: hazards caught vs false alarms, physics vs intent-aware"),
         ("intent_model.png", "Driver-intent model")]


def bypass_figure():
    from .report import BLUE, GREY, RED, TEAL, YELLOW, style
    plt = style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ends = []
    for seed, (cy, width, colour) in enumerate(((0.0, 0.20, TEAL), (0.12, 0.30, BLUE), (-0.10, 0.14, YELLOW))):
        w = World()
        for a, b, c, d in ((-1.0, -1.6, 5.6, -1.6), (5.6, -1.6, 5.6, 1.6), (5.6, 1.6, -1.0, 1.6), (-1.0, 1.6, -1.0, -1.6)):
            w.add(Wall(a, b, c, d))
        w.add(Box(1.8, cy, 0.20, width, 0.0, 0.16))
        sim, rp = make_sim(w, seed=seed)
        drv = BypassDriver(rp.params, pwm=115)
        xs, ys = [], []
        while sim.t < 40 and not sim.car.collided and not drv.done:
            sim.step(*drv.command(sim))
            xs.append(sim.car.x); ys.append(sim.car.y)
        ends.append((sim.min_clearance, ys[-1]))
        ax.plot(xs, np.array(ys) * 100, color=colour, label=f"obstacle {width * 100:.0f} cm wide, {cy * 100:+.0f} cm off line")
        ax.add_patch(plt.Rectangle((1.7, (cy - width / 2) * 100), 0.2, width * 100, color=colour, alpha=0.35))
    ax.axhline(0, color=GREY, ls="--", lw=1)
    ax.set_xlabel("distance along the line (m)"); ax.set_ylabel("sideways offset (cm)")
    ax.set_title("Obstacle bypass: around it, then back onto the line"); ax.legend(frameon=False, fontsize=8)
    ax.set_aspect("auto")
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "bypass_paths.png"), dpi=150); plt.close(fig)
    return ends


def load(name):
    p = os.path.join(MODEL_DIR, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    bypass_figure()
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):
        os.remove(os.path.join(OUT, f))
    lines = ["# Results summary (all numbers from simulation configured with the real car's measurements)\n"]
    n = 0
    for fname, caption in ORDER:
        src = os.path.join(REPORTS, fname)
        if os.path.exists(src):
            n += 1
            shutil.copy(src, os.path.join(OUT, f"{n:02d}_{fname}"))
            lines.append(f"- Slide figure {n:02d}: **{caption}**  (`{n:02d}_{fname}`)")
    demo = load("demo_results.json") or []
    lines.append("\n## Acceptance tests\n")
    for r in demo:
        lines.append(f"- **{'PASS' if r['pass'] else ('n/a' if r['pass'] is None else 'FAIL')}** {r['name']}: {r['detail']}")
    res = json.load(open(os.path.join(HERE, "..", "web", "data", "results.json"), encoding="utf-8")) \
        if os.path.exists(os.path.join(HERE, "..", "web", "data", "results.json")) else {}
    lines.append("\n## Other evaluations\n")
    if "parking" in res:
        p = res["parking"]
        lines.append(f"- Auto-parking (simulator, camera markers): {p['success']}/{p['runs']} correct, median lateral error {p['median_lateral_cm']:.1f} cm")
    if "faults" in res:
        lines.append(f"- Fault injection: {sum(x['pass'] for x in res['faults'])}/{len(res['faults'])} passed")
    if "warning" in res:
        lines.append(f"- Warnings: {res['warning']['summary']}")
    lines.append("\n## Not yet measured on the real car\n")
    lines.append("- Stopping distance above 0.36 m/s, turn gain at the new servo centre, full-lock radius, and every simulated result "
                 "above until it is repeated on the car. See docs/STATUS.md.")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {n} figures and summary.md to reports/slides/")


if __name__ == "__main__":
    main()
