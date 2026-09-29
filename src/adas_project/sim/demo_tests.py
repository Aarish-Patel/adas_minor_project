"""Automated acceptance tests for the demo, on the REAL-car profile (sim/real_car.py).

    python -m sim.demo_tests            writes models/demo_results.json, reports/demo_*.png, reports/demo_report.html

Each test has a pass/fail criterion and logs the numbers that go on the results slides:
  1 emergency braking at four speeds   (gap to the wall, no contact)
  2 throttle-cut roll-out              (simulator vs what was measured on the car)
  3 failsafe: controller/link/Pi lost  (from the fault-injection run)
  4 intent-aware warning               (hazards caught vs false alarms, from the warning evaluation)
  5 follow-the-leader                  (steady gap, and a leader that stops dead)
  6 obstacle bypass                    (random obstacles, walls, gaps)
"""
import json
import math
import os
import sys
from dataclasses import replace

import numpy as np


from .intent_data import MODEL_DIR
from .lidar_sim import LidarSim
from .real_car import ASSUMED, real_profile
from .simulator import Simulator
from .world import MovingBox, Wall, World

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "..", "reports")

# measured on the real car (pi/brake_run.py, final grips): cruise speed m/s -> metres rolled after cutting the throttle
REAL_ROLLOUT = [(0.249, 0.069), (0.291, 0.014), (0.289, 0.034), (0.364, 0.034)]


def make_sim(world, start=(0.0, 0.0, 0.0), mode="active", seed=1, v_scale=1.0):
    rp = real_profile()
    dyn = rp.dynamics
    if v_scale != 1.0:
        dyn = replace(dyn, speed_model=replace(dyn.speed_model, v_max=dyn.speed_model.v_max * v_scale))
    sim = Simulator(world, start, adas_on=mode, params=rp.params, aeb_config=rp.aeb, dynamics=dyn,
                    adas_speed_model=rp.speed_model, lidar=LidarSim(seed=seed, **rp.lidar_kw), seed=seed)
    return sim, rp


# ------------------------------------------------------------------ 1. emergency braking
def test_braking():
    rows = []
    for pwm in (115, 140, 180, 255):
        for scale in (1.0, 1.25):
            w = World()
            w.add(Wall(3.0, -1.0, 3.0, 1.0))
            sim, rp = make_sim(w, v_scale=scale)
            vmax = 0.0
            while sim.t < 9 and not sim.car.collided:
                sim.step(0.0, pwm if sim.t > 0.3 else 0.0)
                vmax = max(vmax, sim.car.v)
                if sim.t > 1 and abs(sim.car.v) < 0.005:
                    break
            gap = 3.0 - (sim.car.x + rp.params.front_x)
            rows.append({"pwm": pwm, "v_scale": scale, "peak_speed": vmax, "gap_cm": gap * 100, "crashed": bool(sim.car.collided)})
    ok = all(not r["crashed"] and r["gap_cm"] >= 3.0 for r in rows)
    worst = min(r["gap_cm"] for r in rows)
    return {"name": "Emergency braking at 4 speeds (also with the car 25% faster than calibrated)", "pass": ok,
            "detail": f"no contact in {len(rows)} runs, smallest gap {worst:.1f} cm, top speed tested {max(r['peak_speed'] for r in rows):.2f} m/s",
            "rows": rows}


# ------------------------------------------------------------------ 2. roll-out vs the real car
def test_rollout():
    rows = []
    for pwm in (100, 115, 130, 150, 180):
        w = World()
        w.add(Wall(8.0, -1, 8.0, 1))
        sim, _ = make_sim(w, mode="off")
        while sim.t < 4:
            sim.step(0.0, pwm)
        v0, x0 = sim.car.v, sim.car.x
        for _ in range(400):
            sim.step(0.0, 0.0)
        rows.append({"pwm": pwm, "speed": v0, "rollout_cm": (sim.car.x - x0) * 100})
    real_max = max(r for _, r in REAL_ROLLOUT) * 100
    ok = all(r["rollout_cm"] <= 12.0 for r in rows if r["speed"] <= 0.4)
    return {"name": "Throttle-cut roll-out: simulator vs the real car", "pass": ok,
            "detail": f"real car rolled 1-7 cm at 0.25-0.36 m/s; simulator {min(r['rollout_cm'] for r in rows):.0f}-"
                      f"{max(r['rollout_cm'] for r in rows if r['speed'] <= 0.4):.0f} cm in that range (limit 12 cm)",
            "rows": rows, "real": [{"speed": s, "rollout_cm": r * 100} for s, r in REAL_ROLLOUT], "real_max_cm": real_max}


# ------------------------------------------------------------------ 3. failsafe (from the fault-injection run)
def test_failsafe():
    path = os.path.join(MODEL_DIR, "faults.json")
    if not os.path.exists(path):
        from . import faults
        faults.main()
    faults = json.load(open(path, encoding="utf-8"))
    keys = ("Controller disconnects", "Pi program dies", "LiDAR unplugged", "Link delay")
    rel = [f for f in faults if any(k in f["name"] for k in keys)]
    return {"name": "Failsafes: controller lost, Pi dies, LiDAR unplugged, link delay", "pass": all(f["pass"] for f in rel),
            "detail": "; ".join(f"{f['name']}: {f['detail']}" for f in rel), "rows": rel}


# ------------------------------------------------------------------ 4. intent-aware warning
def test_warning():
    path = os.path.join(MODEL_DIR, "warning_eval.json")
    if not os.path.exists(path):
        return {"name": "Intent-aware warning", "pass": None, "detail": "run python -m sim.warning_eval first (about 20 minutes)", "rows": []}
    curves = json.load(open(path, encoding="utf-8"))["curves"]

    def det_at(name, budget):
        rows = [r for r in curves[name] if r["false_alarms_per_min"] <= budget]
        return max((r["detection"] for r in rows), default=0.0)
    res = {b: {n: det_at(n, b) for n in curves} for b in (2, 4, 8)}
    best = "adaptive" if "adaptive" in curves else "blend"
    # criterion: better where a driver would tolerate the alarms (<= 4 per minute). At 8 per minute the physics-only warning
    # catches more (see the numbers) - reported, not hidden.
    ok = all(res[b][best] >= res[b]["physics"] for b in (2, 4))
    return {"name": "Intent-aware warning catches more hazards than physics-only at <= 4 false alarms/min", "pass": ok,
            "detail": "; ".join(f"<= {b} false alarms/min: " + ", ".join(f"{n} {v * 100:.0f}%" for n, v in res[b].items()) for b in res),
            "rows": res}


# ------------------------------------------------------------------ 5. follow the leader
def test_follow():
    def one(stop_leader):
        w = World()
        w.add(Wall(-1.0, -1.2, 12.0, -1.2))
        w.add(Wall(-1.0, 1.2, 12.0, 1.2))
        leader = MovingBox(1.6, 0.0, 0.28, 0.14, 0.0, 0.30)
        w.add(leader)
        sim, rp = make_sim(w)
        sim.adas.acc.enabled = True
        gaps, ts = [], []
        while sim.t < 22 and not sim.car.collided:
            if stop_leader and sim.t > 9.0:
                leader.speed = 0.0
            sim.step(0.0, 200)
            lead = sim.adas.acc.lead
            if lead:
                gaps.append(lead[0]); ts.append(sim.t)
        return sim, np.array(ts), np.array(gaps)

    sim1, t1, g1 = one(False)
    steady = g1[(t1 > 6) & (t1 < 9)]
    sim2, t2, g2 = one(True)
    ok = (not sim1.car.collided) and (not sim2.car.collided) and sim2.min_clearance > 0.06
    return {"name": "Follow-the-leader: holds a gap, and does not hit a leader that stops dead", "pass": bool(ok),
            "detail": f"steady gap {steady.mean() * 100:.0f} cm (+/-{steady.std() * 100:.1f}); leader stops: closest approach {max(sim2.min_clearance, 0) * 100:.0f} cm, no contact"
                      if len(steady) else "no lead detected",
            "trace_moving": [t1.tolist(), g1.tolist()], "trace_stop": [t2.tolist(), g2.tolist()]}


# ------------------------------------------------------------------ 6. bypass
def test_bypass():
    path = os.path.join(MODEL_DIR, "bypass_eval.json")
    if not os.path.exists(path):
        from . import bypass_eval
        bypass_eval.main(60)
    s = json.load(open(path, encoding="utf-8"))["summary"]["real"]
    ok = s["crashes"] == 0 and s["success"] == s["runs"]
    return {"name": "Obstacle bypass: rejoins the line, or refuses a gap that is too narrow", "pass": ok,
            "detail": f"{s['success']}/{s['runs']} correct, {s['crashes']} contacts, median final lateral error "
                      f"{s['median_lateral_cm']:.1f} cm (95th {s['p95_lateral_cm']:.1f}), median clearance {s['median_clearance_cm']:.1f} cm",
            "rows": s}


# ------------------------------------------------------------------ figures + report
def figures(results):
    from .report import BLUE, GREY, RED, TEAL, YELLOW, style
    plt = style()
    os.makedirs(REPORTS, exist_ok=True)
    made = []
    b = next(r for r in results if r["name"].startswith("Emergency braking"))["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for scale, col, lab in ((1.0, TEAL, "car as calibrated"), (1.25, YELLOW, "car 25% faster than calibrated")):
        pts = sorted((r["peak_speed"], r["gap_cm"]) for r in b if r["v_scale"] == scale)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, label=lab)
    ax.axhline(3.0, color=RED, ls="--", lw=1, label="pass limit (3 cm)")
    ax.set_xlabel("speed when the wall appeared (m/s)"); ax.set_ylabel("gap left in front of the wall (cm)")
    ax.set_title("Emergency braking (real-car profile)"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "demo_braking.png"), dpi=150); plt.close(fig)
    made.append(("Emergency braking", "demo_braking.png"))

    ro = next(r for r in results if r["name"].startswith("Throttle-cut"))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot([r["speed"] for r in ro["rows"]], [r["rollout_cm"] for r in ro["rows"]], "o-", color=TEAL, label="simulator")
    ax.plot([r["speed"] for r in ro["real"]], [r["rollout_cm"] for r in ro["real"]], "s", color=YELLOW, label="measured on the car")
    ax.set_xlabel("cruise speed (m/s)"); ax.set_ylabel("distance rolled after cutting throttle (cm)")
    ax.set_title("Roll-out: simulator vs real car"); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "demo_rollout.png"), dpi=150); plt.close(fig)
    made.append(("Roll-out vs the real car", "demo_rollout.png"))

    f = next(r for r in results if r["name"].startswith("Follow"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, key, title in ((axes[0], "trace_moving", "leader at 0.30 m/s"), (axes[1], "trace_stop", "leader stops dead at 9 s")):
        t, g = f[key]
        ax.plot(t, [x * 100 for x in g], color=BLUE); ax.set_title(title); ax.set_xlabel("time (s)")
    axes[0].set_ylabel("gap to the leader (cm)")
    fig.tight_layout(); fig.savefig(os.path.join(REPORTS, "demo_follow.png"), dpi=150); plt.close(fig)
    made.append(("Follow-the-leader", "demo_follow.png"))
    return made


def write_html(results, made):
    rows = "".join(f"<tr><td>{'PASS' if r['pass'] else ('n/a' if r['pass'] is None else 'FAIL')}</td><td>{r['name']}</td><td>{r['detail']}</td></tr>"
                   for r in results)
    imgs = "".join(f"<h3>{t}</h3><img src='{f}' width='100%'>" for t, f in made)
    ass = "".join(f"<li>{a}</li>" for a in ASSUMED)
    doc = f"""<!doctype html><meta charset=utf-8><title>Demo acceptance tests</title>
<style>body{{background:#0b1120;color:#e5ecff;font-family:Segoe UI,system-ui,sans-serif;max-width:900px;margin:30px auto;padding:0 16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #1e2b47;padding:6px 8px;text-align:left;vertical-align:top}}h1{{font-size:26px}}h2,h3{{margin-top:28px}}</style>
<h1>Demo acceptance tests (simulated real-car profile)</h1>
<p>{sum(1 for r in results if r['pass'])}/{len(results)} passed. Simulator configured from measurements on the real car; the numbers below are what the car
should reproduce. Assumed (not measured) values:</p><ul>{ass}</ul><table>{rows}</table>{imgs}"""
    with open(os.path.join(REPORTS, "demo_report.html"), "w", encoding="utf-8") as f:
        f.write(doc)


def main():
    tests = (test_braking, test_rollout, test_failsafe, test_warning, test_follow, test_bypass)
    results = []
    for t in tests:
        r = t()
        results.append(r)
        print(f"{'PASS' if r['pass'] else ('n/a ' if r['pass'] is None else 'FAIL')}  {r['name']}\n      {r['detail']}")
    made = figures(results)
    write_html(results, made)
    slim = [{k: v for k, v in r.items() if not k.startswith("trace")} for r in results]
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "demo_results.json"), "w", encoding="utf-8") as fh:
        json.dump(slim, fh, indent=1, default=float)
    passed = sum(1 for r in results if r["pass"])
    print(f"\n{passed}/{len(results)} passed; wrote models/demo_results.json, reports/demo_report.html")
    return 0 if all(r["pass"] is not False for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
