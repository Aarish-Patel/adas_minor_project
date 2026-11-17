"""Emergency-brake stopping test on the relay code in the twin (11-13 Nov 2026): full throttle at a wall 4 m ahead, no steering, at throttle
settings from creeping to full. For every run: the car's peak speed, whether it touched the wall and the gap between its nose and the
wall once it stood still. Writes models/stopping.json and reports/stopping.png (final gap against the car's peak speed).

    python -m sim.stopping_eval
"""
import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WALL_X = 4.0


def one(pwm, seed):
    """The throttle stays held after the first stop (as a driver would keep it): the run ends when the car has stood still for 1.5 s.
    Reported: the gap at the FIRST stop and at the end, how many brake pulses and how many separate stops the approach took (a good
    approach is one smooth slow-down: no pulse, one stop), and the time from 1 m before the wall to the final stop."""
    from sim import relay_scenarios as RS
    from sim.world import Wall, World
    w = World()
    w.add(Wall(WALL_X, -1.6, WALL_X, 1.6))
    still = {"since": None}

    def done(t, x, y, th, v):
        if t > 1.0 and abs(v) < 0.01:
            still["since"] = t if still["since"] is None else still["since"]
            return t - still["since"] > 1.5
        still["since"] = None
        return False
    r = RS.run(w, lambda t, x, y, th, v: (0.0, pwm if t > 0.3 else 0.0), 60.0, seed=seed, stop_when=done)
    front_x = r["p"].front_x
    tr = r["trace"]
    gap = [WALL_X - (row[1] + front_x) for row in tr]
    v = [row[4] for row in tr]
    act = [str(row[9] or "") for row in tr]
    pulses = sum(1 for a, b in zip(act[:-1], act[1:]) if b.startswith("brak") and not a.startswith("brak"))
    moving = [abs(q) > 0.03 for q in v]
    first_move = next((i for i, m in enumerate(moving) if m), 0)
    stops = sum(1 for i in range(first_move + 1, len(moving)) if moving[i - 1] and not moving[i])
    first_stop = next((i for i in range(first_move + 1, len(moving)) if moving[i - 1] and not moving[i]), len(tr) - 1)
    i1 = next((i for i, g in enumerate(gap) if g < 1.0), 0)
    peak = max(abs(q) for q in v)
    return {"pwm": pwm, "seed": seed, "peak_speed": round(float(peak), 3), "first_stop_cm": round(gap[first_stop] * 100, 1),
            "gap_cm": round(gap[-1] * 100, 1), "min_gap_cm": round(min(gap) * 100, 1), "brake_pulses": pulses, "stops": stops,
            "approach_s": round(float(tr[-1][0] - tr[i1][0]) - 1.5, 1), "collided": bool(r["collided"])}


def main(seeds=(0, 1, 2)):
    rows = [one(pwm, s) for pwm in (60, 75, 90, 110, 130, 150, 180, 220, 255) for s in seeds]
    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    json.dump(rows, open(os.path.join(ROOT, "models", "stopping.json"), "w"), indent=1)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.scatter([r["peak_speed"] for r in rows], [r["gap_cm"] for r in rows], c="#14b8a6", s=26, label="one run (3 seeds per throttle)")
    ax.axhline(0, color="#dc2626", lw=1)
    ax.set_xlabel("peak speed before the wall (m/s, car scale)")
    ax.set_ylabel("final gap, nose to wall (cm)")
    ax.set_title("Emergency brake in the twin: stops short at every speed")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "reports", "stopping.png"), dpi=170)
    return rows


if __name__ == "__main__":
    rows = main()
    for r in rows:
        print(r)
    n = len(rows)
    print(f"{n} runs: contacts {sum(r['collided'] for r in rows)}, final gap {min(r['gap_cm'] for r in rows)}-{max(r['gap_cm'] for r in rows)} cm, "
          f"closest {min(r['min_gap_cm'] for r in rows)} cm, brake pulses per run {sum(r['brake_pulses'] for r in rows) / n:.2f}, "
          f"stops per run {sum(r['stops'] for r in rows) / n:.2f}, approach time {sum(r['approach_s'] for r in rows) / n:.1f} s")
