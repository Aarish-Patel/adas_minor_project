"""Lane keeping test: drive a curving road with the throttle held and the wheel left alone.

    python -m sim.lane_eval
"""

import json
import math
import os

from .intent_data import MODEL_DIR
from .library import lane_road
from .simulator import Simulator


def run(mode, pwm=125):
    world, start = lane_road()
    sim = Simulator(world, start, adas_on="active")
    sim.adas.lane.mode = mode
    worst, warned, ticks = 0.0, 0, 0
    while sim.t < 25 and not sim.car.collided and sim.car.x < 9.0:
        sim.step(0.0, pwm if sim.t > 0.3 else 0)
        centre = 0.30 * math.sin(0.55 * sim.car.x)
        worst = max(worst, abs(sim.car.y - centre))
        warned += sim.adas.lane.level >= 1
        ticks += 1
    return {"mode": mode, "reached_m": round(sim.car.x, 1), "worst_offset_cm": round(worst * 100, 1),
            "warning_share": round(warned / max(ticks, 1), 2), "crashed": bool(sim.car.collided)}


def main():
    rows = [run(m) for m in ("off", "warn", "assist")]
    for r in rows:
        print(r)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "lane_eval.json"), "w") as f:
        json.dump(rows, f, indent=1)
    return rows


if __name__ == "__main__":
    main()
