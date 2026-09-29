"""Auto-park on the relay code in the twin: detect the bay from the first LiDAR scan, then back into it.

    python -m sim.park_eval          # writes reports/park.png and models/park.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def bay_from_scan(world_name, start_pose=None):
    from adas.config import load_tuning
    from adas.park import find_bays, park_goal
    from pi.relay_assists import car_params
    from sim.hw_sim import SimLidar, VirtualCar
    from sim.hw_worlds import WORLDS
    world, start = WORLDS[world_name]()
    start = start_pose or start
    tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
    p = car_params(tun.mount)
    car = VirtualCar(world, p, start, threaded=False)
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=0)
    x, y, th = start
    best, _ = lidar._raycast(x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th), th)
    ok = np.isfinite(best) & (best >= 0.2) & (best < 12)
    pts = np.column_stack([best[ok] * np.cos(lidar.ccw[ok]) + p.lidar_x, best[ok] * np.sin(lidar.ccw[ok])])
    bays = find_bays(pts)
    return bays, (park_goal(bays[0]) if bays else None)


def slot_from_scan(world_name):
    """The first parallel slot the scan shows, and the rear-axle goal for it."""
    from adas.config import load_tuning
    from adas.park import find_parallel_slots, park_goal_parallel
    from pi.relay_assists import car_params
    from sim.hw_sim import SimLidar, VirtualCar
    from sim.hw_worlds import WORLDS
    world, start = WORLDS[world_name]()
    tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
    p = car_params(tun.mount)
    car = VirtualCar(world, p, start, threaded=False)
    lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=0)
    x, y, th = start
    best, _ = lidar._raycast(x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th), th)
    ok = np.isfinite(best) & (best >= 0.2) & (best < 12)
    pts = np.column_stack([best[ok] * np.cos(lidar.ccw[ok]) + p.lidar_x, best[ok] * np.sin(lidar.ccw[ok])])
    slots = find_parallel_slots(pts)
    return slots, (park_goal_parallel(slots[0]) if slots else None)


def parallel():
    from sim.autonav_eval import drive
    slots, goal = slot_from_scan("parallel")
    print("slots found:", slots)
    if goal is None:
        return None
    r = drive("parallel", goal[:2], t_max=80.0, heading_deg=goal[2], reverse_first=True)
    ok = r["ok"] and (r["heading_err_deg"] or 99) < 10.0
    print(f"parallel park: {'OK' if ok else 'FAIL'} {r['why']}  t {r.get('t')} s, {r.get('err', 0) * 100:.0f} cm from the slot centre, heading error "
          f"{r.get('heading_err_deg', 0):.1f} deg, min clearance {r.get('min_clear', 0) * 100:.0f} cm, crashed={r.get('crashed')}")
    return ok, r


def main():
    from sim.autonav_eval import drive
    bays, goal = bay_from_scan("parking")
    print("bays found:", bays)
    if goal is None:
        print("no bay")
        return
    r = drive("parking", goal[:2], t_max=60.0, heading_deg=goal[2], reverse_first=True)
    ok = r["ok"] and (r["heading_err_deg"] or 99) < 12.0
    print(f"park: {'OK' if ok else 'FAIL'} {r['why']}  t {r.get('t')} s, {r.get('err', 0) * 100:.0f} cm from the bay centre, heading error "
          f"{r.get('heading_err_deg', 0):.1f} deg, reversed {r.get('reversed_m', 0):.2f} m, min clearance {r.get('min_clear', 0) * 100:.0f} cm, "
          f"crashed={r.get('crashed')}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    for x1, y1, x2, y2 in r["world"].segments():
        ax.plot([x1, x2], [y1, y2], color="0.25", lw=1.5)
    if r.get("planned") is not None:
        ax.plot(r["planned"][:, 0], r["planned"][:, 1], "--", color="tab:blue", lw=1, label="plan")
    tr = np.array(r["trace"])
    ax.plot(tr[:, 0], tr[:, 1], color="tab:green" if ok else "tab:red", lw=2, label="driven")
    ax.plot(*r["goal"], "*", ms=14, color="tab:orange", label="bay centre")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.set_title(f"auto-park (reverse-in): {'parked' if ok else r['why']}")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "..", "reports", "park.png"), dpi=130)
    json.dump({"ok": bool(ok), "err_cm": r.get("err", 0) * 100, "heading_err_deg": r.get("heading_err_deg"),
               "t": r.get("t"), "reversed_m": r.get("reversed_m"), "min_clear_cm": r.get("min_clear", 0) * 100,
               "crashed": bool(r.get("crashed"))}, open(os.path.join(HERE, "..", "models", "park.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
    parallel()
