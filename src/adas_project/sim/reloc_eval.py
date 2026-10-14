"""Relocalisation on the twin: map a room, then drop the car at random poses with no pose guess and see where it thinks it is (TODO P2-2).

    python -m sim.reloc_eval [trials]        # writes models/reloc.json

The room is mapped by a serpentine drive with the twin's LiDAR (720 beams, 8 mm noise) at the true poses (the mapping quality is
slam_eval's business). Trials then place the car at random free poses that were not on the mapping route, give the relocaliser one
scan and, separately, the two-scan `Relocaliser` (the car moves 12 cm between the scans). Worlds: an asymmetric lounge, the
cluttered room, the empty 6 x 4 m room (a symmetric case: no unique answer from one scan, and it must say so instead of guessing)
and a corridor. Reported per world: fixes, wrong fixes (> 15 cm or > 5 deg off), refusals, position / heading error, time per fix.
Also the end-to-end case: a restart with a saved map (`PoseGraphMap.save` / `load`) and a lift-and-drop through `SlamService`.
"""
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


class Twin:
    """The twin's LiDAR at any pose of a world: scans in the vehicle frame, ground truth free-space test."""

    def __init__(self, world_name, seed=0):
        from adas.config import load_tuning
        from pi.relay_assists import car_params
        from sim.hw_sim import SimLidar, VirtualCar
        from sim.hw_worlds import WORLDS
        self.world, start = WORLDS[world_name]()
        self.clear = 0.22 if world_name == "corridor" else 0.4       # metres of free space round the LiDAR for a placement
        tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
        self.p = car_params(tun.mount)
        car = VirtualCar(self.world, self.p, start, threaded=False)
        self.lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
        seg = self.world.segments()
        self.bbox = (seg[:, [0, 2]].min(), seg[:, [0, 2]].max(), seg[:, [1, 3]].min(), seg[:, [1, 3]].max())

    def scan(self, x, y, th):
        lx = self.p.lidar_x
        best, _ = self.lidar._raycast(x + lx * math.cos(th), y + lx * math.sin(th), th)
        r = best + self.lidar.rng.normal(0, 0.008, len(best))
        ok = np.isfinite(best) & (r >= 0.2) & (r < 12)
        c = self.lidar.ccw[ok]
        return np.column_stack([lx + r[ok] * np.cos(c), r[ok] * np.sin(c)])

    def free(self, x, y, th, clear=None):
        clear = self.clear if clear is None else clear
        lx = self.p.lidar_x
        best, _ = self.lidar._raycast(x + lx * math.cos(th), y + lx * math.sin(th), th)
        return bool(np.nanmin(best) >= clear + lx)


def build_map(tw, row=0.45, step=0.2, seed=0):
    """Serpentine coverage drive at the true poses -> PoseGraphMap."""
    from adas.submaps import PoseGraphMap
    x0, x1, y0, y1 = tw.bbox
    g = PoseGraphMap()
    rng = np.random.default_rng(seed)
    ys = np.arange(y0 + 0.4, y1 - 0.3, row)
    for k, y in enumerate(ys):
        xs = np.arange(x0 + 0.4, x1 - 0.3, step)
        xs = xs if k % 2 == 0 else xs[::-1]
        th = 0.0 if k % 2 == 0 else math.pi
        for x in xs:
            if tw.free(x, y, th):
                g.add_scan(np.array([x, y, th]) + rng.normal(0, [0.002, 0.002, 0.001]), tw.scan(x, y, th))
    return g


def random_pose(tw, rng):
    x0, x1, y0, y1 = tw.bbox
    for _ in range(5000):
        x, y, th = rng.uniform(x0 + 0.3, x1 - 0.3), rng.uniform(y0 + 0.3, y1 - 0.3), rng.uniform(-math.pi, math.pi)
        if tw.free(x, y, th):
            return x, y, th
    raise RuntimeError("no free pose in this world")


def angle_err(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def trials(world_name, n=30, seed=7, verbose=True):
    from adas.submaps import Relocaliser
    tw = Twin(world_name, seed=seed)
    g = build_map(tw)
    rng = np.random.default_rng(seed)
    one = {"fix": 0, "wrong": 0, "none": 0, "err_cm": [], "err_deg": [], "ms": []}
    two = {"fix": 0, "wrong": 0, "none": 0, "err_cm": [], "err_deg": []}
    for _ in range(n):
        x, y, th = random_pose(tw, rng)
        t0 = time.perf_counter()
        fix = g.relocalise(tw.scan(x, y, th))
        one["ms"].append((time.perf_counter() - t0) * 1e3)
        if fix is None:
            one["none"] += 1
        else:
            e, a = math.hypot(fix.pose[0] - x, fix.pose[1] - y), angle_err(fix.pose[2], th)
            if e > 0.15 or a > math.radians(5):
                one["wrong"] += 1
            else:
                one["fix"] += 1
            one["err_cm"].append(100 * e)
            one["err_deg"].append(math.degrees(a))
        # two scans, the car moves 12 cm ahead between them (a free stretch: else it stays put)
        rl = Relocaliser(g)
        nx, ny = x + 0.12 * math.cos(th), y + 0.12 * math.sin(th)
        if not tw.free(nx, ny, th, clear=tw.clear - 0.1):
            nx, ny = x, y
        fx = None
        for t, (px, py) in enumerate(((x, y), (nx, ny))):
            fx = rl.update(0.1 * t, _est(x, y, th, px, py), tw.scan(px, py, th))
        if fx is None:
            two["none"] += 1
        else:
            e, a = math.hypot(fx.pose[0] - nx, fx.pose[1] - ny), angle_err(fx.pose[2], th)
            if e > 0.15 or a > math.radians(5):
                two["wrong"] += 1
            else:
                two["fix"] += 1
            two["err_cm"].append(100 * e)
            two["err_deg"].append(math.degrees(a))

    def summ(d):
        out = {k: d[k] for k in ("fix", "wrong", "none")}
        if d["err_cm"]:
            out["err_cm_median"] = float(np.median(d["err_cm"]))
            out["err_cm_max"] = float(np.max(d["err_cm"]))
            out["err_deg_median"] = float(np.median(d["err_deg"]))
        if d.get("ms"):
            out["ms_median"], out["ms_max"] = float(np.median(d["ms"])), float(np.max(d["ms"]))
        return out
    res = {"world": world_name, "trials": n, "map_nodes": len(g.nodes), "map_submaps": len(g.submaps), "single_scan": summ(one),
           "two_scans": summ(two)}
    if verbose:
        s1, s2 = res["single_scan"], res["two_scans"]
        print(f"{world_name:9s} map {res['map_nodes']} nodes / {res['map_submaps']} submaps | 1 scan: fix {s1['fix']}, wrong {s1['wrong']}, "
              f"refused {s1['none']}, median {s1.get('err_cm_median', float('nan')):.1f} cm / {s1.get('err_deg_median', float('nan')):.1f} deg, "
              f"{s1['ms_median']:.0f} ms (max {s1['ms_max']:.0f}) | 2 scans: fix {s2['fix']}, wrong {s2['wrong']}, refused {s2['none']}")
    return res


def _est(x0, y0, th0, x, y):
    """Front-end pose of the car at (x, y) when it started at (x0, y0, th0) as (0, 0, 0): straight-line motion along its heading."""
    return np.array([math.hypot(x - x0, y - y0), 0.0, 0.0])


def kidnap_end_to_end(world_name="lounge", seed=3, verbose=True):
    """Map with the pose graph, save and reload it (a restart), then run a fresh SlamService in lost mode from a random pose."""
    import tempfile
    from adas.submaps import PoseGraphMap, SlamService
    tw = Twin(world_name, seed=seed)
    g = build_map(tw)
    path = os.path.join(tempfile.mkdtemp(), "map.npz")
    g.save(path)
    g2 = PoseGraphMap.load(path)
    rng = np.random.default_rng(seed)
    x, y, th = random_pose(tw, rng)
    svc = SlamService()
    svc.graph = g2
    svc.relocalise()
    t0 = time.perf_counter()
    got = None
    for k in range(6):                                            # the car drives 5 cm per scan from where it was put down
        px, py = x + 0.05 * k * math.cos(th), y + 0.05 * k * math.sin(th)
        svc.submit(np.array([0.05 * k, 0.0, 0.0]), tw.scan(px, py, th))
        end = time.time() + 20
        while time.time() < end and svc.lost and svc.scans_seen <= k:
            time.sleep(0.01)
        if not svc.lost:
            got = k
            break
    dt = time.perf_counter() - t0
    time.sleep(0.05)
    px, py = x + 0.05 * (got or 0) * math.cos(th), y + 0.05 * (got or 0) * math.sin(th)
    est = np.array([0.05 * (got or 0), 0.0, 0.0])
    c = svc.corrected(est)
    err = math.hypot(c[0] - px, c[1] - py)
    svc.stop()
    res = {"world": world_name, "relocalised": got is not None, "scans_needed": None if got is None else got + 1,
           "err_cm": 100 * err, "heading_err_deg": math.degrees(angle_err(c[2], th)), "wall_s": dt}
    if verbose:
        print(f"end to end ({world_name}, saved map reloaded): {res}")
    return res


def main(n=30):
    out = {"worlds": []}
    for w in ("lounge", "room", "open", "corridor", "doorway"):
        out["worlds"].append(trials(w, n))
    out["end_to_end"] = kidnap_end_to_end()
    json.dump(out, open(os.path.join(HERE, "..", "models", "reloc.json"), "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
