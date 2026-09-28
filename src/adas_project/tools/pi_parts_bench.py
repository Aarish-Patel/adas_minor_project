"""Cost on this machine of the pieces around the relay's control loop that the profiler does not cover: the
digital twin's own simulated LiDAR (only in simulation), the GUI's predicted-path overlay, the drive log, and
the LiDAR point conversion. python3 tools/pi_parts_bench.py"""
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def t(label, fn, n=30):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"{label:52s} {ms:7.2f} ms")
    return ms


def main():
    from adas.config import load_tuning
    from pi.path_gate import PathGate
    from pi.relay_assists import RelayAssists, car_params
    from sim.hw_sim import SimLidar, VirtualCar
    from sim.hw_worlds import build
    tun = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
    p = car_params(tun.mount)
    w, st = build("doorway")
    car = VirtualCar(w, p, st, threaded=False)
    lid = SimLidar(car, tun.mount.yaw_offset_deg)
    t("simulated LiDAR scan, 1360 beams (twin only)", lambda: lid._raycast(0.1, 0.0, 0.0))
    best, _ = lid._raycast(0.1, 0.0, 0.0)
    ok = np.isfinite(best)
    cw = (-np.degrees(lid.ccw)) % 360
    cw = np.where(cw > 180, cw - 360, cw)
    pts = [(round(float(a), 1), round(float(d), 3)) for a, d, o in zip(cw, best, ok) if o]
    print(f"   ({len(pts)} points per scan)")
    t("points -> vehicle frame (relay format)", lambda: RelayAssists.points_vehicle_frame(pts, p.lidar_x))
    g = PathGate(p, tun.speed_model)
    g.on_scan(RelayAssists.points_vehicle_frame(pts, p.lidar_x), 1)
    t("GUI predicted path overlay (every packet)", lambda: g.overlay(0.1, 1, 0.5, 0.5))
    import json
    payload = {"points": pts, "plan": g.overlay(0.1, 1, 0.5, 0.5)}
    t("GUI JSON of one scan (per browser poll)", lambda: json.dumps(payload))
    import gzip
    import io
    buf = gzip.GzipFile(fileobj=io.BytesIO(), mode="wb")
    rec = json.dumps({"t": time.time(), "scan": pts}).encode()
    t("drive log: one scan record, gzip", lambda: buf.write(rec + b"\n"))
    # the relay's LiDAR thread (pi/wifi_drive_safety.Clearance._loop), per scan
    import pi.wifi_drive_safety as R
    raw = [(15, (a + R.FRONT_OFFSET_DEG) % 360, d * 1000.0) for a, d in pts]

    def per_point_loop():
        best = None
        for _, angle, dist in raw:
            a = (angle - R.FRONT_OFFSET_DEG) % 360
            a = a if a <= 180 else a - 360
            d_m = dist / 1000.0
            b = d_m - R.body_overhang(a)
            if best is None or b < best:
                best = b
    t("LiDAR thread: per-point Python loop", per_point_loop)
    from adas.tracking import Tracker
    tr = Tracker()
    xy = RelayAssists.points_vehicle_frame(pts, p.lidar_x)
    k = [0]

    def track():
        k[0] += 1
        tr.update(xy, time.time(), 0.3, 0.0)
    t("LiDAR thread: moving-object tracker", track, n=20)
    from adas.tracking import moving_object_contact
    tracks = tr.update(xy, time.time(), 0.3, 0.0)
    t("LiDAR thread: moving-object contact", lambda: moving_object_contact(tracks, 0.0, 1, 0.3, R.VP, horizon=1.5))


if __name__ == "__main__":
    main()
