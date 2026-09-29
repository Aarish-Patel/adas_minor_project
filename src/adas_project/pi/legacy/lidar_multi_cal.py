"""Finer LiDAR yaw calibration using THREE placed objects: one dead ahead, one near +90 deg
(car frame) and one near -90 deg. Three independent bearing measurements cross-check each
other and average out placement error far better than a single object. The car does not move.

    python3 pi/lidar_multi_cal.py

Prints the yaw offset implied by each object alone, and the combined best-fit offset.
Nothing is written automatically - it prints what to put in tuning_real_car.json.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
from pi.lidar_steering_diag import Rig  # noqa: E402
from pi.test_gui import TestGui  # noqa: E402

N_SCANS = 50
SECTOR_HALF_WIDTH = 55.0   # deg: how far the mount could plausibly have drifted from the current guess


def nearest_face_bearing(points, raw_centre, half_width, dmin=0.22, dmax=1.8):
    """Among RAW points within [raw_centre - half_width, raw_centre + half_width], find the closest
    object and return (raw_bearing_of_its_face, distance) or (None, None)."""

    def wrap(a):
        return (a + 180) % 360 - 180
    pts = [(a, d) for a, d in points if dmin <= d <= dmax and abs(wrap(a - raw_centre)) <= half_width]
    if len(pts) < 5:
        return None, None
    dmin_hit = min(d for _, d in pts)
    face = [(a, d) for a, d in pts if d < dmin_hit + 0.06]
    if len(face) < 3:
        return None, None
    xy = np.array([(d * np.cos(np.radians(a)), d * np.sin(np.radians(a))) for a, d in face])
    c = xy.mean(0)
    return float(np.degrees(np.arctan2(c[1], c[0])) % 360), float(np.hypot(*c))


def circ_mean_deg(vals, weights=None):
    v = np.radians(np.array(vals))
    w = np.ones(len(vals)) if weights is None else np.array(weights)
    s, c = np.sum(w * np.sin(v)), np.sum(w * np.cos(v))
    return float(np.degrees(np.arctan2(s, c)) % 360)


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("Finer LiDAR calibration (3 objects)",
            "one object dead ahead, one near +90 deg, one near -90 deg; the car stays still", 0.0,
            activity="MEASURING")
    cur = D.MOUNT.yaw_offset_deg
    centres = {"front": cur, "side_a": (cur + 90) % 360, "side_b": (cur - 90) % 360}
    samples = {k: [] for k in centres}
    t0 = time.time()
    n = 0
    try:
        while n < N_SCANS and time.time() - t0 < 35:
            if rig.is_fresh():
                pts = rig.points()
                got_all = True
                for k, c in centres.items():
                    b, d = nearest_face_bearing(pts, c, SECTOR_HALF_WIDTH)
                    if b is not None:
                        samples[k].append((b, d))
                    else:
                        got_all = False
                n += 1
                gui.set(progress=n / N_SCANS, message=f"front {len(samples['front'])}, side A {len(samples['side_a'])}, "
                                                       f"side B {len(samples['side_b'])} valid scans of {n}")
                if not got_all:
                    gui.set(activity="one or more objects not seen this scan - keep them in place")
            time.sleep(0.12)

        have = {k: v for k, v in samples.items() if len(v) >= 15}
        if "front" not in have:
            gui.log("FAILED: never got a clean reading on the front object")
            gui.set("Finer LiDAR calibration - FAILED", "check the front object is closest in its sector", 1.0, activity="finished")
            return

        stats = {}
        for k, v in have.items():
            bearings = [b for b, d in v]
            stats[k] = {"bearing": circ_mean_deg(bearings), "sd": float(np.std(np.unwrap(np.radians(bearings)))),
                       "n": len(v), "dist": float(np.median([d for _, d in v]))}
            gui.log(f"{k}: raw bearing {stats[k]['bearing']:.2f} deg (sd {np.degrees(stats[k]['sd']):.2f}), "
                    f"{stats[k]['dist']:.2f} m, {stats[k]['n']} scans")

        # offset implied by each object alone, if it is where it's expected to be
        implied = {"front": stats["front"]["bearing"] % 360}
        if "side_a" in stats:
            implied["side_a_as_left"] = (stats["side_a"]["bearing"] - 90) % 360
            implied["side_a_as_right"] = (stats["side_a"]["bearing"] + 90) % 360
        if "side_b" in stats:
            implied["side_b_as_left"] = (stats["side_b"]["bearing"] - 90) % 360
            implied["side_b_as_right"] = (stats["side_b"]["bearing"] + 90) % 360

        for k, v in implied.items():
            gui.log(f"  implied offset from {k}: {v:.2f} deg")

        # best assignment: side_a=+90 & side_b=-90, or the other way round - whichever agrees best with the front reading
        best = None
        if "side_a" in stats and "side_b" in stats:
            for name, oa, ob in (("A=+90,B=-90", implied["side_a_as_left"], implied["side_b_as_right"]),
                                 ("A=-90,B=+90", implied["side_a_as_right"], implied["side_b_as_left"])):
                cand = [implied["front"], oa, ob]
                m = circ_mean_deg(cand)

                def wrap(a):
                    return (a + 180) % 360 - 180
                spread = max(abs(wrap(x - m)) for x in cand)
                if best is None or spread < best[1]:
                    best = (name, spread, m, cand)
            gui.log(f"best side assignment: {best[0]}  (max deviation between the 3 estimates: {best[1]:.2f} deg)")
            offset = best[2]
            n_used = 3
        else:
            offset = implied["front"]
            n_used = 1

        gui.log(f"COMBINED yaw offset: {cur:.1f} -> {offset:.2f} deg  (from {n_used} object(s))")
        gui.set("Finer LiDAR calibration - DONE", f"yaw offset {cur:.1f} -> {offset:.2f} deg", 1.0, activity="finished")
        print("RESULT", {"prev": cur, "offset": offset, "stats": stats, "implied": implied,
                         "assignment": best[0] if best else "front only"}, flush=True)
    finally:
        time.sleep(2)
        rig.close()


if __name__ == "__main__":
    main()
