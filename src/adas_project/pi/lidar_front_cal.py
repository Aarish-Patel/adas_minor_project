"""LiDAR front calibration: put ONE object dead-centre in front of the car (about 0.3-1.2 m away).
The car does not move. The bearing of the object's nearest face is measured over many scans and
the LiDAR yaw offset is corrected so that bearing becomes 0 deg (straight ahead)."""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/pi/rc_car")
import pi.lidar_steering_diag as D  # noqa: E402
from pi.lidar_steering_diag import Rig  # noqa: E402
from pi.test_gui import TestGui, hold, save_result  # noqa: E402

N_SCANS = 40


def object_bearing(points):
    """Bearing (deg) of the nearest object in the front half-plane: centroid of the points within
    6 cm of the closest return (its front face)."""
    pts = [(a, d) for a, d in points if abs(a) < 75 and 0.22 < d < 1.6]
    if len(pts) < 5:
        return None, None
    dmin = min(d for _, d in pts)
    face = [(a, d) for a, d in pts if d < dmin + 0.06]
    if len(face) < 3:
        return None, None
    xy = np.array([(d * np.cos(np.radians(a)), d * np.sin(np.radians(a))) for a, d in face])
    c = xy.mean(0)
    return float(np.degrees(np.arctan2(c[1], c[0]))), float(np.hypot(*c))


def main():
    rig = Rig()
    gui = TestGui(rig)
    gui.set("LiDAR front calibration", "keep one object dead-centre in front of the car; the car stays still", 0.0,
            activity="MEASURING - object bearing", servo=None)
    bearings, dists, last = [], [], None
    t0 = time.time()
    try:
        while len(bearings) < N_SCANS and time.time() - t0 < 25:
            if rig.is_fresh():
                pts = rig.points()
                b, d = object_bearing(pts)
                if b is None:
                    gui.set(activity="NO OBJECT FOUND in front (0.25 - 1.6 m)")
                else:
                    bearings.append(b); dists.append(d)
                    gui.set(progress=len(bearings) / N_SCANS,
                            message=f"object at {d:.2f} m, bearing {b:+.1f} deg  ({len(bearings)}/{N_SCANS})")
            time.sleep(0.12)
        if len(bearings) < 15:
            gui.log("FAILED: not enough valid scans - put a clear object 0.3-1.2 m in front and retry")
            gui.set("LiDAR front calibration - FAILED", activity="finished", progress=1.0)
        else:
            b = float(np.median(bearings)); sd = float(np.std(bearings)); cur = D.MOUNT.yaw_offset_deg
            new = (cur + b) % 360
            if abs(b) > 15 or sd > 3:
                gui.log(f"REJECTED: nearest object is {b:+.1f} deg off centre (sd {sd:.1f}) - not the centred object? nothing saved")
                gui.set("LiDAR front calibration - REJECTED", "check the object is centred and is the nearest thing in front", 1.0, activity="finished")
                hold(40)
                return
            save_result("lidar", {"yaw_offset_deg": new, "was": cur, "bearing_deg": b, "sd": sd, "distance_m": float(np.median(dists))})
            gui.log(f"object bearing {b:+.2f} deg (sd {sd:.2f}) at {np.median(dists):.2f} m")
            gui.log(f"yaw offset {cur:.1f} -> {new:.1f}   (press Apply in the panel)")
            gui.set("LiDAR front calibration - DONE", f"correction {b:+.2f} deg", 1.0, activity="finished")
        hold(40)
    finally:
        pass


if __name__ == "__main__":
    main()
