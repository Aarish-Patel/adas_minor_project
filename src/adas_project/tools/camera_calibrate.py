"""Calibrate the rear webcam (TODO P20): intrinsics and where it is mounted on the car.

    python tools/camera_calibrate.py intrinsics --index 0 [--cols 9 --rows 6 --square 0.024] [--frames 20]
        Hold a printed checkerboard in front of the camera at different angles and distances; a frame is kept when the
        pattern is found and differs from the last one. Result: the camera matrix and distortion coefficients
        (Zhang 2000, cv2.calibrateCamera) with the reprojection error.
    python tools/camera_calibrate.py extrinsics --index 0 --board-x -0.60 --board-y 0.0 [--cols 9 --rows 6 --square 0.024]
        Put the checkerboard FLAT ON THE FLOOR behind the car with its first INNER corner (one square in from the edge) at (board-x, board-y) in the
        vehicle frame (x forward from the rear axle, y left; behind = negative x); the 9 corners of a row run away from
        the car along x, the 6 rows across it.
        Result: the camera height z, pitch and (with the board straight) yaw - solvePnP on the pattern.
    Both write adas/camera_rear.json, which pi/rear_camera.py and the GUI read.

The same functions are unit-tested against the synthetic camera (tests/test_vision.py): known pose in, same pose out.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
CAM_JSON = os.path.join(ROOT, "adas", "camera_rear.json")


def find_board(gray, cols, rows):
    """Inner corners of a checkerboard, or None. Tries the sector-based detector first (robust to blur and strong
    foreshortening, which a floor board seen from a low camera has), then the classic one."""
    import cv2
    try:
        ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY)
        if ok:
            return corners.astype(np.float32)
    except cv2.error:
        pass
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))


def calibrate_intrinsics(gray_images, cols=9, rows=6, square=0.024):
    """-> (camera matrix, distortion, rms reprojection error in pixels, number of views used)."""
    import cv2
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    objp, imgp, size = [], [], None
    for g in gray_images:
        c = find_board(g, cols, rows)
        if c is None:
            continue
        objp.append(obj)
        imgp.append(c)
        size = g.shape[::-1]
    if len(objp) < 6:
        raise ValueError(f"only {len(objp)} usable views - need at least 6")
    rms, K, dist, _, _ = cv2.calibrateCamera(objp, imgp, size, None, None)
    return K, dist, float(rms), len(objp)


def extrinsics_from_board(gray, K, dist, board_xy, cols=9, rows=6, square=0.024):
    """Camera position and orientation in the vehicle frame from a checkerboard lying on the floor.
    board_xy: vehicle-frame (x, y) of the pattern's first inner corner; its columns run along the car's x axis (away
    from it), its rows along y. Returns {'x', 'y', 'z', 'pitch_deg', 'yaw_deg'} of the camera (rear camera: yaw about 180)."""
    import cv2
    c = find_board(gray, cols, rows)
    if c is None:
        return None
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square
    ok, rvec, tvec = cv2.solvePnP(obj, c, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    Rb, _ = cv2.Rodrigues(rvec)
    cam_in_board = -Rb.T @ tvec.reshape(3)                      # camera centre in board coordinates
    best = None
    # a chessboard is only defined up to a 180 deg turn (either end can be the 'first' corner): try both and keep the one
    # that puts the camera on the car (|x| small, above the floor) - the board's z axis points into the floor (solvePnP)
    span = np.array([(cols - 1) * square, (rows - 1) * square, 0.0])
    for flip in (False, True):
        T = np.diag([-1.0, -1.0, -1.0]) if flip else np.diag([1.0, 1.0, -1.0])
        origin = np.array([board_xy[0], board_xy[1], 0.0]) + (np.array([span[0], span[1], 0.0]) if flip else 0.0)
        pos = T @ cam_in_board + origin
        Rcam_veh = T @ Rb.T                                     # camera axes (columns) in the vehicle frame
        fwd = Rcam_veh[:, 2]
        if pos[2] <= 0:
            continue
        cost = abs(pos[0]) + abs(pos[1])
        if best is None or cost < best[0]:
            best = (cost, pos, fwd)
    if best is None:
        return None
    _, pos, fwd = best
    yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
    pitch = math.degrees(math.asin(-fwd[2] / np.linalg.norm(fwd)))
    return {"x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "pitch_deg": float(pitch), "yaw_deg": float(yaw)}


def write_json(K, dist, size, mount, path=CAM_JSON):
    d = {"width": int(size[0]), "height": int(size[1]), "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]),
         "cy": float(K[1, 2]), "dist": [float(v) for v in np.ravel(dist)], "hfov_deg": math.degrees(2 * math.atan(size[0] / 2 / K[0, 0]))}
    d.update({k: mount[k] for k in ("x", "y", "z", "pitch_deg", "yaw_deg") if mount and k in mount})
    json.dump(d, open(path, "w"), indent=1)
    return d


def load_camera(path=CAM_JSON):
    """The mounting + intrinsics as an adas.markers.Camera (rear-facing defaults when there is no calibration yet)."""
    from adas.markers import Camera
    try:
        d = json.load(open(path))
        return Camera(width=d["width"], height=d["height"], hfov_deg=d["hfov_deg"], x=d.get("x", -0.05), y=d.get("y", 0.0),
                      z=d.get("z", 0.10), pitch_deg=d.get("pitch_deg", 25.0), yaw_deg=d.get("yaw_deg", 180.0))
    except (OSError, ValueError, KeyError):
        return Camera(width=320, height=240, hfov_deg=70.0, x=-0.05, z=0.10, pitch_deg=25.0, yaw_deg=180.0)


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["intrinsics", "extrinsics"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--cols", type=int, default=9)
    ap.add_argument("--rows", type=int, default=6)
    ap.add_argument("--square", type=float, default=0.024)
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--board-x", type=float, default=-0.60)
    ap.add_argument("--board-y", type=float, default=0.0)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        sys.exit("camera not found")
    if a.mode == "intrinsics":
        imgs, last = [], None
        print("hold the checkerboard in view at varied angles; q to stop")
        while len(imgs) < a.frames:
            ok, f = cap.read()
            if not ok:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            c = find_board(g, a.cols, a.rows)
            if c is not None and (last is None or np.abs(c - last).mean() > 12):
                imgs.append(g)
                last = c
                print(f"  view {len(imgs)}/{a.frames}")
            cv2.drawChessboardCorners(f, (a.cols, a.rows), c, c is not None) if c is not None else None
            cv2.imshow("intrinsics", f)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        K, dist, rms, n = calibrate_intrinsics(imgs, a.cols, a.rows, a.square)
        print(f"{n} views, reprojection error {rms:.2f} px\nK =\n{K}\ndist = {dist.ravel()}")
        old = {}
        try:
            old = json.load(open(CAM_JSON))
        except (OSError, ValueError):
            pass
        write_json(K, dist, imgs[0].shape[::-1], old)
    else:
        d = json.load(open(CAM_JSON))
        K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]])
        for _ in range(10):
            cap.read()
        ok, f = cap.read()
        m = extrinsics_from_board(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), K, np.array(d["dist"]), (a.board_x, a.board_y),
                                  a.cols, a.rows, a.square)
        if m is None:
            sys.exit("checkerboard not found - lay it flat on the floor behind the car, well lit, fully in view")
        print("camera mounting:", {k: round(v, 3) for k, v in m.items()})
        d.update(m)
        json.dump(d, open(CAM_JSON, "w"), indent=1)


if __name__ == "__main__":
    main()
