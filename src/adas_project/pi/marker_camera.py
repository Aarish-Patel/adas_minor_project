"""Marker camera service (TODO AN2): captures the parking webcam (front or rear, pi/camera.json), finds the bay's ArUco marker with OpenCV
and serves the marker poses and an annotated MJPEG stream. It runs in its own process - never in the relay's control loop - and the relay
reads it through pi/marker_client.py.

    python3 pi/marker_camera.py [--config pi/camera.json] [--port 8091] [--sim] [--bench N]

    http://<pi>:8091/stream   multipart MJPEG, the marker outlined with its id and distance
    http://<pi>:8091/state    JSON: {ok, t, fps, pipeline_ms, facing, marker_id, markers: [{id, x, y, z, facing, dist, bearing, size_px, t}]}
                              (vehicle frame: x ahead of the rear axle, y left, `facing` = the direction the marker's normal points, rad)

Detection: cv2.aruco, dictionary DICT_4X4_50, sub-pixel corner refinement; the pose of each detected marker of the configured id from its four
corners (adas.markers.pose_from_corners, solvePnP / IPPE_SQUARE) with the camera model of pi/camera.json (adas/camera_<facing>.json after a
calibration). The same state keys as pi/rear_camera.py (ok, fps, pipeline_ms) so the dashboard's camera drawer reads either service. --sim
renders a synthetic camera instead (the car driving up to a wall marker and back) to try the chain without a webcam.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

from adas import camera_config                                      # noqa: E402
from adas.markers import aruco_detector, pose_from_corners              # noqa: E402
from pi.rear_camera import STATE, serve                             # noqa: E402  (the same small HTTP server and state box)


class SyntheticSource:
    """A car driving up to a wall marker and back at 0.25 m/s (adas/vision/sim_camera.py) - front or rear facing."""

    def __init__(self, cam, cfg, fps):
        from adas.vision.sim_camera import SimCamera
        self.sim, self.fps, self.i, self.cam = SimCamera(cam, (cam.width, cam.height), seed=4), fps, 0, cam
        front = cfg["facing"] == "front"
        self.sign = 1.0 if front else -1.0
        self.marker = (int(cfg["marker_id"]), self.sign * 1.9, 0.0, cam.z, math.pi if front else 0.0, float(cfg["marker_size_m"]))

    def read(self):
        time.sleep(1.0 / self.fps)
        t = self.i / self.fps
        self.i += 1
        d = 0.25 * (t % 10.0) if (t % 20.0) < 10.0 else 0.25 * (20.0 - t % 20.0)          # up to 2.5 m and back
        return True, self.sim.render((self.sign * d * 0.8, 0.0, 0.0), markers=[self.marker])


class MarkerPipeline:
    def __init__(self, cam, marker_id, size_m, facing):
        import cv2
        self.cv2, self.cam, self.id, self.size, self.facing = cv2, cam, int(marker_id), float(size_m), facing
        self.detect = aruco_detector("DICT_4X4_50", subpixel=True)           # (OpenCV 4.6 on the Pi, 4.7+ on the laptop: adas.markers)
        self.fps, self.t_last = 0.0, None

    def process(self, frame, t):
        cv2 = self.cv2
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids = self.detect(gray)
        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        markers = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(vis, corners, ids)
            for c, i in zip(corners, ids.ravel()):
                if int(i) != self.id:
                    continue
                o = pose_from_corners(c.reshape(4, 2), int(i), self.size, self.cam, t)
                if o is None:
                    continue
                markers.append({"id": o.id, "x": round(o.x, 4), "y": round(o.y, 4), "z": round(o.z, 4), "facing": round(o.facing, 4),
                                "dist": round(o.dist, 3), "bearing": round(o.bearing, 4), "size_px": round(o.size_px, 1), "t": t})
                x, y = int(c[0][:, 0].min()), int(c[0][:, 1].min())
                cv2.putText(vis, f"id {o.id}  {o.dist:.2f} m  {math.degrees(o.bearing):+.0f} deg", (x, max(14, y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 220, 120), 1, cv2.LINE_AA)
        if self.t_last is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(t - self.t_last, 1e-3)
        self.t_last = t
        ms = (time.perf_counter() - t0) * 1000
        label = f"{self.facing} camera: " + (f"marker {self.id} seen" if markers else f"marker {self.id} not seen")
        cv2.putText(vis, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 120) if markers else (70, 80, 240), 1, cv2.LINE_AA)
        state = {"ok": True, "t": t, "fps": round(self.fps, 1), "pipeline_ms": round(ms, 1), "facing": self.facing, "marker_id": self.id,
                 "markers": markers}
        return vis, state


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="camera config (default pi/camera.json)")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="process this many frames, print timing, exit")
    ap.add_argument("--fps", type=float, default=None, help="frames processed per second (default: pi/camera.json process_fps, else 10)")
    a = ap.parse_args()
    cv2.setNumThreads(1)                              # one core at most: the relay's control loop and the LiDAR come first on this Pi
    try:
        os.nice(10)
    except (OSError, AttributeError):
        pass
    cfg = camera_config.load(a.config)
    process_fps = float(a.fps or cfg.get("process_fps", 10.0))
    rotate_deg = int(cfg.get("rotate_deg", 0)) % 360
    if rotate_deg not in (0, 180):
        sys.exit(f"camera.json: rotate_deg must be 0 or 180, not {rotate_deg}")
    cam = camera_config.camera_model(cfg)
    if a.sim:
        src = SyntheticSource(cam, cfg, cfg["fps"])
    else:
        src = cv2.VideoCapture(int(cfg["index"]), cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
        src.set(cv2.CAP_PROP_FRAME_WIDTH, cam.width)
        src.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.height)
        src.set(cv2.CAP_PROP_FPS, cfg["fps"])
        src.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        src.set(cv2.CAP_PROP_BUFFERSIZE, 1)           # the newest frame, not a queue of old ones
        if not src.isOpened():
            sys.exit(f"camera {cfg['index']} not found")
    pipe = MarkerPipeline(cam, cfg["marker_id"], cfg["marker_size_m"], cfg["facing"])
    if not a.bench:
        serve(a.port)
        print(f"marker camera service on :{a.port} (/stream, /state)  {cfg['facing']} camera, marker id {cfg['marker_id']} "
              f"({cfg['marker_size_m'] * 100:.0f} cm)  source: {'synthetic' if a.sim else 'webcam ' + str(cfg['index'])}", flush=True)
    times, n, t_next = [], 0, 0.0
    while True:
        if not a.sim and not a.bench:                 # frames between two processed ones are dropped without decoding (grab, no retrieve)
            if time.time() < t_next:
                src.grab()
                continue
            t_next = time.time() + 1.0 / process_fps
        ok, frame = src.read()
        if not ok:
            with STATE["lock"]:
                STATE["json"] = {"ok": False, "why": "no frame from the camera"}
            time.sleep(0.05)
            continue
        if frame.shape[1] != cam.width:
            frame = cv2.resize(frame, (cam.width, cam.height))
        if rotate_deg == 180:                         # a camera mounted upside down: turned upright here, so the detection, the pose and the stream agree
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        vis, state = pipe.process(frame, time.time())
        times.append(state["pipeline_ms"])
        n += 1
        if a.bench:
            if n >= a.bench:
                t_ = np.array(times[5:])
                print(f"pipeline {t_.mean():.1f} ms mean, {np.percentile(t_, 95):.1f} ms 95th, {t_.max():.1f} ms max over {len(t_)} frames")
                return
            continue
        with STATE["lock"]:
            watched = STATE.get("watchers", 0) > 0
        jpeg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes() if watched else None      # encode only for a viewer
        with STATE["lock"]:
            STATE["json"] = state
            if jpeg is not None:
                STATE["jpeg"] = jpeg


if __name__ == "__main__":
    main()
