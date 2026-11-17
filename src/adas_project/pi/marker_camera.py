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
from adas.markers import aruco_detector, marker_like, pose_from_corners, square_enough              # noqa: E402
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


def cv2_area(q):
    import cv2
    return cv2.contourArea(np.asarray(q, np.float32).reshape(4, 2))


class MarkerPipeline:
    TOLERANT_EVERY = 2                    # frames between two blur-tolerant searches
    MIN_PX, DECODE_X2_PX = 12.0, 30.0     # smallest marker-like quad reported (side, px); one at least this big is worth a 2x decode

    @staticmethod
    def _px(q):
        return float(np.sqrt(abs(cv2_area(q))))

    def __init__(self, cam, marker_id, size_m, facing):
        import cv2
        self.cv2, self.cam, self.id, self.size, self.facing = cv2, cam, int(marker_id), float(size_m), facing
        self.detect = aruco_detector("DICT_4X4_50", subpixel=True)           # (OpenCV 4.6 on the Pi, 4.7+ on the laptop: adas.markers)
        self.detect_blurred = aruco_detector("DICT_4X4_50", subpixel=False, tolerant=True, with_rejected=True)   # the fixed-focus webcam blurs a marker 0.5 m away
        self.fps, self.t_last, self.n = 0.0, None, 0
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    def process(self, frame, t):
        cv2 = self.cv2
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids = self.detect(gray)
        target_found = ids is not None and self.id in ids.ravel()
        cands = []
        self.n += 1
        if self.n % self.TOLERANT_EVERY == 0:                    # the blur-tolerant pass (costly on the Pi: every few frames)
            # contrast-limited equalisation + unsharp mask, at 2x: the fixed-focus webcam washes the marker's black out to grey and smears its
            # bits (11 Nov: a marker 0.9 m ahead decoded only this way). Corners come back in the 1x frame.
            g2 = self.clahe.apply(gray)
            g2 = cv2.addWeighted(g2, 2.0, cv2.GaussianBlur(g2, (0, 0), 3), -1.0, 0)
            g2 = cv2.resize(g2, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            c1, i1, rej = self.detect_blurred(g2)
            c1 = [np.asarray(c, np.float32) / 2.0 for c in c1]
            rej = [np.asarray(q, np.float32) / 2.0 for q in rej]
            quads = [(np.asarray(c, np.float32).reshape(4, 2), int(i)) for c, i in zip(c1, [] if i1 is None else i1.ravel())]
            # (error correction at 1.0 decodes wrong ids from clutter: only THIS id counts as decoded, another is a candidate only if it
            # also has the marker structure)
            quads = [(q, i if i == self.id else -1) for q, i in quads if square_enough(q) and (i == self.id or marker_like(gray, q))]
            quads += [(np.asarray(q, np.float32).reshape(4, 2), -1) for q in rej
                      if self._px(q) >= self.MIN_PX and square_enough(q) and marker_like(gray, q)]
            for q, i in quads:                                   # "there is a marker over there": any id, or an undecoded marker-like quad
                o = pose_from_corners(q, self.id if i < 0 else i, self.size, self.cam, t)
                if o is not None:
                    cands.append({"id": i, "x": round(o.x, 3), "y": round(o.y, 3), "dist": round(o.dist, 2), "bearing": round(o.bearing, 4),
                                  "size_px": round(o.size_px, 1), "t": t})
            if not target_found:
                extra = [q for q, i in quads if i == self.id]
                if extra:
                    corners = list(corners) + [q.reshape(1, 4, 2) for q in extra]
                    ids = np.vstack(([ids] if ids is not None else []) + [np.full((len(extra), 1), self.id, np.int32)])
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
                 "markers": markers, "candidates": cands}
        return vis, state


FOCUS_FILE = os.path.expanduser("~/marker_focus.txt")      # "auto" or a number: read about once a second, so the focus can be tuned live over ssh


def apply_focus(src, value):
    """Webcam focus through V4L2: None leaves it alone, "auto" turns autofocus on, a number turns it off and sets that focus (0-255 on the
    Lenovo/UVC cameras; a small number = far, a large one = near). A fixed-focus camera ignores both. Returns what the camera reports."""
    import cv2
    if value is None:
        return None
    if str(value).lower() == "auto":
        src.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    else:
        src.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        src.set(cv2.CAP_PROP_FOCUS, float(value))
    return src.get(cv2.CAP_PROP_AUTOFOCUS), src.get(cv2.CAP_PROP_FOCUS)


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
        got = apply_focus(src, cfg.get("focus"))
        print(f"focus setting {cfg.get('focus')!r} -> camera reports (autofocus, focus) {got}", flush=True)
    pipe = MarkerPipeline(cam, cfg["marker_id"], cfg["marker_size_m"], cfg["facing"])
    if not a.bench:
        serve(a.port)
        print(f"marker camera service on :{a.port} (/stream, /state)  {cfg['facing']} camera, marker id {cfg['marker_id']} "
              f"({cfg['marker_size_m'] * 100:.0f} cm)  source: {'synthetic' if a.sim else 'webcam ' + str(cfg['index'])}", flush=True)
    times, n, t_next, t_focus, last_focus = [], 0, 0.0, 0.0, str(cfg.get("focus"))
    while True:
        if not a.sim and not a.bench:                 # frames between two processed ones are dropped without decoding (grab, no retrieve)
            if time.time() < t_next:
                src.grab()
                continue
            t_next = time.time() + 1.0 / process_fps
        if not a.sim and not a.bench and time.time() > t_focus:        # live focus tuning from ~/marker_focus.txt
            t_focus = time.time() + 1.0
            try:
                want = open(FOCUS_FILE).read().strip()
            except OSError:
                want = ""
            if want and want != last_focus:
                last_focus = want
                print(f"focus file: {want} -> camera reports {apply_focus(src, want if want.lower() == 'auto' else float(want))}", flush=True)
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
