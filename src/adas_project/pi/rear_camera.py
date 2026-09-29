"""Rear-camera service (TODO P20): captures the back-facing webcam, runs the vision pipeline in its own process (never in
the relay's control loop), serves an annotated MJPEG stream and a JSON state for the GUI and the relay.

    python3 pi/rear_camera.py [--index 0] [--width 320 --height 240 --fps 15] [--port 8091] [--sim] [--yolo]

    http://<pi>:8091/stream   multipart MJPEG, annotated (blobs, looming time to contact, quality)
    http://<pi>:8091/state    JSON: odometry (speed, yaw rate, quality), objects [{foot x, y, ttc}], image quality, fps

Per frame: RearOdometry (speed / yaw from the floor), ImageQuality (blur, brightness, vibration), and every 5th frame
RearObjects (ground-compensated parallax blobs over a 5-frame baseline) with LoomingTracker (time to contact); YOLO on
every 3rd frame if --yolo and weights exist (adas/vision/detector.py). --sim renders a synthetic camera instead (a textured
floor with a box being reversed towards) so the whole chain can be tested and demonstrated without the webcam.
On the Pi 5 at 320x240 the pipeline is designed to stay below ~40 ms per frame (measure with --bench).
"""
import argparse
import collections
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

STATE = {"lock": threading.Lock(), "json": {"ok": False}, "jpeg": b""}


class SyntheticSource:
    """A car reversing at 0.3 m/s towards a block on a textured floor (adas/vision/sim_camera.py)."""

    def __init__(self, cam, fps):
        from adas.vision.sim_camera import SimCamera
        self.sim, self.fps, self.i, self.x = SimCamera(cam, (cam.width, cam.height), seed=4), fps, 0, 0.0
        self.box = (-1.6, 0.03, 0.22, 0.22, 0.16, 0.0)

    def read(self):
        time.sleep(1.0 / self.fps)
        self.x = -0.3 * (self.i / self.fps)
        self.i += 1
        if self.x < -1.25:
            self.i, self.x = 0, 0.0
        return True, self.sim.render((self.x, 0.0, 0.0), boxes=[self.box])


class Pipeline:
    def __init__(self, cam, use_yolo=False):
        import cv2
        from adas.vision.detector import Detector
        from adas.vision.quality import ImageQuality
        from adas.vision.rear_objects import LoomingTracker, RearObjects
        from adas.vision.rear_odometry import RearOdometry
        self.cv2, self.cam = cv2, cam
        self.vo, self.quality = RearOdometry(cam), ImageQuality(cam.fx)
        self.objects, self.loom = RearObjects(cam), LoomingTracker()
        self.det = Detector() if use_yolo else None
        self.frames = collections.deque(maxlen=6)            # (gray, t, cumulative dp x, dp y, dpsi)
        self.cum = np.zeros(3)
        self.n, self.t_last, self.fps, self.last_objects, self.last_dets = 0, None, 0.0, [], []

    def process(self, frame, t):
        cv2 = self.cv2
        t0 = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        odo = self.vo.update(gray, t)
        v = w = q = None
        if odo is not None:
            dt = t - (self.frames[-1][1] if self.frames else t - 1e-3)
            v, w, q = odo[0] / max(dt, 1e-3), odo[2] / max(dt, 1e-3), odo[3]
            self.cum += np.array([odo[0], odo[1], odo[2]])
        self.frames.append((gray, t, self.cum.copy()))
        qual = self.quality.update(gray, t)
        self.n += 1
        objs = self.last_objects
        if self.n % 5 == 0 and len(self.frames) == self.frames.maxlen:
            g0, _, c0 = self.frames[0]
            d = self.cum - c0
            blobs = self.objects.detect(g0, gray, (d[0], d[1]), d[2])
            objs = [{"id": tid, "foot": b.foot, "bbox": b.bbox, "ttc": ttc} for tid, b, ttc in self.loom.update(blobs, t)]
            self.last_objects = objs
        if self.det is not None and self.det.available and self.n % 3 == 0:
            self.last_dets = self.det.detect(frame)
        if self.t_last is not None:
            self.fps = 0.9 * self.fps + 0.1 / max(t - self.t_last, 1e-3)
        self.t_last = t
        ms = (time.perf_counter() - t0) * 1000
        vis = frame.copy()
        for o in objs:
            x, y, bw, bh = o["bbox"]
            col = (72, 82, 239) if o["ttc"] is not None and o["ttc"] < 1.5 else (59, 169, 242)
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2)
            label = "object" if o["ttc"] is None else f"object {o['ttc']:.1f}s"
            cv2.putText(vis, label, (x, max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        for name, conf, (x, y, bw, bh) in self.last_dets:
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (241, 243, 240), 1)
            cv2.putText(vis, f"{name} {conf:.2f}", (x, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (241, 243, 240), 1)
        state = {"ok": True, "t": t, "fps": round(self.fps, 1), "pipeline_ms": round(ms, 1),
                 "odometry": None if v is None else {"v": round(v, 3), "w": round(w, 3), "quality": round(q, 2)},
                 "objects": [{"id": o["id"], "foot": o["foot"], "ttc": o["ttc"]} for o in objs],
                 "detections": [[n, round(c, 2), list(b)] for n, c, b in self.last_dets],
                 "quality": {k: (round(x, 2) if isinstance(x, float) else x) for k, x in qual.items()}}
        return vis, state


def serve(port):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                with STATE["lock"]:
                    body = json.dumps(STATE["json"]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        with STATE["lock"]:
                            jpg = STATE["jpeg"]
                        if jpg:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode()
                                             + b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(0.05)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
            else:
                self.send_response(404)
                self.end_headers()
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    import cv2
    from camera_calibrate import load_camera
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--height", type=int, default=240)
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--yolo", action="store_true")
    ap.add_argument("--bench", type=int, default=0, help="process this many frames, print timing, exit")
    a = ap.parse_args()
    cam = load_camera()
    cam = type(cam)(**{**cam.__dict__, "width": a.width, "height": a.height})
    if a.sim:
        src = SyntheticSource(cam, a.fps)
    else:
        src = cv2.VideoCapture(a.index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_V4L2)
        src.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        src.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        src.set(cv2.CAP_PROP_FPS, a.fps)
        src.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))       # webcams do 30 fps only in MJPG
        if not src.isOpened():
            sys.exit(f"camera {a.index} not found")
    pipe = Pipeline(cam, a.yolo)
    if not a.bench:
        serve(a.port)
        print(f"rear camera service on :{a.port}  (/stream, /state)  source: {'synthetic' if a.sim else 'webcam ' + str(a.index)}")
    times = []
    n = 0
    while True:
        ok, frame = src.read()
        if not ok:
            time.sleep(0.05)
            continue
        if frame.shape[1] != a.width:
            frame = cv2.resize(frame, (a.width, a.height))
        t = time.time()
        vis, state = pipe.process(frame, t)
        times.append(state["pipeline_ms"])
        n += 1
        if a.bench:
            if n >= a.bench:
                t_ = np.array(times[5:])
                print(f"pipeline {t_.mean():.1f} ms mean, {np.percentile(t_, 95):.1f} ms 95th, {t_.max():.1f} ms max over {len(t_)} frames")
                return
            continue
        ok, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with STATE["lock"]:
            STATE["json"], STATE["jpeg"] = state, jpg.tobytes()


if __name__ == "__main__":
    main()
