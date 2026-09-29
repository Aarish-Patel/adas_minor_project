"""Optional learned object detector (YOLO, Ultralytics) for the rear camera. Nothing is downloaded by this code: it uses
weights that are already on disk (env RC_YOLO_WEIGHTS, or models/yolov8n.pt) and reports `available = False` otherwise, so
the rest of the pipeline (parallax blobs, looming, LiDAR fusion) works without it.

On the Pi 5 the nano model at 320 px runs at roughly 8-10 frames/s (NCNN export ~15); it belongs in its own process
(pi/rear_camera.py) so the control loop never waits for it. A floor-level camera on a 1:14 car sees small objects from an
unusual angle - COCO-pretrained weights will miss some; fine-tune on frames recorded by the car (RESEARCH.md section 5).
"""
import os

from adas.vision.fusion import COCO

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


class Detector:
    def __init__(self, weights=None, imgsz=320, conf=0.35):
        self.available, self.model, self.imgsz, self.conf = False, None, imgsz, conf
        path = weights or os.environ.get("RC_YOLO_WEIGHTS") or os.path.join(ROOT, "models", "yolov8n.pt")
        if not os.path.exists(path):
            self.why = f"no weights at {path}"
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self.available = True
            self.why = "ok"
        except Exception as e:                       # missing package / bad weights: run without it
            self.why = str(e)

    def detect(self, frame_bgr):
        """-> [(class name, confidence, (x, y, w, h) px)] for the classes the ADAS cares about."""
        if not self.available:
            return []
        res = self.model.predict(frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)[0]
        out = []
        for b in res.boxes:
            name = COCO.get(int(b.cls))
            if name is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0])
            out.append((name, float(b.conf), (int(x1), int(y1), int(x2 - x1), int(y2 - y1))))
        return out
