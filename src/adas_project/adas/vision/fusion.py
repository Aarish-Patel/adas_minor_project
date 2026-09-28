"""LiDAR + camera fusion for the rear sector: what IS the thing the LiDAR sees behind the car (TODO P20).

The LiDAR gives exact geometry but no identity; the camera gives identity (person / car / chair / box) but only
approximate range. Late fusion (object level): every LiDAR cluster is projected into the image (adas.markers.project,
extrinsics from the mounting) and takes the class of the detection box that contains it. The class then sets the
margin the ADAS keeps and how it predicts the object: a person is given a wider berth (unpredictable, fragile) than a
box, a car / pet / chair in between.
"""
import numpy as np

from adas.markers import project

# class -> (margin multiplier for the crossing planner / safety field, may move)
CLASS_POLICY = {"person": (1.8, True), "child": (2.2, True), "dog": (1.8, True), "cat": (1.6, True), "car": (1.3, True),
                "bicycle": (1.5, True), "chair": (1.0, False), "box": (1.0, False), "unknown": (1.0, False)}
COCO = {0: "person", 1: "bicycle", 2: "car", 3: "car", 5: "car", 7: "car", 15: "cat", 16: "dog", 56: "chair", 57: "chair"}


def label_clusters(clusters, detections, cam, height=0.06, slack_px=12):
    """clusters: [(cx, cy, radius)] in the vehicle frame; detections: [(class name, confidence, (x, y, w, h) px)].
    -> [{'cluster': i, 'cls': name, 'conf': c, 'margin': m, 'may_move': bool}]"""
    out = []
    if not len(clusters):
        return out
    pts = np.array([[c[0], c[1], height] for c in clusters], float)
    uv, z = project(cam, pts)
    for i, ((u, v), depth) in enumerate(zip(uv, z)):
        best = None
        if depth > 0.02 and 0 <= u < cam.width and 0 <= v < cam.height:
            for name, conf, (x, y, w, h) in detections:
                if x - slack_px <= u <= x + w + slack_px and y - slack_px <= v <= y + h + slack_px:
                    if best is None or conf > best[1]:
                        best = (name, conf)
        name, conf = best if best else ("unknown", 0.0)
        margin, moves = CLASS_POLICY.get(name, CLASS_POLICY["unknown"])
        out.append({"cluster": i, "cls": name, "conf": float(conf), "margin": margin, "may_move": moves})
    return out
