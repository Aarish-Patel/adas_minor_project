"""Reverse guidelines on the rear camera image - the standard reversing-camera overlay, from the steering (TODO P20).

The two wheel tracks the car will follow at the current steering (an arc of curvature kappa, body width) are drawn on the
floor in the image, coloured by distance from the bumper: red < 0.3 m, amber < 0.6 m, green beyond, with distance bars
across the path. Built from the same camera geometry as everything else (adas.markers.project), so they lie on the floor.
"""
import math

import numpy as np

from adas.markers import project

RED, AMBER, GREEN, WHITE = (72, 82, 239), (59, 169, 242), (127, 201, 67), (240, 243, 241)     # BGR of the HMI palette


def _band(d):
    return RED if d < 0.30 else AMBER if d < 0.60 else GREEN


def draw_guidelines(frame, cam, kappa, width, bumper_x, length=1.2, direction=-1, bars=(0.3, 0.6, 1.0)):
    """Draw on a copy of frame. kappa: path curvature (1/m, + = left) of the car's path; direction -1 = reversing.
    bumper_x: x of the bumper the guidelines start at (vehicle frame; negative for the rear)."""
    import cv2
    out = frame.copy()
    s = np.linspace(0.02, length, 40)
    sg = direction * s
    if abs(kappa) < 1e-6:
        cx, cy, cth = sg, np.zeros_like(sg), np.zeros_like(sg)
    else:
        cth = kappa * sg
        cx, cy = np.sin(cth) / kappa, (1 - np.cos(cth)) / kappa
    cx = cx + bumper_x
    for side in (-1.0, 1.0):
        px = cx - side * (width / 2) * np.sin(cth)
        py = cy + side * (width / 2) * np.cos(cth)
        uv, z = project(cam, np.column_stack([px, py, np.zeros_like(px)]))
        for i in range(len(s) - 1):
            if z[i] <= 0.02 or z[i + 1] <= 0.02:
                continue
            cv2.line(out, tuple(np.int32(uv[i])), tuple(np.int32(uv[i + 1])), _band(s[i]), 3, cv2.LINE_AA)
    for d in bars:
        i = int(np.argmin(np.abs(s - d)))
        pl = np.array([[cx[i] + (width / 2) * np.sin(cth[i]), cy[i] - (width / 2) * np.cos(cth[i]), 0.0],
                       [cx[i] - (width / 2) * np.sin(cth[i]), cy[i] + (width / 2) * np.cos(cth[i]), 0.0]])
        uv, z = project(cam, pl)
        if (z > 0.02).all():
            cv2.line(out, tuple(np.int32(uv[0])), tuple(np.int32(uv[1])), _band(d), 2, cv2.LINE_AA)
            cv2.putText(out, f"{d:.1f} m", (int(uv[1][0]) + 4, int(uv[1][1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1,
                        cv2.LINE_AA)
    return out
