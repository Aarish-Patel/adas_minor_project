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


# ----------------------------------------------------------------------------------- ghost car and reverse assists (Q1-Q2)
def _arc(kappa, s):
    """Pose (x, y, theta) of the car after reversing s metres along the arc of curvature kappa (vehicle frame at s = 0)."""
    sg = -np.asarray(s, dtype=float)
    if abs(kappa) < 1e-6:
        return sg, np.zeros_like(sg), np.zeros_like(sg)
    th = kappa * sg
    return np.sin(th) / kappa, (1 - np.cos(th)) / kappa, th


def footprint_at(kappa, s, width, rear_x, front_x):
    """The car's rectangle (4 corners, vehicle frame at s=0) after reversing s metres. rear_x < 0 < front_x."""
    x, y, th = (float(v[0]) for v in _arc(kappa, [s]))
    c, sn = math.cos(th), math.sin(th)
    local = [(rear_x, -width / 2), (rear_x, width / 2), (front_x, width / 2), (front_x, -width / 2)]
    return np.array([[x + c * lx - sn * ly, y + sn * lx + c * ly] for lx, ly in local])


def first_contact(kappa, hazards, width, rear_x, front_x, length=1.2, step=0.02, margin=0.02):
    """Distance reversed (m) until the swept body would first touch one of the hazard points (vehicle-frame (x, y) pairs), or
    None. This is what the ghost car shows: the same footprint, moved along the same arc."""
    if hazards is None or len(hazards) == 0:
        return None
    h = np.asarray(hazards, dtype=float)
    for s in np.arange(step, length + 1e-9, step):
        x, y, th = (float(v[0]) for v in _arc(kappa, [s]))
        c, sn = math.cos(th), math.sin(th)
        dx, dy = h[:, 0] - x, h[:, 1] - y
        lx, ly = c * dx + sn * dy, -sn * dx + c * dy
        if ((lx >= rear_x - margin) & (lx <= front_x + margin) & (np.abs(ly) <= width / 2 + margin)).any():
            return float(s)
    return None


def floor_patches(gray, cam, kappa, width, bumper_x, length=1.2, min_cells=6):
    """Puddle / hole / oil-patch warning from the camera alone: the floor in the path is warped to a bird's-eye view
    (adas.markers.project); a patch that is much darker or much smoother than the rest of the path (water and wet floor are
    dark and mirror-like, holes are dark) is flagged. Returns a list of (distance along the path m, cells, kind).
    Deliberately a low-cost heuristic (no learned model): it warns, it never brakes."""
    import cv2
    s = np.linspace(0.05, length, 24)
    cx, cy, cth = _arc(kappa, s)
    cx = cx + bumper_x
    grid = np.zeros((len(s), 5))
    vals = np.zeros((len(s), 5))
    lat = np.linspace(-width / 2, width / 2, 5)
    g = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    tex = np.zeros_like(vals)
    for j, l in enumerate(lat):
        px = cx - l * np.sin(cth)
        py = cy + l * np.cos(cth)
        uv, z = project(cam, np.column_stack([px, py, np.zeros_like(px)]))
        for i in range(len(s)):
            u, v = int(round(uv[i][0])), int(round(uv[i][1]))
            if z[i] > 0.02 and 3 <= u < g.shape[1] - 3 and 3 <= v < g.shape[0] - 3:
                vals[i, j] = g[v - 2:v + 3, u - 2:u + 3].mean()
                tex[i, j] = lap[v - 2:v + 3, u - 2:u + 3].mean()
                grid[i, j] = 1
    if grid.sum() < 20:
        return []
    med, tmed = np.median(vals[grid > 0]), np.median(tex[grid > 0])
    dark = (grid > 0) & (vals < 0.62 * med)
    smooth = (grid > 0) & (tex < 0.25 * max(tmed, 1e-3)) & (np.abs(vals - med) > 0.25 * med)
    out = []
    for name, m in (("dark patch / hole", dark), ("wet or glossy patch", smooth)):
        if m.sum() >= min_cells:
            rows = np.where(m.any(axis=1))[0]
            out.append((float(s[rows[0]]), int(m.sum()), name))
    return out


def draw_ghost(frame, cam, kappa, width, rear_x, front_x, contact=None, marks=(0.3, 0.6, 1.0), fill=True):
    """The ghost car: the car's footprint drawn on the floor where it will be after reversing 0.3, 0.6 and 1.0 m at the
    current steering; nearer ghosts are stronger. A ghost past the first contact distance is drawn red."""
    import cv2
    out = frame.copy()
    layer = out.copy()
    for k, d in reversed(list(enumerate(marks))):
        hit = contact is not None and d >= contact
        col = RED if hit else (WHITE if not fill else GREEN)
        # only the rear half of the body matters to the eye on the floor; draw the whole rectangle's bottom outline
        poly = footprint_at(kappa, d, width, rear_x, min(front_x, rear_x + 0.14))     # the part of the body the camera can see
        uv, z = project(cam, np.column_stack([poly, np.zeros(4)]))
        if (z <= 0.02).any():
            continue
        pts = np.int32(uv)
        if fill:
            cv2.fillConvexPoly(layer, pts, col)
        cv2.polylines(out, [pts], True, col, 2 if k == 0 else 1, cv2.LINE_AA)
        cv2.putText(out, f"{d:.1f} m", (int(uv[:, 0].min()), int(uv[:, 1].max()) + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1,
                    cv2.LINE_AA)
    return cv2.addWeighted(layer, 0.22, out, 0.78, 0) if fill else out


def draw_reverse_assist(frame, cam, kappa, width, rear_x, front_x, hazards=None, patches=(), speed=0.0, length=1.2):
    """The whole reverse-camera overlay: guidelines, ghost car, hazards in the swept path, a STOP banner when the ghost hits
    something (with distance and time to contact at the present reverse speed) and floor-patch warnings. Returns
    (image, info) - info is what the HMI shows as text."""
    import cv2
    contact = first_contact(kappa, hazards, width, rear_x, front_x, length)
    img = draw_guidelines(frame, cam, kappa, width, rear_x, length)
    img = draw_ghost(img, cam, kappa, width, rear_x, front_x, contact)
    info = {"contact_m": contact, "ttc_s": None, "patches": [(round(d, 2), n) for d, _, n in patches], "stop": False}
    if hazards is not None and len(hazards):
        uv, z = project(cam, np.column_stack([np.asarray(hazards)[:, :2], np.zeros(len(hazards))]))
        for (u, v), zz in zip(uv, z):
            if zz > 0.02 and 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (int(u), int(v)), 3, AMBER, -1, cv2.LINE_AA)
    if contact is not None:
        # measured from the bumper: how far the body is from the object, not how far the axle would travel
        info["ttc_s"] = None if speed < 0.05 else round(contact / speed, 1)
        info["stop"] = contact < 0.35
        msg = f"OBJECT IN PATH - {contact * 100:.0f} cm" + (f" ({info['ttc_s']:.1f} s)" if info["ttc_s"] else "")
        colr = RED if info["stop"] else AMBER
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), colr, -1)
        cv2.putText(img, ("STOP  " if info["stop"] else "") + msg, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1,
                    cv2.LINE_AA)
    elif patches:
        d, _, name = patches[0]
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), AMBER, -1)
        cv2.putText(img, f"{name.upper()} AHEAD - {d * 100:.0f} cm", (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1,
                    cv2.LINE_AA)
    return img, info
