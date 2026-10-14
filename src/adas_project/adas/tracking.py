"""Moving-object tracking and collision prediction from LiDAR scans.

Each scan is split into compact clusters (pedestrians, cones, boxes, other
cars; long walls are ignored). Clusters are followed from scan to scan, and a
straight-line fit over the last ~0.5 s gives each one's velocity as seen from
the car. Subtracting the motion the car itself causes (from its speed and
steering) leaves the object's own velocity. Objects that really move are then
projected forward in time and checked against the car's footprint moving along
its steering arc.

Only relative motion is used, so no wheel encoder is needed. The car's speed
only has to be roughly right, and the check tolerates +/-30 % error in it.

"Is it moving?" (TODO P2-3, sim/tracking_eval.py). The first version thresholded the fitted speed (0.10 m/s) and then wanted several
scans in a row, whatever the noise; at 20 mm range noise that flagged a fifth of the wall fragments as moving (a wall cut into
pieces by grazing gaps has a centroid that slides along it), and it needed >= 5 scans to tell a walker from noise. `model="lrt"`
replaces it with a likelihood-ratio test on the track's own history, the standard way to decide 'moving or static' from noisy
positions (a chi-square test on the Kalman-style innovation, e.g. Bar-Shalom, Li & Kirubarajan, "Estimation with Applications to
Tracking and Navigation", 2001, ch. 2 and 11; sequential form after Wald 1945):
  * every position sample carries a covariance: range noise over sqrt(points) plus a sampling jitter, and along the cluster's own
    axis a much larger term for each end that is not a real edge (an end cut by a gap or noise, not by a depth step to the background):
    the aperture problem - motion along a line is not observable from the line;
  * a weighted least-squares line through the last K samples gives the velocity and its covariance; the static hypothesis is the
    car's own motion field, whose error (dv, dw) is estimated from the other tracks (every fixed object must show the same error:
    a robust consensus with the speed model's error as the prior) and enters the covariance; the statistic is the squared
    Mahalanobis distance of the velocity from that hypothesis (chi-square, 2 dof under 'static');
  * K runs over 3..N (a scan statistic), so an object that stood still and starts to walk is seen after ~3 moving samples rather than
    when they outweigh the standing ones in a 0.8 s window. Fast movers cross the threshold with fewer samples than slow ones.
`Track.evidence` is the statistic; `Track.moving` is set after `lrt_persist` scans above the threshold (with hysteresis) and needs no
further filter. A cluster cut at both ends (a piece of a wall) is never `moving` but still serves as a landmark for the ego motion.
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .geometry import path_pose
from .vehicle_params import VehicleParams


@dataclass
class Track:
    id: int
    hist: deque = field(default_factory=lambda: deque(maxlen=8))   # (t, x, y) in each scan's vehicle frame
    radius: float = 0.03
    missed: int = 0
    vel_rel: tuple = (0.0, 0.0)     # velocity seen from the car
    vel_obj: tuple = (0.0, 0.0)     # the object's own velocity (car motion removed)
    moving: bool = False
    moving_count: int = 0
    meas: deque = field(default_factory=lambda: deque(maxlen=8))   # (sigma along the cluster axis, sigma across, axis angle) per sample
    evidence: float = 0.0           # chi-square statistic of 'not static' (lrt model)
    soft: int = 0                   # cut ends of the latest cluster (2 = a piece of a wall: never flagged moving, still a landmark for the ego motion)

    @property
    def pos(self):
        return self.hist[-1][1], self.hist[-1][2]


def cluster_points(points, gap=0.08, min_pts=3, max_extent=0.45):
    """Split ordered scan points into compact clusters. Returns [(centroid, radius)]."""
    if len(points) < min_pts:
        return []
    steps = np.hypot(*np.diff(points, axis=0).T)
    groups = np.split(np.arange(len(points)), np.flatnonzero(steps > gap) + 1)

    out = []
    for g in groups:
        if len(g) < min_pts:
            continue
        pts = points[g]
        centre = pts.mean(axis=0)
        radius = float(np.hypot(*(pts - centre).T).max())
        if 2.0 * radius > max_extent:
            continue
        out.append((centre, max(radius, 0.02)))
    return out


def clusters_ex(points, gap=0.08, min_pts=3, max_extent=0.45, lidar_x=0.12, edge_jump=0.25):
    """Like `cluster_points`, plus what a position measurement of each cluster is worth: [(centre, radius, axis angle, soft_ends, n, length)].
    An end of a cluster is a real edge if the neighbouring beam hits something at least `edge_jump` further away (the object
    stands in front of a background); otherwise it is a cut (a gap at grazing incidence, a dropout, noise) and the cluster's
    extent - hence its centroid - along its own axis is not observable."""
    if len(points) < min_pts:
        return []
    steps = np.hypot(*np.diff(points, axis=0).T)
    splits = np.flatnonzero(steps > gap) + 1
    groups = np.split(np.arange(len(points)), splits)
    rng = np.hypot(points[:, 0] - lidar_x, points[:, 1])
    out = []
    for g in groups:
        if len(g) < min_pts:
            continue
        pts = points[g]
        centre = pts.mean(axis=0)
        radius = float(np.hypot(*(pts - centre).T).max())
        if 2.0 * radius > max_extent:
            continue
        soft = 0
        for i_end, i_nb in ((g[0], g[0] - 1), (g[-1], g[-1] + 1)):
            if i_nb < 0 or i_nb >= len(points) or steps[min(i_end, i_nb)] <= gap or rng[i_nb] - rng[i_end] < edge_jump:
                soft += 1                                        # the scan ends, or the neighbour is close (the cut is not an edge)
        w, vec = np.linalg.eigh(np.cov((pts - centre).T))            # principal axis (the line the cluster lies along)
        axis = math.atan2(vec[1, 1], vec[0, 1])
        length = float(np.ptp((pts - centre) @ vec[:, 1]))
        out.append((centre, max(radius, 0.02), axis, soft, len(g), length))
    return out


def static_velocity(x, y, v, omega):
    """Velocity, in the vehicle frame, of a point that is fixed in the world."""
    return (-v + omega * y, -omega * x)


class Tracker:
    """`model="lrt"` (default): likelihood-ratio moving test on the track history, see the module docstring. `model="legacy"`: the
    fitted-speed threshold with persistence (the original), kept so sim/tracking_eval.py can compare them on the same scans."""

    def __init__(self, gate=None, min_samples=4, max_missed=3, move_threshold=0.10, persist=3, model="lrt", lidar_x=0.12,
                 range_sigma=0.02, jitter=0.015, soft_sigma=0.06, soft_par_sigma=0.7, ego_v=(0.02, 0.08), ego_w=(0.005, 0.15), chi2=18.4, chi2_hold=9.2,
                 lrt_persist=2, lrt_min_samples=3, window_s=0.8):
        self.gate = gate if gate is not None else (0.30 if model == "legacy" else 0.15)   # the lrt test needs tracks that do not hop between neighbours
        self.min_samples = min_samples
        self.max_missed = max_missed
        self.move_threshold = move_threshold
        self.persist = persist
        self.model = model
        self.lidar_x = lidar_x
        self.range_sigma, self.jitter, self.soft_sigma, self.soft_par_sigma = range_sigma, jitter, soft_sigma, soft_par_sigma
        self.ego_v, self.ego_w = ego_v, ego_w                     # ego-motion sigma = a + b |v|: a yaw-rate error grows with speed (servo centre)
        self.chi2, self.chi2_hold = chi2, chi2_hold               # flag / keep-flagged thresholds (hysteresis)
        self.lrt_persist, self.lrt_min_samples, self.window_s = lrt_persist, lrt_min_samples, window_s
        self.tracks = []
        self._next_id = 1

    def _sigmas(self, n, soft, length):
        """(sigma along the cluster axis, sigma across it) of the centroid. Along the axis the position is not observable when an end is a
        cut (a fragment of a wall: its centroid wanders by the fragment's own extent), so each cut end adds `soft_par_sigma` (a
        weight, ~no information); across the axis the centroid noise plus the real along-axis wander (`soft_sigma` per cut end)
        leaking in through the error of the fitted axis (first order: n points spread over `length` with per-point noise sigma give
        an axis off by about sigma * sqrt(12 / n) / length)."""
        s_across = math.sqrt(self.range_sigma ** 2 / max(n, 1) + self.jitter ** 2)
        along = soft * self.soft_sigma
        s_par = math.sqrt(s_across ** 2 + (soft * self.soft_par_sigma) ** 2)
        eps = min(self.range_sigma * math.sqrt(12.0 / max(n, 1)) / max(length, 0.03), 1.0)
        return s_par, math.sqrt(s_across ** 2 + (along * eps) ** 2)

    def update(self, points, t, v, omega):
        """v: signed speed estimate (m/s), omega: yaw rate (rad/s). Returns the track list."""
        clusters = [((c[0], c[1]), c[2:]) for c in clusters_ex(points, lidar_x=self.lidar_x)]
        unmatched = list(range(len(clusters)))

        for tr in self.tracks:
            tr.missed += 1
            if not unmatched:
                continue
            dt = t - tr.hist[-1][0]
            px = tr.pos[0] + tr.vel_rel[0] * dt
            py = tr.pos[1] + tr.vel_rel[1] * dt
            dists = [math.hypot(clusters[i][0][0][0] - px, clusters[i][0][0][1] - py) for i in unmatched]
            k = int(np.argmin(dists))
            if dists[k] < self.gate:
                i = unmatched.pop(k)
                ((cx, cy), r), (axis, soft, n, length) = clusters[i]
                tr.hist.append((t, float(cx), float(cy)))
                tr.meas.append((*self._sigmas(n, soft, length), axis))
                tr.soft = soft
                tr.radius = 0.7 * tr.radius + 0.3 * r
                tr.missed = 0

        for i in unmatched:
            ((cx, cy), r), (axis, soft, n, length) = clusters[i]
            tr = Track(self._next_id, radius=r)
            tr.hist.append((t, float(cx), float(cy)))
            tr.meas.append((*self._sigmas(n, soft, length), axis))
            tr.soft = soft
            self._next_id += 1
            self.tracks.append(tr)

        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]
        for tr in self.tracks:
            while len(tr.hist) > 2 and t - tr.hist[0][0] > self.window_s:
                tr.hist.popleft()
                tr.meas.popleft()
        if self.model == "legacy":
            for tr in self.tracks:
                self._estimate(tr, v, omega)
        else:
            self._estimate_all(v, omega)
        return self.tracks

    # ------------------------------------------------------------------ likelihood-ratio test
    def _windows(self, tr):
        """Weighted least-squares lines p(t) = p0 + v (t - t_now) through the newest K samples, K = lrt_min_samples..n, each sample with
        its own 2x2 covariance (along / across the cluster axis). One vectorised pass (prefix sums of the normal equations from the
        newest sample back): returns (K values, velocities (m, 2), velocity covariances (m, 2, 2)) or None."""
        n = len(tr.hist)
        ts = np.array([h[0] for h in tr.hist])
        P = np.array([[h[1], h[2]] for h in tr.hist])
        m = np.array(tr.meas)
        c, sn = np.cos(m[:, 2]), np.sin(m[:, 2])
        ip, iq = 1.0 / m[:, 0] ** 2, 1.0 / m[:, 1] ** 2
        W = np.empty((n, 2, 2))
        W[:, 0, 0] = c * c * ip + sn * sn * iq
        W[:, 0, 1] = W[:, 1, 0] = c * sn * (ip - iq)
        W[:, 1, 1] = sn * sn * ip + c * c * iq
        tau = ts - ts[-1]
        A = np.zeros((n, 4, 4))
        A[:, :2, :2] = W
        A[:, :2, 2:] = A[:, 2:, :2] = tau[:, None, None] * W
        A[:, 2:, 2:] = (tau ** 2)[:, None, None] * W
        wp = np.einsum("nij,nj->ni", W, P)
        b = np.concatenate([wp, tau[:, None] * wp], axis=1)
        Ac, bc = np.cumsum(A[::-1], axis=0), np.cumsum(b[::-1], axis=0)          # row k = the newest k + 1 samples
        ks = np.array([k for k in range(self.lrt_min_samples, n + 1) if ts[-1] - ts[-k] >= 0.15])
        if not len(ks):
            return None
        try:
            cov = np.linalg.inv(Ac[ks - 1])
        except np.linalg.LinAlgError:
            return None
        th = np.einsum("mij,mj->mi", cov, bc[ks - 1])
        return ks, th[:, 2:], cov[:, 2:, 2:]

    def _ego_prior(self, v):
        return self.ego_v[0] + self.ego_v[1] * abs(v), self.ego_w[0] + self.ego_w[1] * abs(v)

    def _consensus(self, fits, v, omega):
        """The car's own-motion error (dv, dw) seen by the static world: every track that is not moving is a landmark whose apparent
        velocity must equal the car's motion field u(p; v + dv, w + dw) = (-(v + dv) + (w + dw) y, -(w + dw) x). Robust (Huber)
        weighted least squares with the throttle / steering model's error as the prior. Returns (delta, covariance of delta)."""
        sv, sw = self._ego_prior(v)
        P0i = np.diag([1.0 / sv ** 2, 1.0 / sw ** 2])
        H = P0i.copy()
        g = np.zeros(2)
        rows = []
        for tr, (ks, vs, cvs) in fits:
            if len(tr.hist) < 4 or tr.moving or tr.evidence >= self.chi2_hold:
                continue
            i = min(len(ks) - 1, int(np.searchsorted(ks, 6)))                 # a window of about 6 samples
            x, y = tr.pos
            J = np.array([[-1.0, y], [0.0, -x]])
            u = np.array(static_velocity(x, y, v, omega))
            S = cvs[i] + (0.02 ** 2) * np.eye(2)
            rows.append((J, vs[i] - u, np.linalg.inv(S)))
        delta = np.zeros(2)
        if not rows:
            return delta, np.diag([sv ** 2, sw ** 2])
        for _ in range(3):
            H, g = P0i.copy(), np.zeros(2)
            for J, r, Si in rows:
                e = r - J @ delta
                d2 = float(e @ Si @ e)
                wgt = 1.0 if d2 < 6.0 else math.sqrt(6.0 / d2)              # Huber
                H += wgt * J.T @ Si @ J
                g += wgt * J.T @ Si @ r
            delta = np.linalg.solve(H, g)
        return delta, np.linalg.inv(H)

    def _estimate_all(self, v, omega):
        fits = []
        for tr in self.tracks:
            f = self._windows(tr) if len(tr.hist) >= self.lrt_min_samples else None
            if f is None:
                tr.moving, tr.vel_obj, tr.moving_count, tr.evidence = False, (0.0, 0.0), 0, 0.0
                continue
            fits.append((tr, f))
        delta, Pd = self._consensus(fits, v, omega)
        for tr, (ks, vs, cvs) in fits:
            x, y = tr.pos
            J = np.array([[-1.0, y], [0.0, -x]])
            u = np.array(static_velocity(x, y, v + delta[0], omega + delta[1]))
            S = cvs + (J @ Pd @ J.T + (0.01 ** 2) * np.eye(2))[None]
            r = vs - u[None]
            Si = np.linalg.inv(S)
            d2 = np.einsum("mi,mij,mj->m", r, Si, r)
            speed = np.hypot(r[:, 0], r[:, 1])
            d2 = np.where(speed >= self.move_threshold, d2, 0.0)
            k = int(np.argmax(d2))
            tr.evidence = float(d2[k])
            tr.vel_rel = (float(vs[k, 0]), float(vs[k, 1]))
            tr.vel_obj = (float(r[k, 0]), float(r[k, 1])) if d2[k] > 0 else (0.0, 0.0)
            if tr.soft >= 2:                                    # a piece of a wall: helps to know the ego motion, is never 'moving'
                tr.moving_count, tr.moving = 0, False
                tr.vel_obj = (0.0, 0.0)
                continue
            was = tr.moving_count >= self.lrt_persist
            tr.moving_count = tr.moving_count + 1 if tr.evidence >= (self.chi2_hold if was else self.chi2) else 0
            tr.moving = tr.moving_count >= self.lrt_persist

    def _estimate(self, tr, v, omega):
        tr.moving = False
        tr.vel_obj = (0.0, 0.0)
        if len(tr.hist) < self.min_samples:
            return
        ts = np.array([h[0] for h in tr.hist])
        if ts[-1] - ts[0] < 0.25:
            tr.moving_count = 0
            return
        vx = float(np.polyfit(ts, [h[1] for h in tr.hist], 1)[0])
        vy = float(np.polyfit(ts, [h[2] for h in tr.hist], 1)[0])
        tr.vel_rel = (vx, vy)

        x, y = tr.pos
        # The car's speed and turn rate are only approximate (both scale together
        # with speed), so let a single factor k in [0.7, 1.3] explain as much of the
        # motion as possible as a static object. Only what is left over counts.
        ux, uy = static_velocity(x, y, v, omega)
        norm2 = ux * ux + uy * uy
        k = 1.0 if norm2 < 1e-9 else min(max((vx * ux + vy * uy) / norm2, 0.7), 1.3)
        rx, ry = vx - k * ux, vy - k * uy
        tr.vel_obj = (rx, ry)

        if math.hypot(rx, ry) > self.move_threshold:
            tr.moving_count += 1
        else:
            tr.moving_count = 0
        tr.moving = tr.moving_count >= self.persist


def moving_object_contact(tracks, delta, direction, v, p, horizon=2.0, dt=0.02, margin=0.02):
    """Earliest predicted collision with a MOVING object if the car keeps its speed and steering.

    Returns (travel_distance, time, track_id). Distance is how far the car
    travels before contact; inf if no collision is predicted within `horizon`.
    """
    best = (math.inf, math.inf, None)
    movers = [t for t in tracks if t.moving]
    if not movers or v < 0.03:
        return best

    tau = np.arange(0.0, horizon + dt, dt)
    ex, ey, epsi = path_pose(delta, direction * v * tau, p)
    cos_p, sin_p = np.cos(epsi), np.sin(epsi)

    for tr in movers:
        ox = tr.pos[0] + tr.vel_obj[0] * tau
        oy = tr.pos[1] + tr.vel_obj[1] * tau
        dx, dy = ox - ex, oy - ey
        lx = cos_p * dx + sin_p * dy
        ly = -sin_p * dx + cos_p * dy
        gx = np.maximum(np.maximum(p.rear_x - lx, 0.0), lx - p.front_x)
        gy = np.maximum(np.abs(ly) - p.width / 2.0, 0.0)
        hit = np.flatnonzero(np.hypot(gx, gy) <= margin + tr.radius)
        if len(hit) and tau[hit[0]] < best[1]:
            best = (float(v * tau[hit[0]]), float(tau[hit[0]]), tr.id)
    return best
