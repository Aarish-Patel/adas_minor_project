"""How late does the tracker call a moving object moving, and does it ever call a static one moving? (TODO P2-3)

    python -m sim.tracking_eval                  # writes models/tracking_latency.json (about 10 minutes)

Twin scenes seen through the simulated 720-beam LiDAR at 10 Hz with range noise (default 20 mm, the worst case the project tests) and 4 %
dropout: a room with static furniture (boxes, a chair, cones, a wall stub) and, in the crossing runs, a person (a 0.10 m radius circle: legs / torso)
walking across the car's path. The car stands still or drives straight / on an arc; the tracker is fed the car's speed with a scale error
and noise, as the relay does. Per configuration:
  latency      seconds from the moment the person starts walking (visible and static before that) to the first scan where the track on them
               is `moving` and passes what the consumers require (`is_moving_for_assist`: count >= 5, 0.15..2.0 m/s) - "assist rule"; and
               to the first scan where the track's own `moving` flag is up at 0.15..2.0 m/s - "flag only" (what the consumers could use
               if the count >= 5 filter, which exists because the flag was noisy, went away)
  false rate   static-object track-scans flagged moving / all static track-scans, and the number of distinct static tracks that were
               ever flagged (the requirement: 0 at 20 mm noise)
Same seeds for every tracker variant, so the comparison is paired. Tune on the first seeds, report on untouched ones.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

SCAN_DT = 0.1


def scene(person=None, extra_static=True):
    """The furniture room; `person` = (x, y, vx, vy) of a walker who starts walking at t = 1.0 s (kept still until then)."""
    from sim.hw_worlds import room_walls
    from sim.world import Box, Cone, MovingCircle, Wall, World
    w = World()
    room_walls(w, -1.5, 5.5, -2.0, 2.0)
    if extra_static:
        w.add(Box(2.6, -0.9, 0.40, 0.25, 0.3))
        w.add(Box(3.4, 1.1, 0.30, 0.30))
        w.add(Box(1.2, 1.3, 0.25, 0.20, 0.5))
        for dx, dy in ((0, 0), (0.38, 0), (0, 0.38), (0.38, 0.38)):          # a chair: four 2.4 cm legs
            w.add(Cone(4.0 + dx, -1.2 + dy, 0.012))
        w.add(Cone(2.0, 0.6, 0.05))
        w.add(Wall(4.6, -2.0, 4.6, -0.9))
    ped = None
    if person is not None:
        ped = MovingCircle(person[0], person[1], 0.0, 0.0, 0.10)
        w.add(ped)
    from sim.hw_worlds import _finish
    return _finish(w), ped


class Run:
    """One drive: the LiDAR scans at 10 Hz from a moving car, and the person's (and every static object's) true position."""

    def __init__(self, person=None, v=0.0, kappa=0.0, noise=0.02, seed=0, seconds=4.0, v_scale=1.0, v_noise=0.03, walk_from=1.0):
        from adas.config import load_tuning
        from pi.relay_assists import car_params
        from sim.hw_sim import SimLidar, VirtualCar
        self.world, self.ped = scene(person)
        self.person = person
        tun = load_tuning(os.path.join(HERE, "..", "pi", "tuning_real_car.json"))
        self.p = car_params(tun.mount)
        car = VirtualCar(self.world, self.p, (0.0, 0.0, 0.0), threaded=False)
        self.lidar = SimLidar(car, tun.mount.yaw_offset_deg, n=720, seed=seed)
        self.noise, self.seed, self.seconds = noise, seed, seconds
        self.v, self.kappa, self.v_scale, self.v_noise, self.walk_from = v, kappa, v_scale, v_noise, walk_from
        self.rng = np.random.default_rng(seed + 1000)
        self.w_bias = (0.05, -0.05, 0.0)[seed % 3]           # a servo-centre error turns into a yaw-rate error that grows with speed

    def scans(self):
        """Yield (t, points in the vehicle frame, v estimate, omega estimate, truth) per scan."""
        p, lidar = self.p, self.lidar
        x = y = th = 0.0
        t = 0.0
        while t <= self.seconds + 1e-9:
            if self.ped is not None and t >= self.walk_from - 1e-9:
                self.ped.vx, self.ped.vy = self.person[2], self.person[3]
            ox, oy = x + p.lidar_x * math.cos(th), y + p.lidar_x * math.sin(th)
            best, _ = lidar._raycast(ox, oy, th)
            r = best + self.rng.normal(0, self.noise, len(best))
            ok = np.isfinite(best) & (r >= 0.2) & (r < 12) & (self.rng.random(len(best)) > 0.04)
            c = lidar.ccw[ok]
            xy = np.column_stack([p.lidar_x + r[ok] * np.cos(c), r[ok] * np.sin(c)])
            v_est = self.v * self.v_scale + self.rng.normal(0, self.v_noise) if self.v else self.rng.normal(0, 0.005)
            w_est = self.v * self.kappa * self.v_scale + self.w_bias * (self.v / 0.4) + self.rng.normal(0, 0.02)
            yield t, xy, v_est, w_est, (x, y, th)
            # true motion, then the world moves on
            w_true = self.v * self.kappa
            th_mid = th + 0.5 * w_true * SCAN_DT
            x += self.v * math.cos(th_mid) * SCAN_DT
            y += self.v * math.sin(th_mid) * SCAN_DT
            th += w_true * SCAN_DT
            self.world.update(SCAN_DT) if t >= self.walk_from - 1e-9 else None
            t += SCAN_DT
            self.world_t = t

    def static_points(self):
        """(x, y, kind) of the static things in the world frame, for labelling tracks: cones, box corners' centres, walls (as segments)."""
        return [(o.x, o.y) for o in self.world.objects if o.__class__.__name__ == "Cone"]


def is_moving_for_assist(tr):
    """What `RelayAssists._moving_assist` requires before it reacts (pi/relay_assists.py)."""
    sp = math.hypot(*tr.vel_obj)
    return tr.moving and tr.moving_count >= 5 and 0.15 <= sp <= 2.0


def world_pos(tr_pos, pose):
    x, y, th = pose
    lx = 0.0                                     # tracker positions are in the vehicle frame (rear axle)
    c, s = math.cos(th), math.sin(th)
    return x + c * tr_pos[0] - s * tr_pos[1], y + s * tr_pos[0] + c * tr_pos[1]


def label(run, wpos, ped_now):
    """'person' if the track is within 0.25 m of the walker, else 'static' (everything else in the room is static)."""
    if ped_now is not None and math.hypot(wpos[0] - ped_now[0], wpos[1] - ped_now[1]) < 0.25:
        return "person"
    return "static"


def crossing_run(make_tracker, speed, v_car, kappa, noise, seed, from_side=1.0):
    """One crossing: returns (latency s or None, static flagged track-scans, static track-scans, ids of static tracks ever flagged)."""
    x0 = 2.4
    ped = (x0, 0.9 * from_side, 0.0, -speed * from_side)
    run = Run(person=ped, v=v_car, kappa=kappa, noise=noise, seed=seed, seconds=4.0, v_scale=1.15 if seed % 2 else 0.9)
    tracker = make_tracker()
    lat, lat_flag = None, None
    flagged = total = 0
    ever = set()
    for t, xy, v_est, w_est, pose in run.scans():
        tracks = tracker.update(xy, t, v_est, w_est)
        pn = (run.ped.x, run.ped.y)
        for tr in tracks:
            if tr.missed:
                continue
            kind = label(run, world_pos(tr.pos, pose), pn)
            if kind == "person":
                if lat is None and t >= run.walk_from and is_moving_for_assist(tr):
                    lat = t - run.walk_from
                if lat_flag is None and t >= run.walk_from and tr.moving and 0.15 <= math.hypot(*tr.vel_obj) <= 2.0:
                    lat_flag = t - run.walk_from
            else:
                if len(tr.hist) >= 4:
                    total += 1
                    if is_moving_for_assist(tr):
                        flagged += 1
                        ever.add(tr.id)
    return (lat, lat_flag), flagged, total, ever


def static_run(make_tracker, v_car, kappa, noise, seed, seconds=8.0):
    run = Run(person=None, v=v_car, kappa=kappa, noise=noise, seed=seed, seconds=seconds, v_scale=1.15 if seed % 2 else 0.9)
    tracker = make_tracker()
    raw = assist = total = 0
    ever = set()
    for t, xy, v_est, w_est, pose in run.scans():
        for tr in tracker.update(xy, t, v_est, w_est):
            if tr.missed or len(tr.hist) < 4:
                continue
            total += 1
            raw += bool(tr.moving)
            if is_moving_for_assist(tr):
                assist += 1
                ever.add(tr.id)
    return raw, assist, total, ever


def evaluate(make_tracker, seeds, noise=0.02, verbose=True, label_="tracker"):
    """Crossing latency over speeds x car motion and false-moving rates over static drives."""
    res = {"latency": {}, "static": {}}
    for speed in (0.3, 0.5, 1.0):
        for name, vc, k in (("car still", 0.0, 0.0), ("car 0.4 m/s straight", 0.4, 0.0), ("car 0.4 m/s arc", 0.4, 0.5)):
            lats, lats_f, flagged, total, ever = [], [], 0, 0, 0
            for sd in seeds:
                for side in (1.0, -1.0):
                    (lat, lat_f), f, n, e = crossing_run(make_tracker, speed, vc, k, noise, sd, side)
                    lats.append(lat)
                    lats_f.append(lat_f)
                    flagged += f
                    total += n
                    ever += len(e)
            got = [l for l in lats if l is not None]
            got_f = [l for l in lats_f if l is not None]
            res["latency"][f"{speed} m/s, {name}"] = {"detected": len(got), "of": len(lats),
                                                      "median_s": float(np.median(got)) if got else None,
                                                      "max_s": float(np.max(got)) if got else None,
                                                      "flag_detected": len(got_f),
                                                      "flag_median_s": float(np.median(got_f)) if got_f else None,
                                                      "flag_max_s": float(np.max(got_f)) if got_f else None,
                                                      "static_flagged_track_scans": flagged, "static_track_scans": total,
                                                      "static_tracks_ever_flagged": ever}
    for name, vc, k in (("car still", 0.0, 0.0), ("car 0.3 m/s straight", 0.3, 0.0), ("car 0.6 m/s straight", 0.6, 0.0), ("car 0.4 m/s arc", 0.4, 0.5)):
        raw = assist = total = ever = 0
        for sd in seeds:
            r, a, n, e = static_run(make_tracker, vc, k, noise, sd)
            raw, assist, total, ever = raw + r, assist + a, total + n, ever + len(e)
        res["static"][name] = {"raw_moving_track_scans": raw, "assist_moving_track_scans": assist, "track_scans": total,
                               "static_tracks_ever_assist_moving": ever}
    if verbose:
        print(f"== {label_} (noise {noise * 1000:.0f} mm, seeds {list(seeds)})")
        for k, v in res["latency"].items():
            med = "-" if v["median_s"] is None else f"{v['median_s']:.2f}"
            mx = "-" if v["max_s"] is None else f"{v['max_s']:.2f}"
            fm = "-" if v["flag_median_s"] is None else f"{v['flag_median_s']:.2f}"
            print(f"  crossing {k:34s} assist rule (count>=5) {v['detected']:2d}/{v['of']:2d} median {med:>5s} max {mx:>5s} s | "
                  f"flag only {v['flag_detected']:2d}/{v['of']:2d} median {fm:>5s} s | static flagged {v['static_flagged_track_scans']}/{v['static_track_scans']} ({v['static_tracks_ever_flagged']} tracks)")
        for k, v in res["static"].items():
            print(f"  static   {k:34s} raw-moving {v['raw_moving_track_scans']:4d}  assist-moving {v['assist_moving_track_scans']:3d}"
                  f" of {v['track_scans']} track-scans, static tracks ever assist-moving: {v['static_tracks_ever_assist_moving']}")
    return res


def main():
    """Tune seeds 0-2 (thresholds were chosen on these), confirmation seeds 3-8 (never used for tuning), and other noise levels."""
    from adas.tracking import Tracker
    legacy = lambda: Tracker(model="legacy")
    lrt = lambda: Tracker(model="lrt")
    tune, confirm = range(0, 3), range(3, 9)
    out = {"tune_seeds": list(tune), "confirm_seeds": list(confirm)}
    for name, seeds in (("tune", tune), ("confirm", confirm)):
        out[name] = {"legacy": evaluate(legacy, seeds, label_=f"legacy, {name} seeds"), "lrt": evaluate(lrt, seeds, label_=f"lrt, {name} seeds")}
    for noise in (0.008, 0.03):
        out[f"noise_{int(noise * 1000)}mm"] = {"legacy": evaluate(legacy, confirm, noise, label_="legacy"),
                                                "lrt": evaluate(lrt, confirm, noise, label_="lrt")}
    json.dump(out, open(os.path.join(HERE, "..", "models", "tracking_latency.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
