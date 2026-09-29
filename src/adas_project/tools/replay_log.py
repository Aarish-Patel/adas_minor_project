"""Replay a real relay drive log through the CURRENT decision code (TODO P19: "reference logs to find issues").

    python tools/replay_log.py logs/car_20260928/20260928_212420_relay.jsonl.gz [--from 0] [--to 9999] [--assists evasive]
                               [--manual-only] [--csv out.csv]

The real scans (raw RPLIDAR angle x100, distance mm), the driver's input lines and the tuning at the time come from the
log. The state (speed estimate, obstacle memory, driver history) is driven by what the car REALLY did (the logged
commands and servo), so the scans stay consistent with it; at every logged driver packet the current code then decides
what it would send. Comparing that with what was actually sent (the old code) on identical sensor data shows which
interventions the changes since then added or removed - open loop, so it cannot say what the car would have done next,
only what each version would have asked for in the same situation.

Reported: episodes (a contiguous stretch of ticks where the throttle was changed), split by kind, for the log and for
the replay; the throttle removed; ticks that drove against the driver's direction.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
from drivetrain_report import records  # noqa: E402


def load(path, t_from, t_to):
    meta, scans, drv = None, [], []
    t0 = None
    for r in records(path):
        k = r.get("k")
        if k == "meta" and meta is None:
            meta = r
        elif k in ("scan", "drv"):
            t = float(r["t"])
            t0 = t if t0 is None else t0
            if not (t_from <= t - t0 <= t_to):
                continue
            (scans if k == "scan" else drv).append(r)
    return meta, scans, drv, t0


def parse_inp(inp):
    servo, wire = None, None
    for ln in str(inp or "").splitlines():
        q = ln.split()
        try:
            if q[0] == "A" and len(q) == 3:
                servo = (float(q[1]) + float(q[2])) / 2
            elif q[0] == "M" and len(q) == 2:
                wire = float(q[1])
        except (ValueError, IndexError):
            pass
    return servo, wire


def scan_points(rec, front_offset, min_range):
    a = np.array(rec["a"], float) / 100.0
    d = np.array(rec["d"], float) / 1000.0
    ok = d >= min_range
    ang = (a[ok] - front_offset) % 360
    ang = np.where(ang > 180, ang - 360, ang)
    return list(zip(np.round(ang, 1).tolist(), np.round(d[ok], 3).tolist()))


def episodes(flags, times, gap=0.5):
    out, cur = [], None
    for f, t in zip(flags, times):
        if f:
            if cur is None or t - cur[1] > gap:
                cur = [t, t]
                out.append(cur)
            else:
                cur[1] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=1e9)
    ap.add_argument("--assists", nargs="*", default=["evasive"])
    ap.add_argument("--manual-only", action="store_true", help="only ticks where the old code was not in autonomy")
    ap.add_argument("--csv")
    a = ap.parse_args()

    from adas.config import load_tuning
    from pi.path_gate import PathGate
    from pi.relay_assists import RelayAssists, RelayIntent, RelaySpeed, ThrottleSmoother, apply_car_model
    from adas.tracking import Tracker

    meta, scans, drv, t0 = load(a.log, a.t_from, a.t_to)
    tun = load_tuning(os.path.join(ROOT, "pi", "tuning_real_car.json"))
    apply_car_model(tun)
    m = (meta or {}).get("tuning", {}).get("mount", {})
    front_offset = float(m.get("yaw_offset_deg", tun.mount.yaw_offset_deg))
    min_range = float(m.get("min_valid_range_m", 0.2))
    tun.mount.yaw_offset_deg = front_offset
    assist = RelayAssists(tun, plan_mode="inline")
    for n in a.assists:
        assist.set(n, True)
    rint = RelayIntent(assist)
    gate = PathGate(assist.p, tun.speed_model)
    assist.memory = gate.memory
    vest = RelaySpeed(tun.speed_model, assist.p.lidar_x)
    gate.delay_source = vest.latency
    gate.uncertainty_source = vest
    assist.speed = vest
    tracker = Tracker()
    smoother = ThrottleSmoother()
    centre = assist.centre
    print(f"{len(scans)} scans, {len(drv)} driver packets, LiDAR offset {front_offset} deg, assists {a.assists}")

    si, pts, seq = 0, [], 0
    rows = []
    last_t = None
    for r in drv:
        t = float(r["t"])
        while si < len(scans) and float(scans[si]["t"]) <= t:
            pts = scan_points(scans[si], front_offset, min_range)
            seq += 1
            vxy = RelayAssists.points_vehicle_frame(pts, assist.p.lidar_x)
            gate.on_scan(vxy, seq)
            vest.on_scan(pts, seq, float(scans[si]["t"]))
            assist.set_tracks(tracker.update(vxy, float(scans[si]["t"]), vest.v, vest.w))
            si += 1
        dt = 0.05 if last_t is None else min(0.2, max(0.005, t - last_t))
        last_t = t
        servo, wire = parse_inp(r.get("inp"))
        if servo is None or wire is None:
            continue
        phys_in = -wire
        info_old = r.get("assist_info") or {}
        autonomy = "autonomy" in info_old
        out_old = r.get("out") or []
        wire_old = next((float(l.split()[1]) for l in out_old if l.startswith("M ")), wire)
        phys_old = -wire_old
        # the current code's decision
        lines = [f"A {servo:.1f} {servo:.1f}", f"M {int(wire)}"]
        rint.update(t, servo, float(phys_in), vest.v, pts)
        out = assist.process(lines, pts, seq, now=t)
        servo_new, phys = servo, float(phys_in)
        for ln in out:
            q = ln.split()
            if q[0] == "A":
                servo_new = (float(q[1]) + float(q[2])) / 2
            elif q[0] == "M":
                phys = -float(q[1])
        delta = math.atan(-assist.k * (servo_new - assist.centre) * assist.p.wheelbase)
        g_phys, g_brake = gate.decide(dt, phys, delta, vest.v_gate((phys > 0) - (phys < 0)), 0.0,
                                      intent_k_rate=rint.gate_k_rate, trusted=rint.gate_trust, leg=assist.planned_leg())
        act = str(gate.info.get("action") or "")
        g_phys = smoother.step(g_phys, dt, emergency=bool(g_brake) or act.startswith(("holding", "stopped")), v=vest.v)
        # state follows what the car really did
        vest.command(t, dt, phys_old, float(next((float(l.split()[1]) for l in out_old if l.startswith("A ")), servo)))
        rows.append({"t": t - t0, "v": float(r.get("v_est") or 0.0), "phys_in": phys_in, "old": phys_old,
                     "new": g_phys, "old_action": (r.get("gate") or {}).get("action"), "new_action": act or None,
                     "v_allowed_old": (r.get("gate") or {}).get("v_allowed"), "v_allowed_new": gate.info.get("v_allowed"),
                     "autonomy": autonomy, "evasive_new": bool(assist.assists.evading)})
    assist.planner.shutdown()

    sel = [x for x in rows if abs(x["phys_in"]) > 20 and (not a.manual_only or not x["autonomy"])]
    print(f"{len(sel)} ticks with the throttle held{' (manual only)' if a.manual_only else ''}")
    for tag, key, act_key, va in (("OLD code (as logged)", "old", "old_action", "v_allowed_old"),
                                  ("NEW code (replayed)", "new", "new_action", "v_allowed_new")):
        changed = [abs(x[key] - x["phys_in"]) > 2.0 for x in sel]
        times = [x["t"] for x in sel]
        eps = episodes(changed, times)
        cut = sum(max(0.0, abs(x["phys_in"]) - abs(x[key])) for x in sel) / 255.0 * 0.05
        rev = sum(1 for x in sel if x["phys_in"] * x[key] < 0)
        kinds = {}
        for x in sel:
            if abs(x[key] - x["phys_in"]) > 2.0:
                kk = str(x[act_key] or ("evasive" if x["evasive_new"] and key == "new" else "assist"))
                kk = "braking (reverse pulse)" if kk.startswith("braking") else kk.split(" (")[0]
                kinds[kk] = kinds.get(kk, 0) + 1
        print(f"{tag}: {len(eps)} episodes, {sum(changed)} ticks changed, throttle removed {cut:.1f} throttle-s, "
              f"reverse-drive ticks {rev};  by kind {kinds}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
