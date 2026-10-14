"""End-to-end parking test on the twin relay the HMI talks to (tools/sim_car.py) in the Parking Simulator rooms.

    python -m sim.parking_e2e [quick]    # starts its own relay on UDP 4310 / HTTP 8190 (the HMI's 4210 / 8090 stay untouched)

Exactly the HMI's path: HTTP POST /api/sim/world/<room>, ORIGIN, the layout from the live stream (`sim.parking`), a click on the strip
-> gui.parking.ParkingModel.click -> gui.parking.park_target_command -> PARK_TARGET (or PARK_MARKER in the bays room), then a scripted
driver holds the throttle (the dead-man switch) in 20 Hz packets until the relay reports it is done. The result is measured on the
simulator's ground truth (`sim.pose`, the room's own layout): success = 0 contacts, position <= 5 cm and heading <= 5 deg from the slot the
parking rule gives on the TRUE parked cars (sim/parking_validation.py), inside that gap; in the bays room from the marker's bay.
"""
import json
import math
import os
import socket
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
UDP, HTTP = 4310, 8190


def serve():
    """The twin relay as tools/sim_car.py starts it, on the test ports."""
    import pi.wifi_drive_safety as R
    R.UDP_PORT, R.GUI_PORT = UDP, HTTP
    sys.argv = ["sim_car", "--world", "park_street"]
    os.environ.setdefault("RC_LOG_DIR", os.path.join(ROOT, "logs", "sim"))
    from tools import sim_car
    sim_car.main()


class Client:
    """The HMI's link (gui.dashboard.Link) pointed at the test relay, plus a 20 Hz driver."""

    def __init__(self):
        import gui.dashboard as D
        D.CTRL_PORT, D.HTTP_PORT = UDP, HTTP
        self.link = D.Link("127.0.0.1")
        self.drv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def state(self, wait=2.0):
        t0 = time.time()
        while time.time() - t0 < wait:
            st = self.link.latest
            if st is not None and time.time() - self.link.latest_t < 0.5:
                return st
            time.sleep(0.05)
        return None

    def cmd(self, text):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(3.0)
        s.sendto(text.encode(), ("127.0.0.1", UDP))
        try:
            return s.recvfrom(256)[0].decode()
        except OSError:
            return ""

    def drive(self, pwm):
        self.drv.sendto(f"A 87 87\nM {-int(pwm)}\n".encode(), ("127.0.0.1", UDP))


ROOMS_T_MAX = 90.0


def _truth(room, marker_id=23):
    """The room's own layout in the simulator frame (ground truth), plus the car's dimensions."""
    from sim.parking_worlds import build
    world, start = build(room, marker_id)
    return world.layout, start


def _street_truth(truth):
    """The street's true cars as (s0, s1) along the strip (footprint extents) and the strip itself, from the simulator's own layout."""
    from adas.park_select import ParkingStrip
    st = truth["strip"]
    strip = ParkingStrip(st["points"], st["side"], st.get("depth") or 0.38)
    cars = []
    for v in truth["vehicles"]:
        c, sn = abs(math.cos(v["yaw"])), abs(math.sin(v["yaw"]))
        half = 0.5 * (v["l"] * c + v["w"] * sn)
        s_c = strip.project((v["cx"], v["cy"]))[0]
        cars.append((s_c - half, s_c + half))
    return strip, sorted(cars)


def run_one(c, room, policy, marker_id=23, index=None):
    """One parking run on the relay. policy: 's:<m>' (street: click the lane at <m> along it, the car finds the gap) or 'marker'
    (bays room: PARK_MARKER). Returns a result dict scored on the simulator's ground truth."""
    from adas.park_select import KERB_GAP_M
    from adas.vehicle_params import VehicleParams
    from gui.parking import ParkingModel, align_from_poses, park_target_command, sim_to_relay
    from sim.parking_validation import _body_corners, _expected_slot, _tie_ok, _true_gaps
    c.link.http_post(f"/api/sim/world/{room}" + (f"/{marker_id}" if room == "park_perp" else ""))
    time.sleep(0.8)
    c.cmd("ORIGIN")
    for _ in range(10):
        c.drive(0)
        time.sleep(0.05)
    truth, start = _truth(room, marker_id)
    st = None
    for _ in range(60):                      # a FRESH packet: the car back at the room's start and the relay's frame re-origined there
        st = c.state()                       # (a packet from before the reset gives a wrong simulator -> relay alignment)
        if st and st.get("sim", {}).get("world") == room and st["sim"].get("parking") and st.get("world", {}).get("pose")                 and math.hypot(st["sim"]["pose"][0] - start[0], st["sim"]["pose"][1] - start[1]) < 0.02                 and math.hypot(*st["world"]["pose"][:2]) < 0.02:
            break
        time.sleep(0.1)
    else:
        return {"room": room, "policy": policy, "success": 0, "why": "the relay did not reset to the start"}
    layout = st["sim"]["parking"]
    car = VehicleParams(wheelbase=0.20, width=0.20, front_overhang=0.08, rear_overhang=0.05)
    t_start = time.time()
    crashes0 = int(st["sim"].get("crashes") or 0)
    align = align_from_poses(st["sim"]["pose"], st["world"]["pose"])
    expected = None
    if policy == "marker":
        reply = c.cmd("PARK_MARKER")
    else:
        strip, cars = _street_truth(truth)
        s_click = float(policy.split(":")[1])
        x, y, th = strip.point_at(s_click)
        click_sim = (x - math.sin(th) * strip.side * strip.depth / 2, y + math.cos(th) * strip.side * strip.depth / 2)
        expected = _expected_slot(strip, cars, click_sim, car)
        model = ParkingModel(layout)                       # the layout is in the simulator frame, the click goes to the relay in its own
        msg = model.click(click_sim)
        msg["clicked_world_pose"] = [*sim_to_relay(click_sim, align), msg["clicked_world_pose"][2]]
        reply = c.cmd(park_target_command(layout, msg, align))
    t0 = time.time()
    seen_active, done_msg, crashed = False, "", False
    while time.time() - t0 < ROOMS_T_MAX:
        c.drive(150)
        time.sleep(0.05)
        st = c.state(0.3)
        if not st:
            continue
        nav = st.get("nav") or {}
        crashed = crashed or int(st["sim"].get("crashes") or 0) > crashes0
        if crashed:
            c.cmd("GOTO CANCEL")
            done_msg = "contact"
            break
        if nav.get("state") not in (None, "idle"):
            seen_active = True
        elif seen_active or (time.time() - t0 > 4 and nav.get("state") == "idle"):
            done_msg = nav.get("msg", "")
            break
    for _ in range(15):
        c.drive(0)
        time.sleep(0.05)
    st = c.state() or st
    x, y, th_deg = st["sim"]["pose"]
    th = math.radians(th_deg)
    res = {"room": room, "policy": policy, "reply": reply, "msg": done_msg, "crashes": int(st["sim"].get("crashes") or 0) - crashes0,
           "time_s": round(time.time() - t_start, 1), "pose": [round(x, 3), round(y, 3), round(th_deg, 1)]}
    if policy == "marker":
        goal = (truth["marker"]["x"] + 0.06 + car.rear_overhang, truth["marker"]["y"], 0.0)
        ex, ey, eh = x - goal[0], y - goal[1], math.degrees(math.remainder(th - goal[2], 2 * math.pi))
        res.update(pos_err_cm=round(math.hypot(ex, ey) * 100, 1), head_err_deg=round(abs(eh), 1), inside=True, gap_ok=True)
    else:
        bc = 0.5 * (car.front_x + car.rear_x)
        bx, by = x + bc * math.cos(th), y + bc * math.sin(th)
        s_car, _ = strip.to_strip(np.array([[bx, by]]))
        res["expected_s"] = None if expected is None else round(expected.mid, 2)
        res["parked_s"] = round(float(s_car[0]), 2)
        if expected is None:
            res.update(pos_err_cm=None, head_err_deg=None, inside=False, gap_ok=done_msg.startswith("no"))
        else:
            chosen = expected
            if not expected.gap[0] <= s_car[0] <= expected.gap[1] and _tie_ok(strip, cars, click_sim, car, float(s_car[0])):
                from adas.park_select import choose_slots
                chosen = next(sp for sp in choose_slots(_true_gaps(strip, cars), strip.project(click_sim)[0], car, strip.depth)
                              if sp.gap[0] <= s_car[0] <= sp.gap[1])
            cx, cy, sth = strip.point_at(chosen.mid)
            off = max(strip.depth / 2, strip.depth - KERB_GAP_M - car.width / 2)
            cx, cy = cx - math.sin(sth) * strip.side * off, cy + math.cos(sth) * strip.side * off
            g0, g1 = chosen.gap
            cs = _body_corners((x, y, th), car)
            sc, yc = strip.to_strip(np.array(cs))
            res.update(pos_err_cm=round(math.hypot(bx - cx, by - cy) * 100, 1),
                       head_err_deg=round(abs(math.degrees(math.remainder(th - sth, 2 * math.pi))), 1),
                       inside=bool((sc >= g0 - 0.01).all() and (sc <= g1 + 0.01).all() and (yc >= -0.02).all() and (yc <= strip.depth + 0.02).all()),
                       gap_ok=True)
    if policy != "marker" and expected is None:
        res["success"] = int(res["crashes"] == 0 and res["gap_ok"])               # nothing viable: it must refuse
    else:
        res["success"] = int(res["crashes"] == 0 and res["msg"].startswith("arrived") and res["pos_err_cm"] is not None
                             and res["pos_err_cm"] <= POS_TOL_CM and res["head_err_deg"] <= HEAD_TOL_DEG and res["inside"])
    return res


POS_TOL_CM, HEAD_TOL_DEG = 5.0, 5.0
# the street (sim/parking_worlds.STREET_CARS, strip from x -1.2): gaps 0.55 (too short), 1.30, 0.50 (too short), 1.02, 1.90, 0.86,
# then the open lane end; clicks in a gap, on a car, in a too-short gap
CASES = [("park_street", "s:3.0"), ("park_street", "s:4.6"), ("park_street", "s:6.4"), ("park_street", "s:9.1"),
         ("park_street", "s:3.6"), ("park_street", "s:5.3"), ("park_perp", "marker")]


def main(quick=False, cases=None):
    proc = subprocess.Popen([sys.executable, "-m", "sim.parking_e2e", "serve"], cwd=ROOT,
                            stdout=open(os.path.join(ROOT, "reports", "parking_validation", "e2e_relay.log"), "w"), stderr=subprocess.STDOUT)
    try:
        time.sleep(14)
        c = Client()
        time.sleep(2.0)
        out = []
        for room, pol in (cases or (CASES[:2] if quick else CASES)):
            r = run_one(c, room, pol)
            out.append(r)
            print(json.dumps(r), flush=True)
        json.dump(out, open(os.path.join(ROOT, "reports", "parking_validation", "e2e_results.json"), "w"), indent=1)
        print(f"{sum(r['success'] for r in out)}/{len(out)} passed")
        return out
    finally:
        proc.terminate()


def probe():
    c = Client()
    time.sleep(2.5)
    st = c.state()
    print(json.dumps({k: (v if k != "_pts" else len(v)) for k, v in (st or {}).items() if k in ("sim", "assist", "world", "_pts")}, default=str)[:1800])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        serve()
    elif len(sys.argv) > 1 and sys.argv[1] == "probe":
        probe()
    elif len(sys.argv) > 1 and sys.argv[1] == "repeat":         # repeat park_street s:4.6 5
        main(cases=[(sys.argv[2], sys.argv[3])] * int(sys.argv[4]))
    else:
        main(quick=len(sys.argv) > 1 and sys.argv[1] == "quick")
