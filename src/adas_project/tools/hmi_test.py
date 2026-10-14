"""Test the features on the REAL dashboard (gui/ev.py EVWindow) against the twin relay, the way the user drives: clicks on the map
(real mouse events on the plot), tool tiles and room switches, the dead-man throttle held through the same UDP packets rc_controller.py
sends, and screenshots of the whole window at the key moments (TODO AC6).

    python tools/hmi_test.py [scenario ...] [--keep-relay]      -> reports/hmi_test/<scenario>_*.png + results.json

It starts its own twin relay (tools/sim_car.py via sim.parking_e2e serve) on the test ports UDP 4310 / HTTP 8190, so a relay and
dashboard the user has open on 4210 / 8090 stay untouched. Checks are made on the
simulator's ground truth (`sim.pose`, `sim.parking`) and on what the dashboard shows (the path drawn on the map).
"""
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "reports", "hmi_test")
CAR = dict(front_x=0.28, rear_x=-0.05, width=0.20)          # the relay's body (pi/relay_assists.car_params)


# ------------------------------------------------------------------------------------------------ the driver (RT / stick)
class Driver:
    """Sends the controller's packets at 20 Hz: `throttle` (PWM, + forward) and `servo` (deg, 87 = straight)."""

    def __init__(self, port=4310):
        self.port, self.throttle, self.servo, self.stop = port, 0, 87, False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while not self.stop:
            s = int(self.servo)
            self.sock.sendto(f"A {s} {s}\nM {-int(self.throttle)}\n".encode(), ("127.0.0.1", self.port))
            time.sleep(0.05)


def relay_running(port=8190):
    try:
        import urllib.request
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state", timeout=1.0).read()
        return True
    except Exception:
        return False


# ------------------------------------------------------------------------------------------------ geometry on the ground truth
def lane_frame(lay):
    """(ParkingStrip in the simulator frame, cars as (s0, s1)) from the simulator's own layout."""
    from adas.park_select import ParkingStrip
    st = lay["strip"]
    strip = ParkingStrip(st["points"], st["side"], st.get("depth") or 0.38)
    cars = []
    for v in lay.get("vehicles") or []:
        c, sn = abs(math.cos(v["yaw"])), abs(math.sin(v["yaw"]))
        half = 0.5 * (v["l"] * c + v["w"] * sn)
        s_c = strip.project((v["cx"], v["cy"]))[0]
        cars.append((s_c - half, s_c + half))
    return strip, sorted(cars)


def body_corners(pose_deg):
    x, y, th = pose_deg[0], pose_deg[1], math.radians(pose_deg[2])
    c, s = math.cos(th), math.sin(th)
    hw = CAR["width"] / 2
    return np.array([(x + c * lx - s * ly, y + s * lx + c * ly) for lx in (CAR["rear_x"], CAR["front_x"]) for ly in (-hw, hw)])


def parked_check(lay, pose_deg, click_s=None):
    """Where the car stands relative to the lane and the parked cars (truth): inside the lane band, gap to the kerb, gaps to the cars."""
    strip, cars = lane_frame(lay)
    cs = body_corners(pose_deg)
    s, y = strip.to_strip(cs)
    lane_th = strip.point_at(float(s.mean()))[2]
    head = abs(math.degrees(math.remainder(math.radians(pose_deg[2]) - lane_th, 2 * math.pi)))
    behind = [b for a, b in cars if b <= s.min() + 0.02]
    ahead = [a for a, b in cars if a >= s.max() - 0.02]
    out = {"inside_lane": bool((y >= -0.005).all() and (y <= strip.depth + 0.005).all()),
           "road_side_cm": round(float(y.min()) * 100, 1), "kerb_gap_cm": round(float(strip.depth - y.max()) * 100, 1),
           "heading_deg": round(head, 1), "s_mid": round(float(s.mean()), 2),
           "gap_behind_cm": round(float(s.min() - max(behind)) * 100, 1) if behind else None,
           "gap_ahead_cm": round(float(min(ahead) - s.max()) * 100, 1) if ahead else None,
           "overlaps_car": any(a < s.max() and b > s.min() for a, b in cars)}
    if click_s is not None:
        from adas.park_select import choose_slots
        from sim.parking_validation import _true_gaps
        from adas.vehicle_params import VehicleParams
        car = VehicleParams(wheelbase=0.20, width=0.20, front_overhang=0.08, rear_overhang=0.05)
        sl = choose_slots(_true_gaps(strip, cars), click_s, car, strip.depth)
        out["expected_s"] = round(sl[0].mid, 2) if sl else None
        out["in_expected_gap"] = bool(sl and sl[0].gap[0] - 0.02 <= s.min() and s.max() <= sl[0].gap[1] + 0.02)
    return out


# ------------------------------------------------------------------------------------------------ the test run
class HmiTest:
    def __init__(self, win, app, driver, name):
        self.win, self.app, self.driver, self.name = win, app, driver, name
        self.shots, self.log, self.result = [], [], {"scenario": name}
        self.mp = win.panels["map"][1]

    # --- helpers usable from scenarios
    def state(self):
        return self.win.link.latest or {}

    def shot(self, tag):
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, f"{self.name}_{len(self.shots):02d}_{tag}.png")
        self.win.grab().save(path)
        self.shots.append(path)
        return path

    def world_to_widget(self, wx, wy):
        from PySide6 import QtCore
        vb = self.mp.plot.getPlotItem().vb
        scene = vb.mapViewToScene(QtCore.QPointF(-wy, wx))
        return self.mp.plot.mapFromScene(scene)

    def click_map(self, wx, wy, button=None):
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        pos = self.world_to_widget(wx, wy)
        QTest.mouseClick(self.mp.plot.viewport(), button or QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, pos)

    def click_widget(self, w):
        from PySide6 import QtCore
        from PySide6.QtTest import QTest
        QTest.mouseClick(w, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, w.rect().center())

    def sim_to_relay(self, pt):
        """Simulator point -> relay frame with the EXACT alignment of the newest packet (the HMI eases its own over ~0.35 s: after a room
        switch it lags, and a lane drawn from it lands centimetres off - the test must place its clicks where it means them)."""
        from gui.parking import align_from_poses, sim_to_relay
        st = self.state()
        sp, wp = (st.get("sim") or {}).get("pose"), (st.get("world") or {}).get("pose")
        align = align_from_poses(sp, wp) if sp and wp else self.mp.park_align
        return sim_to_relay(pt, align)

    def show_range(self, pts_relay, pad=1.0):
        """Frame these relay-frame points on the map (as the user zooms / pans to where they want to click)."""
        P = np.asarray(pts_relay, float).reshape(-1, 2)
        xs, ys = -P[:, 1], P[:, 0]
        self.mp.follow_btn.setChecked(False)
        self.mp.plot.setXRange(xs.min() - pad, xs.max() + pad)
        self.mp.plot.setYRange(ys.min() - pad, ys.max() + pad)

    def note(self, **kw):
        self.log.append(dict(t=round(time.time() - self.t0, 1), **kw))


def belief_kerb_err_cm(st, lay):
    """Where the car BELIEVES its slot's kerb edge is (the stream's nav.slot, vehicle frame) minus the true kerb, in cm - both taken in the
    simulator frame through the true pose (0 = the car's picture of the lane is right). None without a slot / at a steep heading."""
    sl = (st.get("nav") or {}).get("slot")
    sim = st.get("sim") or {}
    if not sl or not sim.get("pose"):
        return None
    cx, cy, ch = sim["pose"]
    ch = math.radians(ch)
    if abs(ch) > 0.3:
        return None
    lx = 0.12
    P = np.array([(lx + d * math.cos(math.radians(a)), -d * math.sin(math.radians(a))) for a, d in sl])
    W = np.column_stack([cx + math.cos(ch) * P[:, 0] - math.sin(ch) * P[:, 1], cy + math.sin(ch) * P[:, 0] + math.cos(ch) * P[:, 1]])
    strip, _ = lane_frame(lay)
    _, yl = strip.to_strip(W)
    return round(100 * float(np.sort(yl)[-2:].mean() - strip.depth), 1)


def watch_path(test, s):
    """Every tick while the car is on autonomy: is the planned path in the stream, and drawn on the map?"""
    st = test.state()
    nav = st.get("nav") or {}
    if nav.get("state") == "driving":
        in_stream = bool((st.get("plan") or {}).get("maneuver"))
        xd = test.mp.path.getData()[0]
        drawn = xd is not None and len(xd) >= 2 and test.mp.path.isVisible()
        s["drive_ticks"] += 1
        s["path_stream"] += int(in_stream)
        s["path_drawn"] += int(bool(drawn))


WEAVE = [(1.0, 120, 107), (1.0, 120, 67), (1.0, 120, 107), (1.0, 120, 67), (2.0, 120, 87)]
TO_FAR_SIDE = [(1.2, 130, 110), (1.2, 130, 64), (2.5, 140, 87)]      # forward, across to the fence side of the road, on along it


def run_parking(test, s_click, lateral=0.5, manual=None, room="park_street", switch=True, pulse=None):
    """Generator scenario: switch to the street (switch=False: park again from where the car stands), drive by hand first (`manual`:
    [(seconds, throttle PWM, servo deg)]), pick the Park tool, click the lane at s_click (fraction `lateral` of the lane depth from the
    road edge) and hold RT (`pulse` = (on s, off s): released and pressed again as a driver does) until the car is done; check on the
    truth."""
    mp, win, drv = test.mp, test.win, test.driver
    win.toggle("map", True)
    yield 0.5
    if switch:
        test.click_widget(mp.park_seg.btns[room])
        yield 3.0
    st = test.state()
    for _ in range(40):
        st = test.state()
        if (st.get("sim") or {}).get("parking") and (st.get("sim") or {}).get("world") == room:
            break
        yield 0.2
    lay = st["sim"]["parking"]
    for dur, thr, servo in manual or []:                      # the user drives by hand first
        drv.throttle, drv.servo = thr, servo
        yield dur
    if manual:
        drv.throttle, drv.servo = 0, 87
        yield 1.5
        st = test.state()
        test.shot("after_manual")
    strip, cars = lane_frame(lay)
    x, y, th = strip.point_at(s_click)
    nx, ny = -math.sin(th) * strip.side, math.cos(th) * strip.side
    click_sim = (x + nx * strip.depth * lateral, y + ny * strip.depth * lateral)
    click_rel = test.sim_to_relay(click_sim)
    car_rel = (st.get("world") or {}).get("pose") or [0, 0, 0]
    test.show_range([click_rel, car_rel[:2], test.sim_to_relay(strip.point_at(max(0, s_click - 1.5))[:2]),
                     test.sim_to_relay(strip.point_at(s_click + 1.5)[:2])])
    test.click_widget(mp.mode["park"])
    yield 0.8
    test.shot("before_click")
    test.click_map(*click_rel)
    test.note(event="clicked", s=s_click, click_relay=[round(v, 3) for v in click_rel], hmi_sel=mp.park_sel)
    yield 0.5
    drv.throttle = 150                                          # RT held
    watch = {"drive_ticks": 0, "path_stream": 0, "path_drawn": 0}
    seen_active, t0, last_shot, msgs = False, time.time(), time.time(), []
    no_path_shot = False
    crashes0 = int((st.get("sim") or {}).get("crashes") or 0)
    while time.time() - t0 < (150 if pulse else 90):
        st = test.state()
        nav = st.get("nav") or {}
        if nav.get("msg") and (not msgs or msgs[-1] != nav.get("msg")):
            msgs.append(nav.get("msg"))
        if pulse:
            drv.throttle = 150 if (time.time() - t0) % sum(pulse) < pulse[0] else 0
        be = belief_kerb_err_cm(st, lay)
        if be is not None:
            watch.setdefault("belief_kerb_err_cm", []).append(be)
        n_drawn = watch["path_drawn"]
        watch_path(test, watch)
        if watch["drive_ticks"] and watch["path_drawn"] == n_drawn and not no_path_shot:
            no_path_shot = True                                 # evidence: the first tick driving without a path drawn
            test.shot("no_path")
        if nav.get("state") not in (None, "idle"):
            seen_active = True
        elif seen_active or time.time() - t0 > 6:
            break
        if time.time() - last_shot > 4.0:
            last_shot = time.time()
            test.shot("driving")
        yield 0.1
    drv.throttle = 0
    yield 1.5
    st = test.state()
    test.shot("end")
    test.result["t_s"] = round(time.time() - t0, 1)
    sim = st.get("sim") or {}
    res = parked_check(lay, sim["pose"], s_click)
    be = watch.pop("belief_kerb_err_cm", [])
    if be:                                   # the car's picture of the lane vs the truth: at the start, worst, at the end
        res["belief_kerb_err_cm"] = {"first": be[0], "worst": max(be, key=abs), "last": be[-1]}
    res.update(msgs=msgs, crashes=int(sim.get("crashes") or 0) - crashes0, **watch)
    res["path_shown_pct"] = round(100 * watch["path_drawn"] / max(watch["drive_ticks"], 1))
    # parked = the whole body inside the lane, not touching the kerb (>= 1 cm: the wheels must not clip it) and close to it (<= 8 cm)
    res["ok"] = bool(res["inside_lane"] and res["crashes"] == 0 and 1.0 <= res["kerb_gap_cm"] <= 8.0 and res["heading_deg"] <= 5.0
                     and res.get("in_expected_gap") and res["path_shown_pct"] >= 90 and msgs and msgs[-1].startswith("arrived"))
    test.result.update(res)


SCENARIOS = {
    "park_open": lambda t: run_parking(t, 2.2),                   # the middle of the 1.29 m gap
    "park_tight": lambda t: run_parking(t, 8.27),                  # the 0.86 m gap
    "park_far": lambda t: run_parking(t, 6.5),                     # the 1.89 m gap, far down the street
    "park_on_car": lambda t: run_parking(t, 3.1),                  # on a parked car: nearest viable spot
    "park_short_gap": lambda t: run_parking(t, 3.55),              # in the 0.50 m gap: nearest viable spot
    "park_after_driving": lambda t: run_parking(t, 4.65, manual=WEAVE),
    "park_again_elsewhere": lambda t: park_twice(t),             # parked in one gap, then a click far down the street
    "park_rt_pulses": lambda t: run_parking(t, 4.65, pulse=(2.0, 0.8)),
    "park_from_far_side": lambda t: run_parking(t, 2.2, manual=TO_FAR_SIDE),     # past the gap, on the fence side of the road
    "park_then_goto": lambda t: park_then_goto(t),                # parked, then click-to-go down the road: pull out first
    "bay_park": lambda t: bay_park(t),                            # BAYS: park at the ArUco marker
    "goto_and_home": lambda t: goto_and_home(t),                  # ROOM: go to through the doorway, then return to start
    "zone_limit": lambda t: zone_limit(t),                        # ROOM: a 5 km/h zone drawn on the map, driven through by hand
    "brake_wall": lambda t: brake_wall(t),                        # ROOM: full throttle at the end wall, no steering
    # AD1: a lane drawn as a rectangle - over the street's own lane (click in the 1.02 m gap), and along a room wall (a kerb the LiDAR sees)
    "lane_street": lambda t: drawn_lane(t, "park_street", (-1.2, 0.55), (9.2, 0.93), (3.45, 0.74)),
    "lane_room_wall": lambda t: drawn_lane(t, "room", (0.3, 1.2), (3.3, 1.6), (1.8, 1.4)),
}


def park_then_goto(test, goal_sim=(6.2, -0.35)):
    """Parked in a gap, then a click-to-go down the road (user, 30 Sep: 'make sure the car can get out of the parking as well during click
    and go'): it must pull out and arrive at the clicked point (truth within 15 cm), without touching anything."""
    yield from run_parking(test, 2.2)
    first = {k: test.result.get(k) for k in ("ok", "kerb_gap_cm", "heading_deg", "s_mid")}
    mp, drv = test.mp, test.driver
    st = test.state()
    crashes0 = int((st.get("sim") or {}).get("crashes") or 0)
    goal_rel = test.sim_to_relay(goal_sim)
    car_rel = (st.get("world") or {}).get("pose") or [0, 0, 0]
    test.show_range([goal_rel, car_rel[:2]])
    test.click_widget(mp.mode["goto"])
    yield 0.8
    test.click_map(*goal_rel)
    yield 0.5
    test.shot("goto_clicked")
    drv.throttle = 150
    t0, msgs, seen, last_shot = time.time(), [], False, time.time()
    while time.time() - t0 < 60:
        st = test.state()
        nav = st.get("nav") or {}
        if nav.get("msg") and (not msgs or msgs[-1] != nav.get("msg")):
            msgs.append(nav.get("msg"))
        if nav.get("state") not in (None, "idle"):
            seen = True
        elif seen or time.time() - t0 > 6:
            break
        if time.time() - last_shot > 4.0:
            last_shot = time.time()
            test.shot("goto_driving")
        yield 0.1
    drv.throttle = 0
    yield 1.5
    st = test.state()
    test.shot("goto_end")
    sim = st.get("sim") or {}
    x, y = sim["pose"][0], sim["pose"][1]
    err = math.hypot(x - goal_sim[0], y - goal_sim[1])
    test.result = {"scenario": test.name, "first": first, "goto_msgs": msgs, "goto_err_cm": round(err * 100, 1),
                   "crashes": int(sim.get("crashes") or 0) - crashes0,
                   "pulled_out": any("pulling out" in m for m in msgs),
                   "ok": bool(first.get("ok") and err < 0.15 and int(sim.get("crashes") or 0) == crashes0
                              and msgs and msgs[-1].startswith("arrived"))}


def _switch_room(test, room):
    mp = test.mp
    test.win.toggle("map", True)
    yield 0.5
    test.click_widget(mp.park_seg.btns[room])
    for _ in range(60):
        yield 0.2
        st = test.state()
        sim = st.get("sim") or {}
        if sim.get("world") == room and st.get("world", {}).get("pose") and math.hypot(*st["world"]["pose"][:2]) < 0.05:
            break
    yield 1.0


def _hold_until_idle(test, tag, t_max=60.0, throttle=150):
    """RT held until the autonomy has run and ended: (messages, crashes during it)."""
    drv = test.driver
    st = test.state()
    crashes0 = int((st.get("sim") or {}).get("crashes") or 0)
    drv.throttle = throttle
    t0, msgs, seen, last_shot = time.time(), [], False, time.time()
    watch = {"drive_ticks": 0, "path_stream": 0, "path_drawn": 0}
    while time.time() - t0 < t_max:
        st = test.state()
        nav = st.get("nav") or {}
        if nav.get("msg") and (not msgs or msgs[-1] != nav.get("msg")):
            msgs.append(nav.get("msg"))
        watch_path(test, watch)
        if nav.get("state") not in (None, "idle"):
            seen = True
        elif seen or time.time() - t0 > 6:
            break
        if time.time() - last_shot > 4.0:
            last_shot = time.time()
            test.shot(tag)
        yield 0.1
    drv.throttle = 0
    yield 1.5
    st = test.state()
    test.result.update(path_shown_pct=round(100 * watch["path_drawn"] / max(watch["drive_ticks"], 1)))
    return msgs, int((st.get("sim") or {}).get("crashes") or 0) - crashes0


def bay_park(test):
    """BAYS: 'Park at the marker' - reverse into the bay of the ArUco marker; truth: <= 5 cm, <= 5 deg (sim.parking_e2e's goal)."""
    yield from _switch_room(test, "park_perp")
    st = test.state()
    marker = st["sim"]["parking"]["marker"]
    test.shot("before")
    test.click_widget(test.mp.park_go)
    yield 0.5
    box = {}
    gen = _hold_until_idle(test, "driving")
    try:
        while True:
            yield next(gen)
    except StopIteration as e:
        box["out"] = e.value
    msgs, crashes = box["out"]
    test.shot("end")
    x, y, th = test.state()["sim"]["pose"]
    gx, gy = marker["x"] + 0.06 + 0.05, marker["y"]
    err, head = math.hypot(x - gx, y - gy), abs(math.remainder(math.radians(th), 2 * math.pi))
    test.result.update(msgs=msgs, crashes=crashes, pos_err_cm=round(err * 100, 1), head_deg=round(math.degrees(head), 1),
                       ok=bool(crashes == 0 and err <= 0.05 and math.degrees(head) <= 5 and msgs and msgs[-1].startswith("arrived")))


def goto_and_home(test, goal_sim=(4.2, -0.05)):
    """ROOM: Go to through the doorway, then 'Return to start'; truth: arrive within 10 cm, back at the start within 10 cm, no contact."""
    yield from _switch_room(test, "room")
    mp = test.mp
    from gui.parking import align_from_poses, sim_to_relay
    st0 = test.state()
    align0 = align_from_poses(st0["sim"]["pose"], st0["world"]["pose"])     # world frame vs truth at the start (drift = change of it)
    goal_rel = test.sim_to_relay(goal_sim)
    test.show_range([goal_rel, (0.0, 0.0)])
    test.click_widget(mp.mode["goto"])
    yield 0.8
    test.click_map(*goal_rel)
    yield 0.5
    test.shot("goto_clicked")
    box = {}
    gen = _hold_until_idle(test, "goto")
    try:
        while True:
            yield next(gen)
    except StopIteration as e:
        box["out"] = e.value
    msgs, crashes = box["out"]
    x, y, _ = test.state()["sim"]["pose"]
    e_goto = math.hypot(x - goal_sim[0], y - goal_sim[1])
    shown = test.result.get("path_shown_pct")
    test.shot("at_goal")
    st = test.state()
    tx, ty = sim_to_relay((x, y), align0)                                     # where the car truly is, in the start's world frame
    wx, wy = st["world"]["pose"][:2]
    test.result["loc_err_at_goal_cm"] = round(math.hypot(tx - wx, ty - wy) * 100, 1)
    home_tile = next(t for t in mp.findChildren(QtWidgetsFeatureTile()) if "Return to start" in t.name.text())
    test.click_widget(home_tile)
    yield 0.5
    gen = _hold_until_idle(test, "home", t_max=120)
    try:
        while True:
            yield next(gen)
    except StopIteration as e:
        box["out"] = e.value
    msgs2, crashes2 = box["out"]
    test.shot("home_end")
    x, y, _ = test.state()["sim"]["pose"]
    e_home = math.hypot(x, y)
    test.result["crash_log"] = (test.state().get("sim") or {}).get("crash_log")
    test.result.update(goto_msgs=msgs, goto_err_cm=round(e_goto * 100, 1), goto_path_shown_pct=shown, home_msgs=msgs2,
                       home_err_cm=round(e_home * 100, 1), crashes=crashes + crashes2,
                       ok=bool(crashes + crashes2 == 0 and e_goto <= 0.10 and e_home <= 0.10 and msgs and msgs[-1].startswith("arrived")
                               and msgs2 and msgs2[-1].startswith("arrived")))


def zone_limit(test, kph=5.0):
    """ROOM: a 5 km/h zone drawn with the Rectangle tool across the path, then driven through by hand at a high throttle; truth: the car's
    real speed in the zone (past its first 0.25 m) stays within 0.03 m/s of the limit (5 km/h full size = 0.099 m/s on the car)."""
    from PySide6 import QtCore
    from PySide6.QtTest import QTest
    yield from _switch_room(test, "room")
    mp, drv = test.mp, test.driver
    x0, x1, y0, y1 = 0.8, 2.6, -0.45, 0.15                              # simulator frame, across the straight path
    a, b = test.sim_to_relay((x0, y0)), test.sim_to_relay((x1, y1))
    test.show_range([a, b, (0.0, 0.0)])
    test.click_widget(mp.mode["rect"])
    mp.kph.setValue(kph)
    yield 0.5
    QTest.mousePress(mp.plot.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, test.world_to_widget(*a))
    QTest.mouseMove(mp.plot.viewport(), test.world_to_widget(*b))
    QTest.mouseRelease(mp.plot.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, test.world_to_widget(*b))
    yield 1.0
    test.shot("zone_drawn")
    limit = kph / 3.6 / 14.0
    speeds, t0 = [], time.time()
    drv.throttle, drv.servo = 200, 87
    while time.time() - t0 < 8.0:
        st = test.state()
        sim = st.get("sim") or {}
        x = sim.get("pose", [0])[0]
        if x0 + 0.25 <= x + 0.115 <= x1:
            speeds.append(float(sim.get("v_true") or 0.0))
        if x > 3.2:
            break
        yield 0.05
    drv.throttle = 0
    yield 1.0
    test.shot("end")
    mp.clear_zones()
    vmax = max(speeds) if speeds else None
    test.result.update(zone_limit_mps=round(limit, 3), zone_vmax_mps=None if vmax is None else round(vmax, 3), samples=len(speeds),
                       ok=bool(speeds and vmax <= limit + 0.03))


def front_gap_truth(st):
    """Free distance ahead of the front bumper to the simulator's true walls (sim.walls: polar, angle cw deg, range from the LiDAR), within
    the car's width, m."""
    best = math.inf
    for poly in (st.get("sim") or {}).get("walls") or []:
        for a, d in poly:
            x, y = d * math.cos(math.radians(a)), -d * math.sin(math.radians(a))
            if abs(y) < CAR["width"] / 2 and x > 0:
                best = min(best, x - (CAR["front_x"] - 0.12))
    return best


def brake_wall(test):
    """ROOM: full throttle straight ahead, hands off the stick; truth: no contact, stopped close to whatever is ahead (the end wall
    through the doorway, or the partition when the steering noise drifts the car off the doorway)."""
    yield from _switch_room(test, "room")
    drv = test.driver
    st = test.state()
    crashes0 = int((st.get("sim") or {}).get("crashes") or 0)
    drv.throttle, drv.servo = 255, 87
    t0, last_shot = time.time(), time.time()
    while time.time() - t0 < 12.0:
        if time.time() - last_shot > 2.0:
            last_shot = time.time()
            test.shot("driving")
        yield 0.1
    st = test.state()
    test.shot("end")
    drv.throttle = 0
    yield 1.0
    sim = st.get("sim") or {}
    gap = front_gap_truth(st)
    test.result.update(crashes=int(sim.get("crashes") or 0) - crashes0, stop_gap_cm=round(gap * 100, 1) if math.isfinite(gap) else None,
                       ok=bool(int(sim.get("crashes") or 0) == crashes0 and 0.0 < gap < 0.30))


def drawn_lane(test, room, corner_a, corner_b, click_sim, kerb="y1", truth_strip=None):
    """AD1: draw a parking lane with the Parking lane tool (a rectangle dragged on the map, simulator-frame corners a -> b), pick Park,
    click at click_sim and hold RT; truth: the whole body inside the drawn lane, 1-8 cm off its kerb, heading <= 5 deg, no contact."""
    from PySide6 import QtCore
    from PySide6.QtTest import QTest
    from adas.park_select import ParkingStrip
    yield from _switch_room(test, room)
    mp = test.mp
    a, b = test.sim_to_relay(corner_a), test.sim_to_relay(corner_b)
    test.show_range([a, b, (0.0, 0.0)])
    test.click_widget(mp.mode["lane"])
    yield 0.5
    QTest.mousePress(mp.plot.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, test.world_to_widget(*a))
    QTest.mouseMove(mp.plot.viewport(), test.world_to_widget(*b))
    QTest.mouseRelease(mp.plot.viewport(), QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, test.world_to_widget(*b))
    yield 1.0
    test.shot("lane_drawn")
    lane = dict(mp.lane or {})
    test.click_widget(mp.mode["park"])
    yield 0.6
    test.click_map(*test.sim_to_relay(click_sim))
    yield 0.5
    test.shot("clicked")
    box = {}
    gen = _hold_until_idle(test, "driving", t_max=120)
    try:
        while True:
            yield next(gen)
    except StopIteration as e:
        box["out"] = e.value
    msgs, crashes = box["out"]
    test.shot("end")
    x, y, th = test.state()["sim"]["pose"]
    # the drawn lane in the simulator frame (the test drew it from simulator corners)
    x0, x1 = sorted((corner_a[0], corner_b[0]))
    y0, y1 = sorted((corner_a[1], corner_b[1]))
    yk, yr = (y1, y0) if kerb == "y1" else (y0, y1)
    strip = ParkingStrip([[x0, yr], [x1, yr]], 1 if yk > yr else -1, abs(y1 - y0))
    cs = body_corners((x, y, th))
    s, yl = strip.to_strip(cs)
    lane_th = strip.point_at(float(s.mean()))[2]
    head = abs(math.degrees(math.remainder(math.radians(th) - lane_th, 2 * math.pi)))
    kerb_gap = strip.depth - float(yl.max())
    inside = bool((yl >= -0.005).all() and kerb_gap >= 0.01 and (s >= -0.01).all() and (s <= strip.length + 0.01).all())
    test.result.update(lane=lane, msgs=msgs, crashes=crashes, inside_lane=inside, kerb_gap_cm=round(kerb_gap * 100, 1),
                       heading_deg=round(head, 1),
                       ok=bool(inside and crashes == 0 and kerb_gap <= 0.08 and head <= 5.0 and msgs and msgs[-1].startswith("arrived")))
    mp.set_lane(None, note=False)


def QtWidgetsFeatureTile():
    from gui.controls import FeatureTile
    return FeatureTile


def park_twice(test):
    first = {}
    yield from run_parking(test, 2.2)
    first.update(test.result)
    test.result = {"scenario": test.name, "first": {k: first.get(k) for k in ("ok", "kerb_gap_cm", "heading_deg", "s_mid")}}
    yield from run_parking(test, 6.5, switch=False)
    test.result["ok"] = bool(test.result.get("ok") and first.get("ok"))


def main(argv):
    names = [a for a in argv if not a.startswith("--")] or list(SCENARIOS)
    relay = None
    if not relay_running():
        os.makedirs(OUT, exist_ok=True)
        relay = subprocess.Popen([sys.executable, "-m", "sim.parking_e2e", "serve"], cwd=ROOT,
                                 stdout=open(os.path.join(OUT, "relay.log"), "w"), stderr=subprocess.STDOUT)
        for _ in range(60):
            if relay_running():
                break
            time.sleep(0.5)
    import gui.dashboard as D
    D.CTRL_PORT, D.HTTP_PORT = 4310, 8190
    import shutil
    import gui.ev as EV
    os.makedirs(OUT, exist_ok=True)
    test_zones = os.path.join(OUT, "zones_test.json")          # the test's own zones file: the user's gui/zones.json is never written
    if os.path.exists(EV.ZONES_FILE):
        shutil.copy(EV.ZONES_FILE, test_zones)
    EV.ZONES_FILE = test_zones
    EV.LANE_FILE = os.path.join(OUT, "lane_test.json")          # ... nor gui/parking_lane.json
    from PySide6 import QtCore, QtWidgets
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from gui import theme
    from gui.controls import gui_thread_gc
    from gui.dashboard import Link
    from gui.ev import EVWindow
    gui_thread_gc()
    theme.load_mode()
    app.setStyleSheet(theme.STYLE)
    driver = Driver()
    win = EVWindow(Link("127.0.0.1"))
    win.resize(1700, 1000)
    win.show()
    results = []
    queue = list(names)
    box = {"gen": None, "test": None, "wake": time.time() + 4.0}

    def tick():
        if time.time() < box["wake"]:
            return
        if box["gen"] is None:
            if not queue:
                timer.stop()
                app.quit()
                return
            name = queue.pop(0)
            test = HmiTest(win, app, driver, name)
            test.t0 = time.time()
            box["test"], box["gen"] = test, SCENARIOS[name](test)
        try:
            box["wake"] = time.time() + float(next(box["gen"]))
        except StopIteration:
            r = box["test"].result
            r["shots"] = [os.path.relpath(p, ROOT) for p in box["test"].shots]
            r["log"] = box["test"].log
            results.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("shots", "log")}), flush=True)
            box["gen"], box["wake"] = None, time.time() + 1.0
            driver.throttle, driver.servo = 0, 87
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"scenario": box["test"].name, "ok": False, "error": repr(e)})
            box["gen"] = None
    timer = QtCore.QTimer()
    timer.timeout.connect(tick)
    timer.start(50)
    app.exec()
    driver.stop = True
    os.makedirs(OUT, exist_ok=True)
    json.dump(results, open(os.path.join(OUT, "results.json"), "w"), indent=1)
    print(f"{sum(bool(r.get('ok')) for r in results)}/{len(results)} passed")
    if relay is not None and "--keep-relay" not in argv:          # the relay and its planner worker processes
        subprocess.run(["taskkill", "/PID", str(relay.pid), "/T", "/F"], capture_output=True)


if __name__ == "__main__":
    main(sys.argv[1:])
