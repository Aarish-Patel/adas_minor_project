"""GUI smoke tests (offscreen): the EV window and both lab windows build, every drawer fits inside the window at common
screen sizes, and a live state (and 'no data') is drawn without exceptions (TODO J1, Q6, Q7)."""
import os
import sys
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

try:
    from PySide6 import QtCore, QtWidgets
    HAVE_QT = True
except ImportError:                                            # pragma: no cover
    HAVE_QT = False

STATE = {"mode": "normal", "follow_enabled": False, "_pts": [[a, 1.5] for a in range(-60, 61, 4)],
         "drive": {"v": 0.4, "pwm_in": 120, "servo": 90, "centre": 87, "steer_max_deg": 30},
         "assist": {"enabled": {"evasive": True, "moving": True}, "info": {}},
         "gate": {"action": "pass", "free_m": 1.4},
         "plan": {"state": "clear", "pred": [[0, 0.3], [0, 0.6], [0, 0.9]]}, "nav": {"state": "idle"},
         "intent": {"p": 0.2}, "world": {"pose": [0, 0, 0], "zones": [], "mode": "eco", "rss_min_m": 0.2},
         "health": {"state": "normal", "causes": []}, "esp32": {"state": "ok"}}


class Link:
    host = "127.0.0.1"

    def __init__(self, state=None):
        self.state = state

    def snapshot(self):
        import time
        return self.state, time.time(), [time.time() - 0.5, time.time()]

    def send(self, *a, **k):
        pass

    def __getattr__(self, name):
        return lambda *a, **k: None


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class GuiSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        from gui.controls import gui_thread_gc
        gui_thread_gc()                              # as the HMI does: garbage only collected on the GUI thread
        from gui import theme
        theme.set_mode("dark")
        cls.app.setStyleSheet(theme.STYLE)

    def test_drawers_fit_and_state_draws(self):
        from gui.ev import EVWindow
        w = EVWindow(Link(None))
        w.refresh()                                            # 'no data' state
        w.link.state = STATE
        w.refresh()                                            # live state
        for W, H in ((1366, 768), (1920, 1080), (1100, 700)):
            w.resize(W, H)
            w.show()
            self.app.processEvents()
            for key, d in w.docks.items():
                d.show()
                self.app.processEvents()
                self.assertLessEqual(w.width(), W + 2, f"{key} drawer widened the window at {W}x{H}")
                self.assertLessEqual(d.geometry().right(), w.rect().right() + 1, f"{key} drawer off-screen at {W}x{H}")
                self.assertGreater(w.centralWidget().height(), 300, f"{key} drawer squeezed the scene at {W}x{H}")
                d.hide()
        w.close()

    def test_car_render_stays_inside_the_real_footprint(self):
        """The hot-rod model with its LiDAR must not be bigger than the car the LiDAR and the gate see (rear .. front, width)."""
        from gui.dashboard import GEO
        from gui.ev import car_geometry
        lo = np.array([np.inf] * 3)
        hi = -lo
        for name, (V, F), col in car_geometry():
            lo, hi = np.minimum(lo, V.min(axis=0)), np.maximum(hi, V.max(axis=0))
            self.assertEqual(F.max() < len(V), True, name)
        eps = 1e-6
        self.assertGreaterEqual(lo[0], GEO["rear"] - eps)
        self.assertLessEqual(hi[0], GEO["front"] + eps)
        self.assertLessEqual(hi[1], GEO["width"] / 2 + eps)
        self.assertGreaterEqual(lo[1], -GEO["width"] / 2 - eps)
        self.assertGreaterEqual(lo[2], -eps)
        self.assertLess(hi[2], 0.25)                               # low: a hot rod, LiDAR on top

    def test_map_zoom_pan_fit(self):
        from gui.ev import EVWindow
        w = EVWindow(Link(STATE))
        w.show()
        mp = w.panels["map"][1]
        vb = mp.plot.getPlotItem().vb
        w0 = vb.viewRect().width()
        mp.zoom(2.0)
        self.assertLess(vb.viewRect().width(), w0 * 0.6)
        self.assertFalse(mp.follow_btn.isChecked())                # zooming by hand ends 'follow car'
        c0 = vb.viewRect().center().x()
        vb.translateBy(x=1.5, y=0)
        self.assertAlmostEqual(vb.viewRect().center().x(), c0 + 1.5, places=3)
        mp.fit()                                                   # frames the zones and the car
        r = vb.viewRect()
        for z in mp.zones:
            from gui.ev import zone_outline
            out = zone_outline(z)
            xs = -out[:, 1]
            self.assertLessEqual(r.left(), xs.min() + 1e-6)
            self.assertGreaterEqual(r.right(), xs.max() - 1e-6)
        w.close()

    def test_wheel_scrolls_the_drawer_and_ctrl_wheel_zooms_the_map(self):
        from PySide6 import QtGui
        from gui.ev import EVWindow
        w = EVWindow(Link(STATE))
        w.resize(1100, 700)
        w.show()
        w.toggle("map", True)
        self.app.processEvents()
        mp = w.panels["map"][1]
        sa = w.docks["map"].widget()
        sb = sa.verticalScrollBar()
        self.assertGreater(sb.maximum(), 0)                        # the drawer is taller than the window: it scrolls
        vp = mp.plot.viewport()
        pos = QtCore.QPointF(vp.width() / 2, vp.height() / 2)

        def wheel(mods):
            ev = QtGui.QWheelEvent(pos, vp.mapToGlobal(pos), QtCore.QPoint(0, 0), QtCore.QPoint(0, -120), QtCore.Qt.NoButton,
                                   mods, QtCore.Qt.NoScrollPhase, False)
            QtWidgets.QApplication.sendEvent(vp, ev)
        r0 = mp.plot.getPlotItem().vb.viewRect().width()
        v0 = sb.value()
        wheel(QtCore.Qt.NoModifier)
        self.app.processEvents()
        self.assertGreater(sb.value(), v0)                         # scrolled
        self.assertAlmostEqual(mp.plot.getPlotItem().vb.viewRect().width(), r0, places=6)   # not zoomed
        wheel(QtCore.Qt.ControlModifier)
        self.app.processEvents()
        self.assertNotAlmostEqual(mp.plot.getPlotItem().vb.viewRect().width(), r0, places=3)   # zoomed
        w.close()

    def test_theme_switch_rebuilds_the_window(self):
        from gui import theme
        from gui.ev import EVWindow
        w = EVWindow(Link(STATE))
        w.show()
        w.toggle_theme()
        self.app.processEvents()
        self.assertEqual(theme.MODE, "light")
        self.assertEqual(theme.C["bg"], theme.PALETTES["light"]["bg"])
        self.assertIsNot(self.app._ev_window, w)
        self.app._ev_window.refresh()
        self.app._ev_window.toggle_theme()                        # and back: leaves the module in dark for the other tests
        self.assertEqual(theme.MODE, "dark")
        self.app._ev_window.close()

    def test_lab_windows_build(self):
        from gui.lab_windows import MonteCarloWindow, TrainingWindow
        for cls in (MonteCarloWindow, TrainingWindow):
            w = cls()
            w.show()
            for _ in range(5):
                self.app.processEvents()
                w.stage.timer.timeout.emit()
            w.close()

    def test_lab_charts_and_empty_states(self):
        """Charts carry their legend on a pill above the plot (none inside), and a chart with no data shows its placeholder."""
        from gui.lab_windows import MonteCarloWindow, TrainingWindow
        w = TrainingWindow()
        w.show()
        self.app.processEvents()
        for card in (w.tab.c_loss, w.tab.c_ap):
            self.assertIsNone(card.plot.getPlotItem().legend)
        self.assertFalse(w.tab.c_loss.empty.isHidden() and not w.tab.curves)
        w.close()
        m = MonteCarloWindow()
        m.show()
        self.app.processEvents()
        self.assertGreaterEqual(len(m.legend.findChildren(QtWidgets.QLabel)), 8)      # dot + name per system
        self.assertIn("blocks", m.stage.scene._meshes)                              # obstacles are solid blocks
        m.close()

    def test_main_window_no_data_state(self):
        from gui.ev import EVWindow

        class Silent:
            host = "127.0.0.1"

            def snapshot(self):
                return None, 0.0, []
        w = EVWindow(Silent())
        w.show()
        w.refresh()
        self.assertTrue(any(it.visible() for it in w.scene.idle_rings))           # rings + pulse instead of a blank scene
        self.assertIn("Waiting", w.pill.head.text())
        self.assertIn("no data", w.intent_txt.text())
        w.mini.grab()                                                               # the radar placeholder paints
        w.close()

    def test_interpolator_is_continuous(self):
        """20 Hz packets of a car driving a curve, sampled at 60 Hz: no frame moves more than the true motion per frame."""
        import math
        from gui.render import StateInterpolator
        it = StateInterpolator()
        v, w = 1.2, 0.8                                                # m/s, rad/s
        pose = lambda t: (v / w * math.sin(w * t), v / w * (1 - math.cos(w * t)), w * t)
        prev, worst, t_end = None, 0.0, 3.0
        for k in range(int(t_end * 60)):
            now = k / 60.0
            for j in range(int(now * 20) + 1):                           # every packet sent so far has arrived
                if j / 20.0 > it.cur[0] - 1e-9 if it.cur else True:
                    it.push(j / 20.0, pose(j / 20.0))
            p = it.pose_at(now)
            if p is not None and prev is not None:
                worst = max(worst, math.hypot(p[0] - prev[0], p[1] - prev[1]))
            prev = p
        self.assertLess(worst, 1.6 * v / 60.0)                          # <= 1.6 x the real per-frame distance (no steps)

    def test_interpolator_late_packet_and_reset(self):
        from gui.render import StateInterpolator
        it = StateInterpolator()
        it.push(0.00, (0.0, 0.0, 0.0))
        it.push(0.05, (0.05, 0.0, 0.0))
        held = [it.pose_at(0.05 + 0.15 + d)[0] for d in (0.05, 0.5, 2.0)]   # the next packet never comes
        self.assertLess(held[2] - held[1], 1e-9)                          # extrapolation is capped, then it holds
        self.assertLess(held[2], 0.05 + 0.101)                            # ... and never runs away (cap = 0.1 s of motion)
        it.push(3.0, (5.0, 5.0, 1.0))                                     # an origin reset / restart: no glide across the room
        self.assertEqual(it.pose_at(3.0)[:2], (5.0, 5.0))

    def test_frame_clock_fixed_steps(self):
        from gui.render import FrameClock
        c = FrameClock(1.0 / 60.0)
        c.acc = 0.0
        c.t_last -= 0.0335                                               # 33.5 ms since the last frame = two fixed steps
        dt, n = c.tick()
        self.assertEqual(n, 2)
        self.assertGreater(c.fps, 0.0) if len(c.frames) > 1 else None

    def test_frame_loop_runs_and_reports_fps(self):
        from gui.ev import EVWindow

        class Moving:
            host = "127.0.0.1"

            def __init__(self):
                self.k = 0

            def send(self, cmd):
                pass

            def snapshot(self):
                import time
                self.k += 1
                x = 0.02 * self.k
                return ({"world": {"pose": [x, 0.0, 0.0], "zones": []}, "drive": {}, "_pts": np.empty((0, 2))},
                        time.time(), [time.time() - 0.05, time.time()])
        w = EVWindow(Moving())
        w.show()
        for _ in range(12):
            w.frame()
            self.app.processEvents()
        self.assertIsNotNone(w.interp.pose_at(__import__("time").time()))
        self.assertIn("packet", w.clock.profile())
        w.close()

    # --- Parking Simulator (TODO G3-2 .. G3-4, AB1: the street is one continuous lane, no spaces)
    def test_parking_worlds(self):
        """Both rooms build; only LiDAR-visible things are physical objects. Bays: every occupied bay holds a parked vehicle, every
        free one is empty. Street: no spaces at all - parked cars at different spacing along one lane with a depth."""
        from gui.parking import ParkingModel, corners
        from sim.hw_worlds import build
        from sim.parking_worlds import layout_for_hmi
        world, start = build("park_perp")
        m = ParkingModel(layout_for_hmi(world, start))
        self.assertEqual((len(m.spaces), sum(1 for s in m.spaces if not s["occupied"])), (5, 3))
        segs = np.asarray(world.segments(), float).reshape(-1, 4)
        pts = np.vstack([segs[:, :2], segs[:, 2:]]) - np.array(start[:2])
        for sp in m.spaces:
            c = corners(sp)
            inside = ((pts[:, 0] > c[:, 0].min() + 0.02) & (pts[:, 0] < c[:, 0].max() - 0.02) &
                      (pts[:, 1] > c[:, 1].min() + 0.02) & (pts[:, 1] < c[:, 1].max() - 0.02)).any()
            self.assertEqual(bool(inside), sp["occupied"], sp["id"])
        world, start = build("park_perp", marker_id=42)
        self.assertEqual(layout_for_hmi(world, start)["marker"]["id"], 42)          # the adjustable ArUco id
        world, start = build("park_street")
        st = ParkingModel(layout_for_hmi(world, start))
        self.assertTrue(st.street)
        self.assertEqual(st.spaces, [])                                             # no predefined spaces
        self.assertGreater(st.depth, 0.3)
        self.assertGreaterEqual(len(st.vehicles), 5)
        xs = sorted((v["cx"] - v["l"] / 2, v["cx"] + v["l"] / 2) for v in st.vehicles)
        gaps = {round(b[0] - a[1], 2) for a, b in zip(xs, xs[1:])}
        self.assertGreaterEqual(len(gaps), 4)                                       # the cars stand at different spacing

    def test_parking_click(self):
        from gui.parking import ParkingModel
        from sim.hw_worlds import build
        from sim.parking_worlds import layout_for_hmi
        m = ParkingModel(layout_for_hmi(*build("park_street")))                     # relay frame: start pose (0, -0.15)
        msg = m.click((1.0, 0.85))                                                  # in the lane, in the gap after the 2nd car
        self.assertIsNone(msg["selected_space_id"])                                 # the HMI does not choose: the car does
        self.assertAlmostEqual(msg["s"], 2.2, places=2)                             # position along the lane (strip starts at -1.2)
        self.assertAlmostEqual(msg["clicked_world_pose"][2], 0.0)                   # strip heading
        self.assertIsNotNone(m.click((1.0, 0.70)))                                  # the road edge itself counts
        self.assertIsNone(m.click((3.0, -0.6)))                                     # the far side of the road: not the lane
        self.assertIsNone(m.click((3.0, 1.40)))                                     # the sidewalk behind the lane
        g = m.ghost(msg["s"])
        self.assertAlmostEqual(g["cx"], 1.0, places=2)                              # the ghost sits where the user points
        self.assertTrue(0.70 < g["cy"] - g["w"] / 2 and g["cy"] + g["w"] / 2 < 0.70 + m.depth)   # inside the lane
        perp = ParkingModel(layout_for_hmi(*build("park_perp")))
        self.assertEqual(perp.click((-2.7, 0.46))["selected_space_id"], "P2")      # bays: on occupied P3 -> nearest free P2

    def test_parking_layout_frame(self):
        """Published in the relay's frame: origin and heading of the car's start pose."""
        import math
        from sim.parking_worlds import layout_for_hmi, perpendicular
        world, _ = perpendicular()
        a = layout_for_hmi(world, (0.0, 0.0, 0.0))
        b = layout_for_hmi(world, (1.0, 2.0, math.pi / 2))
        ca, cb = a["spaces"][0], b["spaces"][0]
        # a point (x, y) in the sim frame is at R(-th)(p - start) in the relay frame
        self.assertAlmostEqual(cb["cx"], (ca["cy"] - 2.0), places=2)
        self.assertAlmostEqual(cb["cy"], -(ca["cx"] - 1.0), places=2)
        self.assertAlmostEqual(cb["yaw"], ca["yaw"] - math.pi / 2, places=3)

    def test_parking_layout_follows_the_relays_drifted_frame(self):
        """The relay's estimated pose drifts away from the true pose (20 degrees in the twin after some driving). The layout is
        stored in the simulator frame; the HMI measures the frame change from the (true pose, estimated pose) pair, so a click on the
        lane IN THE RELAY'S FRAME lands on the lane, and the click and the strip go to the car in the relay's frame."""
        import json
        import math
        import time
        from gui.ev import EVWindow
        from sim.hw_worlds import build
        from sim.parking_worlds import layout_for_hmi
        lay = layout_for_hmi(build("park_street")[0], (0.0, 0.0, 0.0))                # simulator frame
        dth, tx, ty = math.radians(20.0), 0.4, -0.3                                     # the relay's frame is rotated and shifted
        sim_pose = (2.0, 0.5, 30.0)                                                     # x, y (m), deg
        c, s = math.cos(dth), math.sin(dth)
        est = [c * sim_pose[0] - s * sim_pose[1] + tx, s * sim_pose[0] + c * sim_pose[1] + ty, math.radians(30.0) + dth]

        class Link:
            host = "127.0.0.1"

            def __init__(self):
                self.sent = []

            def send(self, cmd):
                self.sent.append(cmd)

            def http_post(self, path):
                pass

            def snapshot(self):
                return ({"world": {"pose": est, "zones": []}, "drive": {}, "_pts": np.empty((0, 2)),
                         "sim": {"world": "park_street", "parking": lay, "walls": [], "pose": list(sim_pose)}},
                        time.time(), [time.time()])
        link = Link()
        w = EVWindow(link)
        w.show()
        w.refresh()
        for f in w._align_s:
            f.snap(f.target)
        mp = w.panels["map"][1]
        mp.set_park_align(tuple(f.value for f in w._align_s))
        self.assertAlmostEqual(mp.park_align[0], dth, places=3)
        px, py = 1.0, 0.75                                                              # a lane point in the simulator frame
        rx, ry = c * px - s * py + tx, s * px + c * py + ty                             # ... and in the relay's frame
        mp._park_click((rx, ry))
        self.assertAlmostEqual(mp.park_sel, 2.2, places=2)                              # on the lane, 2.2 m along it
        pose = w.parking_published["clicked_world_pose"]
        self.assertAlmostEqual(pose[0], rx, places=2)                                   # published in the relay's frame
        self.assertAlmostEqual(pose[2], dth, places=2)                                  # strip heading turned by the frame change
        time.sleep(0.2)
        cmd = next(x for x in link.sent if x.startswith("PARK_TARGET"))
        d = json.loads(cmd[len("PARK_TARGET"):])
        self.assertNotIn("spaces", d)                                                   # no spaces: the car finds the gaps
        x0, y0 = lay["strip"]["points"][0]
        self.assertAlmostEqual(d["strip"]["points"][0][0], c * x0 - s * y0 + tx, places=2)   # strip moved into the relay's frame
        self.assertAlmostEqual(d["strip"]["depth"], lay["strip"]["depth"], places=3)
        w.close()

    def test_parking_hmi_publishes_selection_and_draws_objects(self):
        import time
        from gui.ev import EVWindow
        from sim.hw_worlds import build
        from sim.parking_worlds import layout_for_hmi
        lay = layout_for_hmi(*build("park_street"))
        slot = [[1.2, 43.0], [1.5, 20.0], [1.4, 30.0], [1.1, 50.0]]                   # a chosen slot from the relay (polar corners)

        class Link:
            host = "127.0.0.1"

            def __init__(self):
                self.sent = []

            def send(self, cmd):
                self.sent.append(cmd)

            def http_post(self, path):
                self.sent.append(path)

            def snapshot(self):
                return ({"world": {"pose": [0.0, -0.15, 0.0], "zones": []}, "drive": {}, "_pts": np.empty((0, 2)),
                         "nav": {"state": "driving", "msg": "parking", "goal": [1.3, 35.0], "slot": slot},
                         "sim": {"world": "park_street", "parking": lay, "walls": []}}, time.time(), [time.time()])
        link = Link()
        w = EVWindow(link)
        w.show()
        w.refresh()
        mp = w.panels["map"][1]
        self.assertTrue(mp.park_card.isVisible() or not mp.isVisible())                # the card follows the simulator
        self.assertIsNotNone(w.parking_model)
        got = []
        mp.parking_selected.connect(got.append)
        mp._park_hover((1.0, 0.8))
        self.assertAlmostEqual(mp.park_hover, 2.2, places=2)                            # hover: a car-sized ghost on the lane
        mp._park_click((1.0, 0.8))
        self.assertAlmostEqual(got[-1]["s"], 2.2, places=2)
        self.assertEqual(w.parking_published["clicked_world_pose"][:2], [1.0, 0.8])
        time.sleep(0.2)
        self.assertTrue(any(c.startswith("PARK_TARGET") for c in link.sent))          # the click goes to the car at once
        w.toggle("map", True)                                                           # the map draws only while it is open
        w.refresh()
        keys = w.scene._park_keys
        self.assertTrue({"park_sel", "park_veh0", "park_road"} <= keys)                # ghost, parked cars and road are 3D objects
        self.assertTrue(w.scene._meshes["park_sel"].visible())
        self.assertTrue(mp.park_layer.ghosts["sel"].isVisible())                       # the click ghost on the map
        self.assertEqual(len(mp.slot.getData()[0]), 5)                                  # the car's chosen slot on the map
        self.assertTrue(w.scene._meshes["slot"].visible())                              # ... and in 3D
        mp._park_click((3.0, -0.6))                                                     # off the lane: no new target
        self.assertAlmostEqual(mp.park_sel, 2.2, places=2)
        mp._switch_world("park_perp")                                                   # the room switch asks the simulator
        time.sleep(0.7)
        self.assertIn("/api/sim/world/park_perp/23", link.sent)
        self.assertIn("ORIGIN", link.sent)                                              # and re-origins the relay's world frame
        w.close()

    def test_drawn_parking_lane(self):
        """AD1: a parking lane dragged as a rectangle on the map works without the simulator (the car): the far long side from the car
        is the kerb, the Park tool clicks on it and sends PARK_TARGET with the lane as drawn; flip / delete; saved to its own file."""
        import json
        import os
        import tempfile
        import time
        import gui.ev as EV
        from gui.parking import lane_geometry, lane_layout, make_lane
        # geometry: a lane along x, the car below it (y -0.5) -> road edge y 0.2, kerb y 0.6, the lane on the left of +x
        lane = make_lane((0.5, 0.6), (3.0, 0.2), (0.0, -0.5))
        pts, side, depth, kerb, _ = lane_geometry(lane)
        self.assertEqual((pts, side), ([[0.5, 0.2], [3.0, 0.2]], 1))
        self.assertAlmostEqual(depth, 0.4)
        self.assertEqual(kerb, [[0.5, 0.6], [3.0, 0.6]])
        lane_y = make_lane((1.0, 1.0), (1.4, 4.0), (0.0, 2.0))                     # along y, the car at smaller x: road edge x 1.0
        pts, side, depth, _, _ = lane_geometry(lane_y)
        self.assertEqual((pts[0][0], side), (1.0, -1))                                   # the lane lies right of +y (larger x)
        lay = lane_layout(lane, (0.3, 1.0, -0.5))                                        # moved into the simulator frame ...
        from gui.parking import sim_to_relay
        back = sim_to_relay(lay["strip"]["points"][0], (0.3, 1.0, -0.5))
        self.assertAlmostEqual(back[0], 0.5, places=3)                                   # ... and back into the relay's by PARK_TARGET
        self.assertAlmostEqual(back[1], 0.2, places=3)

        class Link:
            host = "127.0.0.1"

            def __init__(self):
                self.sent = []

            def send(self, cmd):
                self.sent.append(cmd)

            def http_post(self, path):
                self.sent.append(path)

            def snapshot(self):                                                          # the car: no simulator keys at all
                return ({"world": {"pose": [0.0, -0.5, 0.0], "zones": []}, "drive": {}, "_pts": np.empty((0, 2)), "nav": {}},
                        time.time(), [time.time()])
        saved = EV.LANE_FILE
        EV.LANE_FILE = os.path.join(tempfile.mkdtemp(), "lane.json")
        try:
            link = Link()
            w = EV.EVWindow(link)
            w.show()
            w.refresh()
            mp = w.panels["map"][1]
            self.assertIsNone(w.parking_model)                                          # nothing to park on yet
            mp.set_lane(lane)
            w.refresh()
            self.assertTrue(os.path.exists(EV.LANE_FILE))
            self.assertIsNotNone(w.parking_model)
            self.assertTrue(w.parking_model.user_lane and w.parking_model.street)
            mp._park_click((1.8, 0.4))                                                   # a click on the drawn lane
            time.sleep(0.2)
            cmds = [c for c in link.sent if c.startswith("PARK_TARGET")]
            self.assertTrue(cmds)
            d = json.loads(cmds[-1][len("PARK_TARGET"):])
            self.assertEqual(d["strip"]["points"], [[0.5, 0.2], [3.0, 0.2]])             # the lane as drawn (no simulator: no alignment)
            self.assertEqual((d["strip"]["side"], d["strip"]["depth"]), (1, 0.4))
            mp.flip_lane()
            self.assertEqual(mp.lane["road"], "y1")                                      # the kerb is now the near side
            mp.set_lane(None)
            w.refresh()
            self.assertIsNone(w.parking_model)
            self.assertFalse(os.path.exists(EV.LANE_FILE))
            w.close()
        finally:
            EV.LANE_FILE = saved


if __name__ == "__main__":
    unittest.main()
