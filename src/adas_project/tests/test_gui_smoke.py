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


if __name__ == "__main__":
    unittest.main()
