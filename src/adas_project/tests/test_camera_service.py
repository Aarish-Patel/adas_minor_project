"""The relay's supervisor for the marker camera service (pi/camera_service.py, TODO AP2)."""
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pi.camera_service as CS


class CameraServiceTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = os.path.join(self.tmp.name, "cam.log")
        self.old = CS.RESTART_AFTER_S
        CS.RESTART_AFTER_S = 0.3

    def tearDown(self):
        CS.RESTART_AFTER_S = self.old
        self.tmp.cleanup()

    def wait(self, cond, secs=10.0):
        end = time.time() + secs
        while time.time() < end and not cond():
            time.sleep(0.05)
        return cond()

    def test_starts_restarts_and_stops(self):
        svc = CS.CameraService({"autostart": True}, cmd=[sys.executable, "-c", "import time; time.sleep(0.6)"], log_path=self.log, device_check=False)
        self.assertTrue(svc.start())
        try:
            self.assertTrue(self.wait(lambda: svc.starts >= 3), f"started only {svc.starts} times")          # it exits after 0.6 s: started again
        finally:
            svc.stop()
        self.assertIsNotNone(svc.proc.poll())                                                                 # stopped with the relay

    def test_autostart_off_never_starts(self):
        svc = CS.CameraService({"autostart": False}, cmd=[sys.executable, "-c", "pass"], log_path=self.log, device_check=False)
        self.assertFalse(svc.start())
        time.sleep(0.5)
        self.assertEqual(svc.starts, 0)

    def test_no_webcam_no_start(self):
        svc = CS.CameraService({"autostart": True, "index": 99}, cmd=[sys.executable, "-c", "pass"], log_path=self.log)
        svc.start()
        time.sleep(1.5)
        svc.stop()
        self.assertEqual(svc.starts, 0)


if __name__ == "__main__":
    unittest.main()
