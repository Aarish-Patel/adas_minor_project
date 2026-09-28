"""Rear-camera vision algorithms (adas/vision) checked against the synthetic camera's exact ground truth."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.markers import Camera, detect_image, project
from adas.vision.fusion import label_clusters
from adas.vision.guidelines import draw_guidelines
from adas.vision.quality import ImageQuality
from adas.vision.rear_objects import LoomingTracker, RearObjects
from adas.vision.rear_odometry import RearOdometry
from adas.vision.sim_camera import SimCamera

REAR = Camera(x=-0.05, z=0.12, pitch_deg=32.0, yaw_deg=180.0, hfov_deg=70.0)


def gray(f):
    import cv2
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


class OdometryTest(unittest.TestCase):
    def run_motion(self, v, w, n=40):
        sim = SimCamera(REAR, (320, 240), seed=3)
        vo = RearOdometry(sim.cam)
        x = y = th = t = 0.0
        dt, est = 1 / 15, []
        for i in range(n):
            r = vo.speed_yaw(sim.render((x, y, th)), t)
            if r is not None and i > 3:
                est.append(r)
            th += w * dt
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
            t += dt
        return np.median(np.array(est), axis=0)

    def test_forward_reverse_and_turning(self):
        for v, w in ((0.4, 0.0), (0.7, 0.0), (-0.3, 0.0), (0.5, 0.5), (0.5, -0.8)):
            ev, ew, q = self.run_motion(v, w)
            self.assertLess(abs(ev - v), max(0.02, 0.05 * abs(v)), (v, w, ev))
            self.assertLess(abs(ew - w), 0.06, (v, w, ew))
            self.assertGreater(q, 0.7)


class ObjectsTest(unittest.TestCase):
    def test_box_behind_is_found_by_parallax_and_looming_gives_time_to_contact(self):
        sim = SimCamera(REAR, (320, 240), seed=2)
        det, trk = RearObjects(sim.cam), LoomingTracker()
        box = (-1.35, 0.03, 0.22, 0.22, 0.16, 0.0)             # a 22 cm block, 1.35 m behind the start
        x, v, dt, t = 0.0, -0.3, 1 / 15, 0.0                    # reversing towards it at 0.3 m/s
        frames, ttcs, foots = [], [], []
        K = 6                                                    # a 0.4 s baseline: parallax of a low box is tiny per frame
        for i in range(50):
            g = gray(sim.render((x, 0.0, 0.0), boxes=[box]))
            frames.append(g)
            if i >= K:
                blobs = det.detect(frames[i - K], g, (v * dt * K, 0.0), 0.0)
                for tid, b, ttc in trk.update(blobs, t):
                    if ttc is not None:
                        gap = (x + REAR.x) - (box[0] + box[2] / 2)   # the CAMERA (5 cm behind the axle) to the block's near face
                        ttcs.append((ttc, -gap / v))
                    if b.foot is not None:
                        foots.append(b.foot)
            x, t = x + v * dt, t + dt
        self.assertGreater(len(foots), 10)
        self.assertGreater(len(ttcs), 5)
        est = np.array([a for a, b in ttcs])
        truth = np.array([b for a, b in ttcs])
        self.assertLess(np.mean(est[len(est) // 2:]), np.mean(est[:len(est) // 2]))     # it shrinks as it gets closer
        ok = truth > 0.4
        self.assertLess(np.median(np.abs(np.log(est[ok] / truth[ok]))), math.log(2.0), list(zip(est, truth))[-5:])  # < x2
        self.assertTrue(all(0.2 < a < 4.0 for a in est[-5:]))


class QualityTest(unittest.TestCase):
    def test_blur_is_detected(self):
        import cv2
        sim = SimCamera(REAR, (320, 240), seed=1)
        q = ImageQuality(sim.cam.fx)
        sharp = q.update(sim.render((0, 0, 0)), 0.0)
        blurry = ImageQuality(sim.cam.fx).update(cv2.GaussianBlur(sim.render((0, 0, 0)), (0, 0), 6.0), 0.0)
        self.assertLess(blurry["blur"], 0.3 * sharp["blur"])
        self.assertIn("blurred", ImageQuality(sim.cam.fx, blur_min=sharp["blur"] * 0.5).update(
            cv2.GaussianBlur(sim.render((0, 0, 0)), (0, 0), 6.0), 0.0)["why"])

    def test_vibration_amplitude_and_frequency_are_recovered(self):
        import cv2
        sim = SimCamera(REAR, (320, 240), seed=1, noise=1.0)
        base = sim.render((0.0, 0.0, 0.0))
        q = ImageQuality(sim.cam.fx)
        amp, f, fps = 2.0, 6.0, 30.0
        out = None
        for i in range(60):
            t = i / fps
            dy = amp * math.sin(2 * math.pi * f * t)
            M = np.float32([[1, 0, 0], [0, 1, dy]])
            out = q.update(cv2.warpAffine(base, M, (320, 240), borderMode=cv2.BORDER_REFLECT), t)
        self.assertLess(abs(out["jitter_px"] - amp / math.sqrt(2)), 0.5, out)     # RMS of a sine
        self.assertLess(abs(out["vib_hz"] - f), 1.6, out)


class MarkerTest(unittest.TestCase):
    def test_marker_behind_the_car_is_found_with_pose(self):
        cam = Camera(x=-0.05, z=0.12, pitch_deg=18.0, yaw_deg=180.0, hfov_deg=70.0)
        sim = SimCamera(cam, (640, 480), seed=1, noise=0.0)
        marker = (7, -0.80, 0.10, 0.09, 0.0, 0.10)               # facing +x (towards the car), 0.8 m behind the axle
        frame = sim.render((0.0, 0.0, 0.0), markers=[marker])
        obs = detect_image(frame, sim.cam, {7: 0.10})
        self.assertEqual([o.id for o in obs], [7])
        o = obs[0]
        self.assertLess(abs(o.x - (-0.80)), 0.04)
        self.assertLess(abs(o.y - 0.10), 0.04)


class FusionAndGuidelinesTest(unittest.TestCase):
    def test_lidar_cluster_takes_the_class_of_the_box_it_falls_in(self):
        cam = Camera(x=-0.05, z=0.12, pitch_deg=32.0, yaw_deg=180.0, hfov_deg=70.0)
        uv, _ = project(cam, np.array([[-0.7, 0.05, 0.06]]))
        u, v = uv[0]
        dets = [("person", 0.8, (int(u - 30), int(v - 60), 60, 90))]
        lab = label_clusters([(-0.7, 0.05, 0.05), (-0.5, -0.4, 0.05)], dets, cam)
        self.assertEqual(lab[0]["cls"], "person")
        self.assertGreater(lab[0]["margin"], 1.5)
        self.assertEqual(lab[1]["cls"], "unknown")

    def test_guidelines_draw_on_the_floor_inside_the_image(self):
        sim = SimCamera(REAR, (320, 240), seed=1)
        frame = sim.render((0, 0, 0))
        out = draw_guidelines(frame, sim.cam, kappa=0.8, width=0.20, bumper_x=-0.17)
        self.assertFalse(np.array_equal(out, frame))
        self.assertGreater(int(np.abs(out.astype(int) - frame.astype(int)).sum()), 5000)


if __name__ == "__main__":
    unittest.main()


class CalibrationTest(unittest.TestCase):
    """tools/camera_calibrate.py against the synthetic camera: a known mounting must come back."""

    def test_extrinsics_recover_height_pitch_yaw_and_position(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
        from camera_calibrate import extrinsics_from_board
        cam = Camera(x=-0.05, y=0.01, z=0.11, pitch_deg=38.0, yaw_deg=180.0, hfov_deg=70.0)
        sim = SimCamera(cam, (640, 480), seed=1, noise=0.0)
        sq = 0.028
        # a 9x6 inner-corner pattern has 10x7 squares; its first inner corner is one square in from the outer corner
        frame = sim.render((0.0, 0.0, 0.0), board=(-0.50 - sq, -0.11 - sq, 10, 7, sq))
        g = gray(frame)
        m = extrinsics_from_board(g, sim.cam.K, None, (-0.50, -0.11), 9, 6, sq)
        self.assertIsNotNone(m)
        self.assertLess(abs(m["z"] - 0.11), 0.012, m)
        self.assertLess(abs(m["pitch_deg"] - 38.0), 2.5, m)
        self.assertLess(abs(abs(m["yaw_deg"]) - 180.0), 3.0, m)
        self.assertLess(abs(m["x"] - (-0.05)), 0.03, m)
