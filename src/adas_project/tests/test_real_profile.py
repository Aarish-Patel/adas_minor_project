"""The simulator configured from the real car's measurements, checked against those measurements."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sim.real_car import real_profile, K_CURV_PER_SERVO_DEG, curvature_to_steer  # noqa: E402
from sim.session import Session  # noqa: E402
from sim.simulator import Simulator  # noqa: E402
from sim.lidar_sim import LidarSim  # noqa: E402
from sim.world import Cone, Wall, World  # noqa: E402


def sim_for(world, start=(0.0, 0.0, 0.0), mode="active"):
    rp = real_profile()
    return Simulator(world, start, adas_on=mode, params=rp.params, aeb_config=rp.aeb, dynamics=rp.dynamics,
                     adas_speed_model=rp.speed_model, lidar=LidarSim(seed=1, **rp.lidar_kw), seed=1), rp


class Geometry(unittest.TestCase):
    def test_extents_match_ruler_measurements(self):
        p = real_profile().params
        self.assertAlmostEqual(p.front_x - p.lidar_x, 0.16, places=3)
        self.assertAlmostEqual(p.lidar_x - p.rear_x, 0.17, places=3)
        self.assertAlmostEqual(p.width, 0.20, places=3)

    def test_steering_gain_matches_measured_curvature(self):
        rp = real_profile()
        from adas.vehicle_params import steer_to_delta
        s = curvature_to_steer(10 * K_CURV_PER_SERVO_DEG, rp.params)
        kappa = math.tan(steer_to_delta(s, rp.params)) / rp.params.wheelbase
        self.assertAlmostEqual(kappa, 0.68, delta=0.03)

    def test_measured_range_of_steering_is_reproduced(self):
        """The gain was measured up to about 24 servo degrees (radius ~0.6 m); full lock is an extrapolation."""
        rp = real_profile()
        from adas.vehicle_params import steer_to_delta
        s = curvature_to_steer(24 * K_CURV_PER_SERVO_DEG, rp.params)
        self.assertLess(s, 1.0)                        # inside the stick range
        kappa = math.tan(steer_to_delta(s, rp.params)) / rp.params.wheelbase
        self.assertAlmostEqual(1.0 / kappa, 0.61, delta=0.03)


class Braking(unittest.TestCase):
    def test_rolls_out_short_after_throttle_cut_like_the_car(self):
        """Measured on the car: cruise 0.25-0.36 m/s, cut throttle -> rolled 0-7 cm."""
        w = World()
        w.add(Wall(6.0, -1, 6.0, 1))
        sim, rp = sim_for(w, mode="off")
        while sim.car.v < 0.29 and sim.t < 4:
            sim.step(0.0, 140)
        x0, v0 = sim.car.x, sim.car.v
        for _ in range(300):
            sim.step(0.0, 0.0)
        self.assertGreater(v0, 0.2)
        self.assertLess(sim.car.x - x0, 0.10)

    def test_aeb_stops_before_wall_at_every_speed(self):
        for pwm in (100, 140, 180, 255):
            w = World()
            w.add(Wall(3.0, -1.0, 3.0, 1.0))
            sim, rp = sim_for(w)
            sim.run(lambda t, s: (0.0, pwm if t > 0.3 else 0.0), 8.0)
            self.assertFalse(sim.car.collided, f"crashed at PWM {pwm}")
            self.assertGreater(3.0 - (sim.car.x + rp.params.front_x), 0.01)

    def test_side_obstacle_does_not_cause_false_braking(self):
        w = World()
        w.add(Cone(1.5, 0.30, 0.03))
        w.add(Wall(4.5, -1, 4.5, 1))
        sim, _ = sim_for(w)
        sim.run(lambda t, s: (0.0, 140 if 0.3 <= t < 2.0 else 0.0), 3.0, stop_when_stopped=False)
        self.assertLess(sim.max_level, 3)


class SessionBypass(unittest.TestCase):
    def test_both_profiles_complete_the_bypass(self):
        for profile in ("sim", "real"):
            s = Session()
            s.profile = profile
            s.load("bypass")
            s.toggle_bypass()
            for _ in range(9000):
                s.tick(0.02)
                if s.bypass.done or s.sim.car.collided:
                    break
            c = s.sim.car
            self.assertFalse(c.collided, profile)
            self.assertEqual(s.bypass.state, "DONE", profile)
            self.assertLess(abs(c.y), 0.06, profile)
            self.assertLess(abs(math.degrees(c.theta)), 4.0, profile)
            self.assertGreater(c.x, 3.0, profile)


if __name__ == "__main__":
    unittest.main()
