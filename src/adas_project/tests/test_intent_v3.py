"""v3 driver-intent pieces (adas/intent_net.py, sim/train_intent_torch.py, sim/twin_intent_data.py): the car's numpy
evaluators give exactly the training numbers, the physics floor, the tick features, and the twin data labels."""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from adas.intent_net import (HORIZON3, TICK_DIMS, WINDOW, Z_FREE_NOW, Z_STICK, Z_STOP, Z_THR, Z_V, IntentNet,
                             flat_features, flat_features_batch, physics_floor, physics_floor_batch, tick_vector,
                             window_of)


def random_windows(n, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.uniform(0, 1, (n, WINDOW, TICK_DIMS))
    W[:, :, Z_STICK] = np.round(rng.normal(0, 0.3, (n, WINDOW)) * 2) / 2 * (rng.random((n, 1)) < 0.5)
    W[:, :, Z_THR] = rng.uniform(-1, 1, (n, 1)) * (1 + 0.1 * rng.normal(size=(n, WINDOW)))
    W[:, :, Z_V] = rng.uniform(-0.8, 0.8, (n, WINDOW))
    return W


class FeaturesTest(unittest.TestCase):
    def test_batch_equals_single(self):
        W = random_windows(200)
        B = flat_features_batch(W)
        for i in range(0, 200, 7):
            np.testing.assert_allclose(B[i], flat_features(W[i]), atol=1e-12)
            self.assertEqual(physics_floor_batch(W[i:i + 1])[0], physics_floor(W[i]))

    def test_window_padding(self):
        z = [np.full(TICK_DIMS, k, float) for k in range(5)]
        W = window_of(z)
        self.assertEqual(W.shape, (WINDOW, TICK_DIMS))
        self.assertTrue((W[:WINDOW - 5] == 0).all() and W[-1, 0] == 4)

    def test_tick_vector_sees_far_and_backwards(self):
        """v2 saturated at 1.5 m ahead; v3 sees 2.5 m, and behind the car when reversing."""
        from adas.config import load_tuning
        from pi.relay_assists import K_CURV_PER_SERVO_DEG as K, car_params
        tun = load_tuning(os.path.join(os.path.dirname(__file__), "..", "pi", "tuning_real_car.json"))
        p = car_params(tun.mount)
        c = float(tun.servo.left_center)
        ys = np.linspace(-0.6, 0.6, 61)
        wall_ahead = np.column_stack([np.full_like(ys, 2.2), ys])
        z = tick_vector(c, 200, 0.8, wall_ahead, p, c, K)
        free = z[Z_FREE_NOW] * HORIZON3
        self.assertTrue(1.5 < free < 2.3, free)                    # beyond v2's 1.5 m cap
        wall_behind = np.column_stack([np.full_like(ys, -0.6), ys])
        z = tick_vector(c, -150, -0.5, wall_behind, p, c, K)
        self.assertLess(z[Z_STOP] * 5.0, 5.0)                      # direction of travel = backwards
        self.assertLess(z[8] * HORIZON3, 0.8)


class PhysicsFloorTest(unittest.TestCase):
    def window(self, free_m, v, thr, stick_moving=False):
        W = np.zeros((WINDOW, TICK_DIMS))
        W[:, Z_V], W[:, Z_THR] = v, thr
        stop = 0.05 + abs(v) * 0.2 + v * v / 8.0
        W[:, Z_STOP] = min(free_m / stop, 5.0) / 5.0
        if stick_moving:
            W[-8:, Z_STICK] = np.linspace(0, 0.5, 8)
        return W

    def test_frozen_driver_inside_stopping_distance(self):
        self.assertEqual(physics_floor(self.window(0.15, 0.8, 0.9)), 0.95)

    def test_not_when_room_to_stop_braking_or_steering(self):
        self.assertEqual(physics_floor(self.window(1.0, 0.8, 0.9)), 0.0)      # room to stop
        W = self.window(0.15, 0.8, 0.9)
        W[-1, Z_THR] = 0.0
        self.assertEqual(physics_floor(W), 0.0)                             # throttle released
        self.assertEqual(physics_floor(self.window(0.15, 0.8, 0.9, stick_moving=True)), 0.0)
        self.assertEqual(physics_floor(self.window(0.05, 0.05, 0.3)), 0.0)    # barely moving


class ExportTest(unittest.TestCase):
    """The numpy evaluators on the car reproduce PyTorch exactly."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except Exception:
            raise unittest.SkipTest("PyTorch not installed")

    def roundtrip(self, kind):
        import torch
        from sim.train_intent_torch import export_gru, export_mlp, torch_models
        torch.manual_seed(0)
        _, _, MLP, GRUNet = torch_models()
        W = random_windows(20, seed=1)
        if kind == "mlp":
            X = flat_features_batch(W)
            mu, sd = X.mean(0), X.std(0) + 1e-6
            model = MLP(X.shape[1], [32, 16], 0.0).eval()
            with torch.no_grad():
                ref = model(torch.tensor((X - mu) / sd, dtype=torch.float32)).numpy()
            ex = export_mlp(model, mu, sd, 1.7)
        else:
            flat = W.reshape(-1, TICK_DIMS)
            mu, sd = flat.mean(0), flat.std(0) + 1e-6
            model = GRUNet(TICK_DIMS, 16).eval()
            with torch.no_grad():
                ref = model(torch.tensor((W - mu) / sd, dtype=torch.float32)).numpy()
            ex = export_gru(model, mu, sd, 1.7)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            json.dump(ex, open(path, "w"))
            net = IntentNet(path)
        mine = np.array([net.logit3(w) for w in W])
        np.testing.assert_allclose(mine, ref / 1.7, atol=1e-4)
        p = net.risk(W[0], floor=False)
        self.assertAlmostEqual(p, 1 / (1 + math.exp(-ref[0] / 1.7)), places=4)

    def test_mlp(self):
        self.roundtrip("mlp")

    def test_gru(self):
        self.roundtrip("gru")


class TwinDataTest(unittest.TestCase):
    def test_frozen_full_throttle_at_a_wall_is_labelled_a_crash(self):
        """A scripted straight run with no reaction ends in contact, and the last 2 s before it are positives."""
        from sim.twin_intent_data import run
        for seed in range(40):
            rng = np.random.default_rng(seed)
            r = run((seed, "straight", 0.0, False))
            if r["style"] == "none" and r["crashed"]:
                y, tte = r["y"], r["tte"]
                self.assertEqual(y[-1], 1)
                self.assertTrue(np.all(y[tte <= 2.0] == 1))
                self.assertTrue(np.all(y[tte > 2.0] == 0))
                return
        self.fail("no frozen straight run crashed in 40 seeds")

    def test_randomisation_level_zero_is_the_nominal_twin(self):
        from sim.twin_intent_data import randomise
        a = randomise(np.random.default_rng(1), 0.0)
        b = randomise(np.random.default_rng(2), 0.0)
        self.assertEqual(a, b)
        wide = [randomise(np.random.default_rng(s), 1.6) for s in range(200)]
        self.assertTrue(all(w["noise"] >= 0 and w["dropout"] >= 0 and w["coast_decel"] >= 1.0 for w in wide))


if __name__ == "__main__":
    unittest.main()
