"""Physics-informed machine-learning model of the car: DC-motor physics + Ackermann steering geometry, with a
learned correction for whatever the physics misses. numpy only - it fits and runs on the Pi as well.

SPEED (Johnson DC motor through a gearbox):
    motor torque falls linearly with speed (back-EMF): tau = Kt/R * (V*u - Ke*w)
    road load: rolling/friction drag c0*sign(v) plus viscous drag c1*v
    =>  dv/dt = a*u - b*v - c*sign(v)          (u = PWM, physical, + forward)
  Three physical constants, fitted by least squares. Everything else follows from them:
    time constant 1/b, dead-band c/a, steady speed (a*u - c)/b, coast-down deceleration c + b*v.
STEERING (kinematic bicycle with Ackermann, wheelbase L known, 0.20 m):
    wheel angle   delta = g_side * (s - s0) + h * (s - s0)^3      servo linkage: centre s0, left/right gains, bend
    curvature     kappa = tan(delta) / (L + K_us * v^2)            K_us: understeer growing with speed
ML CORRECTION: kernel ridge regression (RBF) on the residuals the physics leaves, for acceleration (features
PWM, speed) and curvature (features servo, speed). An RBF correction fades to zero away from the training data,
so outside what was driven the model falls back to pure physics instead of extrapolating a curve-fit.
"""
import json
import math

import numpy as np

WHEELBASE = 0.20


class KernelRidge:
    """RBF kernel ridge regression, numpy only. Small: at most `max_centers` training points are kept."""

    def __init__(self, length_scales, lam=0.3, max_centers=250):
        self.ls = np.asarray(length_scales, float)
        self.lam, self.max_centers = lam, max_centers
        self.X = self.alpha = None

    def _k(self, A, B):
        d = (A[:, None, :] - B[None, :, :]) / self.ls
        return np.exp(-0.5 * (d * d).sum(-1))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if len(X) > self.max_centers:
            idx = np.linspace(0, len(X) - 1, self.max_centers).astype(int)
            X, y = X[idx], y[idx]
        K = self._k(X, X)
        self.X = X
        self.alpha = np.linalg.solve(K + self.lam * np.eye(len(X)), y)
        return self

    def predict(self, X):
        if self.X is None:
            return np.zeros(len(np.atleast_2d(X)))
        return self._k(np.atleast_2d(np.asarray(X, float)), self.X) @ self.alpha

    def to_dict(self):
        return {"ls": self.ls.tolist(), "lam": self.lam, "X": None if self.X is None else self.X.tolist(),
                "alpha": None if self.alpha is None else self.alpha.tolist()}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["ls"], d["lam"])
        if d.get("X") is not None:
            m.X, m.alpha = np.array(d["X"]), np.array(d["alpha"])
        return m


class CarModel:
    def __init__(self):
        # speed physics
        self.a, self.b, self.c = 0.0048, 5.0, 0.25          # a per PWM, b 1/s, c m/s^2  (placeholders)
        # steering physics
        self.s0, self.g_left, self.g_right, self.h, self.k_us = 87.0, 0.013, 0.013, 0.0, 0.0
        self.vss_ml = KernelRidge([0.20], lam=0.5)          # correction to the steady speed, feature PWM/255
        self.kap_ml = KernelRidge([14.0], lam=1.0)          # correction to the curvature, feature servo offset
        self.delay = 0.12
        self.use_ml = True
        self.report = {}

    # ------------------------------------------------------------------ speed
    def accel_physics(self, u, v):
        return self.a * u - self.b * v - self.c * np.sign(v) * (np.abs(v) > 1e-3)

    def vss_correction(self, u):
        u = np.atleast_1d(np.asarray(u, float))
        if not self.use_ml:
            return np.zeros_like(u)
        return np.where(u != 0, np.sign(u) * self.vss_ml.predict(np.abs(u)[:, None] / 255.0), 0.0)

    def accel(self, u, v):
        u, v = np.asarray(u, float), np.asarray(v, float)
        # physics + learned shift of the steady speed (the motor/battery curve the straight line misses)
        base = self.accel_physics(u, v) + self.b * self.vss_correction(u).reshape(np.shape(u))
        # static friction: a stopped car with too little PWM stays stopped
        stuck = (np.abs(v) < 1e-3) & (np.abs(self.a * u) <= self.c)
        return np.where(stuck, 0.0, base)

    def steady_speed(self, u):
        phys = max(0.0, (self.a * abs(u) - self.c) / self.b)
        return (phys + float(self.vss_correction(abs(u))[0]) if phys > 0 else 0.0) * (1 if u >= 0 else -1)

    def pwm_for_speed(self, v):
        """Inverse of the physics steady state (what the relay's speed caps need)."""
        if v <= 0:
            return 0.0
        return min(255.0, (self.b * v + self.c) / self.a)

    def deadband(self):
        return self.c / self.a

    def v_max(self):
        return self.steady_speed(255.0)

    def step_speed(self, v, u, dt):
        nv = v + float(self.accel(u, v)) * dt
        if u == 0 and v * nv < 0:
            nv = 0.0                                          # coasting never reverses the car
        return nv

    # ------------------------------------------------------------------ steering
    def wheel_angle(self, servo):
        d = np.asarray(servo, float) - self.s0
        g = np.where(d < 0, self.g_left, self.g_right)
        return g * d + self.h * d ** 3

    def curvature_physics(self, servo, v):
        """Path curvature, + = turning RIGHT (servo above the centre turns right on this car)."""
        return np.tan(self.wheel_angle(servo)) / (WHEELBASE + self.k_us * np.asarray(v, float) ** 2)

    def curvature(self, servo, v):
        base = self.curvature_physics(servo, v)
        if self.use_ml:
            s, _ = np.broadcast_arrays(np.asarray(servo, float) - self.s0, np.asarray(v, float))
            base = base + self.kap_ml.predict(s.ravel()[:, None]).reshape(np.shape(base))
        return base

    def servo_for_curvature(self, kappa_right, v=0.3):
        """Numerical inverse (monotonic in servo)."""
        grid = np.linspace(40, 140, 401)
        k = self.curvature(grid, np.full_like(grid, v))
        order = np.argsort(k)
        return float(np.interp(kappa_right, k[order], grid[order]))

    # ------------------------------------------------------------------ fitting
    def simulate_speed(self, t_eval, cmd_pwm, t0, v0=0.0, dt=0.01):
        ts = np.arange(t0, t_eval[-1] + dt, dt)
        i = np.searchsorted(cmd_pwm[:, 0], ts - self.delay, side="right") - 1
        u = np.where(i >= 0, cmd_pwm[np.clip(i, 0, len(cmd_pwm) - 1), 1], 0.0)
        vs = np.empty_like(ts)
        v = v0
        for k, uk in enumerate(u):
            v = self.step_speed(v, uk, dt)
            vs[k] = v
        return np.interp(t_eval, ts, vs)

    def fit_speed(self, t, v, good, cmd_pwm):
        """Johnson-motor ODE dv/dt = a*u - b*v - c*sign(v) fitted by simulating the logged commands and matching
        the measured speed (robust to noise, unlike fitting differentiated accelerations); command delay by grid.
        Then the ML: a smooth correction of the steady-speed curve from what is left at steady driving."""
        from scipy.optimize import least_squares
        t, v, good = np.asarray(t), np.asarray(v), np.asarray(good, bool)
        self.use_ml = False
        best = None
        for delay in np.arange(0.0, 0.26, 0.04):
            self.delay = delay

            def resid(p):
                self.a, self.b, self.c = p
                return (self.simulate_speed(t, cmd_pwm, t[0] - 0.5) - v)[good]
            r = least_squares(resid, x0=[0.005, 6.0, 0.3], bounds=([1e-4, 0.3, 0.0], [0.05, 60.0, 5.0]))
            if best is None or r.cost < best[0].cost:
                best = (r, delay)
        r, self.delay = best
        self.a, self.b, self.c = (float(x) for x in r.x)
        v_phys = self.simulate_speed(t, cmd_pwm, t[0] - 0.5)
        # steady samples: same command for >= 0.7 s and moving
        i = np.searchsorted(cmd_pwm[:, 0], t - self.delay, side="right") - 1
        u = np.where(i >= 0, cmd_pwm[np.clip(i, 0, len(cmd_pwm) - 1), 1], 0.0)
        since = np.zeros(len(t))
        for k in range(1, len(t)):
            since[k] = since[k - 1] + (t[k] - t[k - 1]) if u[k] == u[k - 1] else 0.0
        steady = good & (u != 0) & (since > 0.7) & (np.abs(v) > 0.05)
        if steady.sum() >= 8:
            self.vss_ml.fit(np.abs(u[steady])[:, None] / 255.0, np.sign(u[steady]) * (v[steady] - v_phys[steady]))
        self.use_ml = True
        v_ml = self.simulate_speed(t, cmd_pwm, t[0] - 0.5)
        self.report["speed"] = {"a": self.a, "b": self.b, "c": self.c, "delay_s": float(self.delay),
                                "time_constant_s": 1 / self.b, "deadband_pwm": self.deadband(), "v_max": self.v_max(),
                                "rms_speed_physics": float(np.sqrt(np.mean((v_phys - v)[good] ** 2))),
                                "rms_speed_physics_ml": float(np.sqrt(np.mean((v_ml - v)[good] ** 2))),
                                "n": int(good.sum()), "steady_samples": int(steady.sum())}

    def fit_steering(self, servo, v, kappa):
        """Gauss-Newton on the linkage + understeer parameters, then ML on the residuals."""
        servo, v, kappa = map(lambda z: np.asarray(z, float), (servo, v, kappa))
        p = np.array([self.s0, self.g_left, self.g_right, 0.0, 0.0])

        def model(p):
            s0, gl, gr, h, kus = p
            d = servo - s0
            delta = np.where(d < 0, gl, gr) * d + h * d ** 3
            return np.tan(delta) / (WHEELBASE + kus * v * v)
        for _ in range(40):
            r = kappa - model(p)
            J = np.empty((len(servo), 5))
            for i in range(5):
                dp = np.zeros(5)
                dp[i] = 1e-4 * max(1.0, abs(p[i]))
                J[:, i] = (model(p + dp) - model(p - dp)) / (2 * dp[i])
            prior = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 5.0])            # understeer pulled toward 0 unless needed
            step, *_ = np.linalg.lstsq(J.T @ J + prior, J.T @ r - prior @ np.array([0, 0, 0, 0, p[4]]), rcond=None)
            p = p + step
            p[4] = max(0.0, p[4])
            if np.abs(step).max() < 1e-7:
                break
        self.s0, self.g_left, self.g_right, self.h, self.k_us = (float(x) for x in p)
        res = kappa - self.curvature_physics(servo, v)
        self.kap_ml.fit((servo - self.s0)[:, None], res)
        full = self.curvature_physics(servo, v) + self.kap_ml.predict((servo - self.s0)[:, None])
        self.report["steering"] = {"servo_centre": self.s0, "gain_left_rad_per_deg": self.g_left,
                                   "gain_right_rad_per_deg": self.g_right, "cubic": self.h, "understeer_K": self.k_us,
                                   "min_radius_m": float(1 / max(abs(self.curvature_physics(140, 0.2)),
                                                                abs(self.curvature_physics(35, 0.2)))),
                                   "rms_kappa_physics": float(np.sqrt(np.mean(res ** 2))),
                                   "rms_kappa_physics_ml": float(np.sqrt(np.mean((kappa - full) ** 2))), "n": int(len(servo))}

    # ------------------------------------------------------------------ persistence
    def to_dict(self):
        return {"speed": {"a": self.a, "b": self.b, "c": self.c},
                "steering": {"s0": self.s0, "g_left": self.g_left, "g_right": self.g_right, "h": self.h, "k_us": self.k_us},
                "vss_ml": self.vss_ml.to_dict(), "kap_ml": self.kap_ml.to_dict(), "delay": self.delay,
                "report": self.report}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        m = cls()
        m.a, m.b, m.c = d["speed"]["a"], d["speed"]["b"], d["speed"]["c"]
        st = d["steering"]
        m.s0, m.g_left, m.g_right, m.h, m.k_us = st["s0"], st["g_left"], st["g_right"], st["h"], st["k_us"]
        m.vss_ml, m.kap_ml = KernelRidge.from_dict(d["vss_ml"]), KernelRidge.from_dict(d["kap_ml"])
        m.delay = d.get("delay", 0.12)
        m.report = d.get("report", {})
        return m


def training_data(path_rows, cmd_servo, cmd_pwm, delay=0.12):
    """From a reconstructed path (t, x, y, th, trusted; pi frame: th positive = right) and the logged commands:
    arrays for fitting. Speed and acceleration come from a smoothed derivative of the scan-matched path."""
    t, x, y, th, ok = path_rows.T
    # smooth positions over 5 scans, then differentiate
    k = np.ones(5) / 5
    xs, ys, ths = (np.convolve(z, k, mode="same") for z in (x, y, th))
    dt = np.gradient(t)
    vx, vy = np.gradient(xs) / dt, np.gradient(ys) / dt
    v = vx * np.cos(ths) + vy * np.sin(ths)
    w = np.gradient(ths) / dt
    acc = np.gradient(np.convolve(v, k, mode="same")) / dt

    def hold(series, tt):
        i = np.searchsorted(series[:, 0], tt, side="right") - 1
        return np.where(i >= 0, series[np.clip(i, 0, len(series) - 1), 1], series[0, 1])
    u = hold(cmd_pwm, t - delay)
    s = hold(cmd_servo, t - delay - 0.05)
    good = (ok > 0) & (np.convolve(ok, np.ones(5), mode="same") >= 5) & (dt > 0.02) & (dt < 0.4)
    good[:3] = good[-3:] = False
    turning = good & (np.abs(v) > 0.10)
    return {"u": u[good], "v": v[good], "acc": acc[good],
            "servo": s[turning], "v_turn": v[turning], "kappa": (w[turning] / v[turning])}
