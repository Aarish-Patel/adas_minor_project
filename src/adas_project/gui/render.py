"""Frame pacing and motion smoothing for the HMI (TODO G3-1).

The car sends its state at ~20 Hz; the screen is drawn at 60 Hz. Drawing the newest packet as it arrives makes everything
step at 20 Hz and jitter with the network. This module decouples the two, the way game engines and cockpit HMIs do:

  FrameClock         fixed-timestep accumulator (Fiedler, "Fix Your Timestep") + FPS / frame-time meter + per-stage profile
  StateInterpolator  buffers the last two poses and returns the pose one packet interval in the past, interpolated (entity
                     interpolation, Bernier / Valve "Source Multiplayer Networking"); holds or extrapolates briefly when a
                     packet is late
  Smooth / SmoothAngle  frame-rate independent exponential smoothing (critically damped follower) for cameras and needles

Nothing here draws or imports the car's runtime; it only turns packets into a continuous pose.
"""
import collections
import math
import time


def wrap_pi(a):
    return math.remainder(a, 2 * math.pi)


class Smooth:
    """Exponential follower: x moves toward the target with time constant tau seconds, independent of the frame rate."""

    def __init__(self, value=0.0, tau=0.12):
        self.value, self.target, self.tau = float(value), float(value), tau

    def snap(self, v):
        self.value = self.target = float(v)

    def step(self, dt, target=None):
        if target is not None:
            self.target = float(target)
        self.value += (self.target - self.value) * (1.0 - math.exp(-dt / self.tau))
        return self.value


class SmoothAngle(Smooth):
    def step(self, dt, target=None):
        if target is not None:
            self.target = self.value + wrap_pi(float(target) - self.value)
        self.value += (self.target - self.value) * (1.0 - math.exp(-dt / self.tau))
        return self.value


class StateInterpolator:
    """Entity interpolation of the car's world pose (x, y, theta).

    push(t_rx, pose) on every packet; pose_at(now) returns the pose for render time now - delay, where delay is 1.5 measured
    packet intervals: the last few packets are kept and the pair that brackets the render time is interpolated, so network
    jitter of up to half an interval never shows. If the newest packet is late the pose is extrapolated on the last velocity for
    at most `max_extrap` seconds, then held - never teleported.
    """

    KEEP = 5

    def __init__(self, max_extrap=0.10):
        self.buf = collections.deque(maxlen=self.KEEP)        # (t_rx, x, y, th), oldest first
        self.interval = 0.05
        self.max_extrap = max_extrap

    @property
    def cur(self):
        return self.buf[-1] if self.buf else None

    def reset(self):
        self.buf.clear()

    def push(self, t_rx, pose):
        x, y, th = (float(pose[0]), float(pose[1]), float(pose[2]))
        if self.buf:
            t, px, py, _ = self.buf[-1]
            dt = t_rx - t
            if dt <= 1e-4:
                return
            if dt > 1.0 or math.hypot(x - px, y - py) > 1.5:  # a restart / origin reset: do not glide across the room
                self.buf.clear()
            else:
                self.interval += 0.2 * (min(dt, 0.25) - self.interval)
        self.buf.append((t_rx, x, y, th))

    @property
    def delay(self):
        return min(0.15, max(0.04, 1.5 * self.interval))

    def pose_at(self, now):
        """(x, y, th) at render time; None before the first packet."""
        if not self.buf:
            return None
        if len(self.buf) == 1:
            return self.buf[0][1:]
        rt = now - self.delay
        if rt <= self.buf[0][0]:
            return self.buf[0][1:]
        for a, b in zip(self.buf, list(self.buf)[1:]):
            if a[0] <= rt <= b[0]:
                k = (rt - a[0]) / max(b[0] - a[0], 1e-4)
                return a[1] + (b[1] - a[1]) * k, a[2] + (b[2] - a[2]) * k, a[3] + wrap_pi(b[3] - a[3]) * k
        a, b = self.buf[-2], self.buf[-1]                     # past the newest packet: brief extrapolation, then hold
        span = max(b[0] - a[0], 1e-4)
        k = min(rt - b[0], self.max_extrap) / span
        return b[1] + (b[1] - a[1]) * k, b[2] + (b[2] - a[2]) * k, b[3] + wrap_pi(b[3] - a[3]) * k


class OneEuro:
    """The One Euro filter (Casiez, Roussel & Vogel, "1 Euro filter: a simple speed-based low-pass filter for noisy input in interactive
    systems", CHI 2012): a first-order low-pass whose cutoff rises with the (filtered) speed of the signal - `min_cutoff` Hz at rest kills
    jitter, + `beta` Hz per unit/s while it moves keeps the lag small. `update(x, dt)` returns the filtered value."""

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-4))

    def reset(self):
        self.x = self.dx = None

    def update(self, x, dt):
        if self.x is None or dt <= 0.0:
            self.x, self.dx = x, 0.0 if self.dx is None else self.dx
            return self.x
        dx = (x - self.x) / dt
        self.dx += self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * abs(self.dx)
        self.x += self._alpha(cutoff, dt) * (x - self.x)
        return self.x


class PoseSmoother:
    """The displayed ego pose: One Euro on x, y (m) and the unwrapped heading (rad). At rest the scan matcher's millimetre / tenth-of-a-
    degree corrections (10 Hz) would shake the car model and every camera that follows it; they are filtered out, while driving the
    cutoff rises with the speed so the car does not trail its own LiDAR points. A jump (reset, relocalisation) is taken at once."""

    JUMP_M, JUMP_RAD = 0.5, math.radians(30)

    def __init__(self, min_cutoff=0.6, beta=6.0, heading_min_cutoff=0.5, heading_beta=3.0):
        self.fx, self.fy = OneEuro(min_cutoff, beta), OneEuro(min_cutoff, beta)
        self.fth = OneEuro(heading_min_cutoff, heading_beta)
        self._th = None

    def reset(self):
        for f in (self.fx, self.fy, self.fth):
            f.reset()
        self._th = None

    def update(self, pose, dt):
        x, y, th = (float(pose[0]), float(pose[1]), float(pose[2]))
        if self._th is not None:
            th = self._th + wrap_pi(th - self._th)                # unwrapped
            if math.hypot(x - self.fx.x, y - self.fy.x) > self.JUMP_M or abs(th - self.fth.x) > self.JUMP_RAD:
                self.reset()
        self._th = th
        return self.fx.update(x, dt), self.fy.update(y, dt), wrap_pi(self.fth.update(th, dt))


class FrameClock:
    """Fixed-timestep accumulator and frame statistics.

    tick() returns (frame_dt, n_steps): the real elapsed time (clamped) and how many fixed steps of `step` seconds the
    simulation-side animations should advance (at most 4, so a stall never causes a spiral). fps is a 1 s moving average;
    mark(name) accumulates per-stage time for the profiler (profile() -> {stage: mean ms over the last ~1 s}).
    """

    def __init__(self, step=1.0 / 60.0):
        self.step = step
        self.t_last = time.perf_counter()
        self.acc = 0.0
        self.frames = collections.deque(maxlen=120)
        self.worst_ms = 0.0
        self._stage = collections.defaultdict(lambda: collections.deque(maxlen=60))
        self._t_mark = None

    def tick(self):
        now = time.perf_counter()
        dt = now - self.t_last
        self.t_last = now
        self.frames.append(now)
        dt = min(dt, 0.1)
        self.acc += dt
        n = 0
        while self.acc >= self.step and n < 4:
            self.acc -= self.step
            n += 1
        if n == 4:
            self.acc = 0.0
        return dt, n

    @property
    def fps(self):
        if len(self.frames) < 2:
            return 0.0
        span = self.frames[-1] - self.frames[0]
        return (len(self.frames) - 1) / span if span > 0 else 0.0

    @property
    def frame_ms(self):
        return 1000.0 / self.fps if self.fps else 0.0

    def begin(self):
        self._t_mark = time.perf_counter()

    def mark(self, name):
        now = time.perf_counter()
        if self._t_mark is not None:
            self._stage[name].append(1000.0 * (now - self._t_mark))
        self._t_mark = now

    def profile(self):
        return {k: sum(v) / len(v) for k, v in self._stage.items() if v}
