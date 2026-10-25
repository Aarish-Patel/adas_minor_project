"""Virtual bumper: contact with something the LiDAR cannot see, inferred from motion.

A dark, absorbing or very low object returns no LiDAR points, so neither the brake gate nor the planner knows it is there (incident
1 Oct 2026; the twin's dark box in ROOM). The car then pushes into it - the throttle is on, nothing brakes, but the scan-matched pose does
not move. That is the signature of a bumper hit: robot vacuums use a physical bumper for exactly this and mark the hit in their map, as
navigation stacks feed bumper / range contacts into an obstacle layer (ROS Nav2 range and bumper layers). Here the "bumper" is the pose:
throttle clearly above the dead-band for WINDOW_S while the pose moved less than MIN_MOVE_M and the gate was not braking -> contact on
the side the car was driving towards. The caller marks the bumper line in the obstacle memory (adas/memory.ObstacleMemory.add_bumper):
the gate then stops pushing (no hammering of the drivetrain) and the planners route round it.
"""
import math

WINDOW_S = 0.4          # pushing this long without moving = contact (5 scans at 12 Hz; pushes are often cut after ~0.6 s)
MIN_MOVE_M = 0.015      # moved less than this over the window
LAG_S = 0.15            # motor + servo lag before a commanded speed shows in the pose
RANGE_SCALE = 2.7       # a distance ahead / behind must change this much more than the pose (4 cm vs 1.5 cm) to count as motion
MIN_SCANS = 3           # new scan-matched poses needed in the window: without fresh matches (the matcher can skip scans) no
                        # judgement - a frozen pose looked like a stuck car while it was reversing into a space (25 Oct)
REVERSAL_SETTLE_S = 0.35  # after a reversal of the throttle the car rolls on, stops and comes back - net movement ~0 although it never
                          # stood still (false contacts at every parking-stroke reversal, 25 Oct): judged only after this
MIN_SPEED = 0.05        # the throttle commands at least this speed (m/s) - above the dead-band (~11 PWM); creeping pushes run at PWM 30-40


def _moved(a, b):
    """The largest change between two motion signatures: (x, y) of the scan-matched pose, optionally followed by the distances straight
    ahead / behind in the scan - along a featureless street the matcher does not see the car move but those distances do (25 Oct).
    Components that are not finite in both are skipped."""
    best = 0.0
    for i, (u, v) in enumerate(zip(a, b)):
        if u is not None and v is not None and math.isfinite(u) and math.isfinite(v):
            best = max(best, abs(u - v) / (1.0 if i < 2 else RANGE_SCALE))   # the distances jitter more (which returns fall in the cone)
    return best


class ContactDetector:
    def __init__(self, pwm_min, window_s=WINDOW_S, min_move_m=MIN_MOVE_M, speed_of=None):
        """speed_of(pwm) -> m/s (the car's throttle model): a slow creep gets the time it needs to move 2 x min_move_m plus LAG_S
        before it counts as stuck, so a slow start is not taken for a contact."""
        self.pwm_min, self.window_s, self.min_move_m, self.speed_of = pwm_min, window_s, min_move_m, speed_of
        self.since, self.ref, self.dir = None, None, 0
        self.contacts = 0

    def update(self, t, physical, pose, braking, stamp=None):
        """t (s), physical = throttle sent to the motor (PWM, + forward), pose = the motion signature (scan-matched x, y
        and optionally the distances straight ahead / behind, adas/contact._moved), braking = the gate is acting,
        stamp = the pose's scan counter: a pose dead-reckoned between scans moves with the commanded speed even when the car is stuck,
        so only the pose of a new scan counts (the last scan pose is kept otherwise). Returns +1 / -1 on a new contact ahead / behind, else 0."""
        fresh = False
        if stamp is not None:
            if stamp != getattr(self, "_stamp", None):
                self._stamp, self._scan_pose, fresh = stamp, pose, True
            pose = getattr(self, "_scan_pose", None)
        d = (physical > 0) - (physical < 0)
        if braking or abs(physical) < self.pwm_min or d == 0 or pose is None:
            self.since, self.ref = None, None                               # (dir kept: a reversal through zero is still a reversal)
            return 0
        if self.since is None or d != self.dir:
            settle = REVERSAL_SETTLE_S if self.dir == -d else 0.0           # a reversal (not a start from standstill): let it settle first
            self.since, self.ref, self.dir = t + settle, tuple(pose), d
            self.n_scans = 0
            return 0
        if t < self.since:
            self.ref = tuple(pose)                                          # settling after a reversal: follow the car
            return 0
        if _moved(pose, self.ref) > self.min_move_m:
            self.since, self.ref, self.n_scans = t, tuple(pose), 0              # it moves: no contact
            return 0
        if fresh or stamp is None:
            self.n_scans = getattr(self, "n_scans", 0) + 1
        window = self.window_s
        if self.speed_of is not None:
            window = max(window, 2.0 * self.min_move_m / max(abs(self.speed_of(abs(physical))), 0.02) + LAG_S)
        if t - self.since >= window and getattr(self, "n_scans", 0) >= MIN_SCANS:
            self.since, self.ref, self.n_scans = t, tuple(pose), 0          # (one report per window while it keeps pushing)
            self.contacts += 1
            return d
        return 0
